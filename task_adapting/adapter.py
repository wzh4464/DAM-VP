from arguments import Arguments
import utils.logging as logging
from utils.train_utils import cosine_lr
from utils.functional import set_seed
from data_utils import loader as data_loader
import models.backbones.backbone_vit as bb
from models.builder import get_current_device
from models import prompters
import os
from statistics import mode
import sys
from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import os.path as osp
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.cluster as cluster

from utils.distributed import get_rank

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))


logger = logging.get_logger("dam-vp")


def hash_tensor(tensor, num_classes, seed=2024):
    """Hash the tensor to a class index.
    """
    import hashlib

    # 确保随机种子的一致性
    torch.manual_seed(seed)

    # 将张量扁平化为一维向量
    flattened_tensor = tensor.flatten()

    # 将张量转换为字节串
    tensor_bytes = flattened_tensor.to('cpu').numpy().tobytes()

    # 使用哈希函数（例如SHA256）
    hash_object = hashlib.sha256(tensor_bytes)
    hash_digest = hash_object.hexdigest()

    # 将哈希值转换为0到num_classes-1之间的整数
    hash_int = int(hash_digest, 16) % num_classes

    return hash_int

class Adapter(object):
    """A Gather of Our Task Adapting Methods.
    """

    def __init__(self, args: Arguments, model: nn.Module):
        super(Adapter, self).__init__()
        self.args = args
        self.model = model.eval()

        self.logger = logging.get_logger("dam-vp")

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)

        self.device = args.device
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.logger.info(f"self.args.device: {self.args.device}")
        self.logger.info(f"self.rank: {self.rank}")

        # destination to this gpu -- .to(self.devicename)
        self.devicename = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else get_current_device())
        self.logger.info(f"self.devicename: {self.devicename}")

        self.local_batch_size = args.batch_size // args.world_size

    def nums_of_learnable_params(self, model):
        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    def loss_function(self, logits, target):
        """Loss function to predict GT target.
        """
        loss = self.criterion(logits, target)
        return loss

    def load_prompter(self, prompter_path=None):
        """Load the trained visual prompter.
        """
        prompter = prompters.__dict__[self.args.prompt_method](
            self.args).to(self.devicename)
        if prompter_path is not None:
            checkpoint = torch.load(
                prompter_path, map_location=self.devicename)
            prompter.load_state_dict(checkpoint['state_dict'])
            logger.info(
                "Loading meta-trained visual prompts from {}".format(prompter_path))
        return prompter

    def get_active_neuron_index(self):
        """Search the most active neurons in the representaion.
        """
        set_seed(self.args.seed)
        with torch.no_grad():
            inPut = torch.randn(512, 3, self.args.crop_size,
                                self.args.crop_size).to(self.devicename)
            outPut = self.model.forward_features(inPut)  # [512, emd_dim]
            outPut = outPut.std(0, unbiased=False)  # [emd_dim]
            indices = outPut.sort(0, descending=True)[1]
            # indices = indices[:100]
            # indices = torch.unique(indices)
        return indices

    def rep2logit(self, output, num_classes):
        """Convert the output representations to logits. 
        把输出的表示转换为logits
        """
        if self.args.adapt_method == "prompt_wo_head":
            # activity aware
            indices = self.indices[:num_classes]
            indices = torch.unique(indices)
            logits = output[:, indices]
            # # scaled average voting
            # logits = F.avg_pool1d(output.unsqueeze(1), kernel_size=output.size(-1)//num_classes).squeeze(1)
            # logits = logits[:, :num_classes] if logits.size(-1) > num_classes else logits
        else:
            raise NotImplementedError
        return logits

    def get_prompted_image(self, image, prototype_gather=None, prompter=None, prompter_gather=None):
        """Obtain the prompted batch images.
        """
        if self.args.wo_da:
            assert prompter is not None
            prompted_image = prompter(image)
        else:
            assert prototype_gather is not None
            assert prompter_gather is not None
            prompted_image = []
            with torch.no_grad():
                rep_batch = self.model.forward_features(image)  # [N, emd_dim]
                rep_batch_sum = (rep_batch**2).sum(dim=-1,
                                                   keepdims=True)  # [N, 1]
                prototype_gather_sum = (
                    prototype_gather**2).sum(dim=-1, keepdims=True).T  # [1, M]
                distance_matrix = torch.sqrt(
                    rep_batch_sum + prototype_gather_sum - 2 * torch.mm(rep_batch, prototype_gather.T))  # [N, M]
                indices = torch.argmin(distance_matrix, dim=-1)  # [B]

            prompted_image = [
                prompter_gather[indices[idx]](image[idx].unsqueeze(0))
                for idx in range(rep_batch.size(0))
            ]
            return indices, torch.cat(prompted_image, dim=0)

    def coarse_clustering(self, data_loader):
        """Diversity-Aware Adaption on downstream data.
        We roughly divide the downstream task data into several partitions, each 
        partition presents a coarsely divided cluster. Different clusters correspond 
        to different prompters. 
        """

        # if saved exists, load it
        if osp.exists(f"{self.args.output_dir}/prototype_gather_{self.devicename}.pth") and osp.exists(f"{self.args.output_dir}/num_coarse_classes_{self.devicename}.pth"):
            logger.info(
                f"Loading prototype gather from {self.args.output_dir}/prototype_gather_{self.devicename}.pth")
            save_dict = torch.load(
                f"{self.args.output_dir}/prototype_gather_{self.devicename}.pth")
            self.prototype_gather = save_dict["prototype_gather"]
            self.num_coarse_classes = save_dict["num_coarse_classes"]
            return

        train_loader, _, _ = data_loader
        threshold_dict = {
            "resnet50-1k": 21,
            "vit-b-1k": 31,
            "vit-b-22k": 10,
            "swin-b-22k": 20,
            "moco-v3-b-1k": 18
        }
        hc = cluster.AgglomerativeClustering(
            n_clusters=None,
            linkage='average',
            distance_threshold=threshold_dict[self.args.pretrained_model]
        )
        with torch.no_grad():
            for i, sample in enumerate(train_loader):
                image = sample["image"].to(self.devicename)
                rep = self.model.forward_features(image)
                rep_gather = rep if i < 1 else torch.cat(
                    [rep_gather, rep], dim=0)

                if rep_gather.size(0) > 1000:
                    rep_gather = rep_gather[:1000]
                    break

        y_pred = hc.fit(rep_gather.detach().cpu().numpy()).labels_
        y_pred = torch.from_numpy(y_pred).to(self.devicename)
        coarse_class_idx = torch.unique(y_pred)
        self.num_coarse_classes = len(coarse_class_idx)
        logger.info(
            f"Nums of coarsely divided categories for test dataset {self.args.test_dataset}: {len(coarse_class_idx)}"
        )

        prototype_gather = []
        for i in range(len(coarse_class_idx)):
            pos = torch.where(y_pred == i)[0]
            prototype = rep_gather[pos].mean(0).unsqueeze(0)
            prototype_gather.append(prototype)
        self.prototype_gather = torch.cat(prototype_gather)
        logger.info(
            f"Nums of prototypes of coarse clusters for test dataset {self.args.test_dataset}: {self.prototype_gather.size(0)}"
        )
        save_dict = {
            "prototype_gather": self.prototype_gather,
            "num_coarse_classes": self.num_coarse_classes
        }
        # torch.save(save_dict, f"{self.args.output_dir}/prototype_gather_{self.devicename}.pth")


    def our_method(self, test_data, prompter_path):
        """Diversity-Aware Meta Visual Prompting (Head-Freezing/Missing Version).
        """
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)

        # register logger
        logger = self.logger

        self.model.discard_classifier()  # freeze the head
        self.indices = self.get_active_neuron_index()  # get the active neurons
        if not self.args.wo_da:
            self.coarse_clustering(test_data)

        if self.args.wo_da:
            # prompter = deepcopy(prompter)
            optimizer = torch.optim.SGD(
                prompter.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=0
            )
        else:
            prompter_gather, prompter_params_gather = [], []
            for i in range(self.num_coarse_classes):
                prompter_gather.append(
                    deepcopy(prompter)
                )
                prompter_params_gather.append(
                    {'params': prompter_gather[i].parameters()}
                )
            optimizer = torch.optim.SGD(
                prompter_params_gather,
                lr=self.lr,
                momentum=0.9,
                weight_decay=0
            )
        scheduler = cosine_lr(  # 依据cosine函数调整学习率
            optimizer,
            self.lr,
            len(train_loader)*self.args.epochs//5,
            len(train_loader)*self.args.epochs
        )

        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        BEST_ACC_VAL = -np.inf
        if self.args.wo_da:
            best_prompter = deepcopy(prompter)
        else:
            best_prompter_gather = deepcopy(prompter_gather)

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                global_step = len(train_loader) * epoch + i  # 整个训练过程中的step
                scheduler(global_step)  # 调整学习率
                image = sample["image"].to(self.devicename)
                label = sample["label"].to(self.devicename)
                _, prompted_image = self.get_prompted_image(image, prompter=prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                output = self.model.forward_features(prompted_image)
                logits = self.rep2logit(output, num_classes)
                loss = self.loss_function(logits, label)
                optimizer.zero_grad()
                loss.backward()
                # logger.info(prompter.pad_up.grad)
                optimizer.step()
                if (i + 1) % 1 == 0:
                    logger.info(
                        f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{i}/{len(train_loader)}], Training loss: {loss.item()}, device: {self.devicename}"
                    )
            # validate
            with torch.no_grad():
                num_total, correct = 0, 0
                for sample in val_loader:
                    image = sample["image"].to(self.devicename)
                    label = sample["label"].to(self.devicename)
                    _, prompted_image = self.get_prompted_image(image, prompter=prompter) \
                        if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                    output = self.model.forward_features(prompted_image)
                    logits = self.rep2logit(output, num_classes)
                    pred = torch.argmax(logits, dim=-1)
                    correct += (pred == label).sum().item()
                    num_total += image.size(0)
                acc_val = float(correct / num_total)
                logger.info(
                    "[Prompt Validating] Epoch: {}, Val acc: {}".format(epoch, acc_val))
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)

            # test
            # torch.save(best_prompter_gather,
            #     f"{self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(f"Saving best prompter gather to {self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            
            # test
            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    num_total, correct = 0, 0
                    for sample in test_loader:
                        image = sample["image"].to(self.devicename)
                        label = sample["label"].to(self.devicename)
                        _, prompted_image = self.get_prompted_image(image, prompter=best_prompter) \
                            if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=best_prompter_gather)
                        output = self.model.forward_features(prompted_image)
                        logits = self.rep2logit(output, num_classes)
                        pred = torch.argmax(logits, dim=-1)
                        correct += (pred == label).sum().item()
                        num_total += image.size(0)
                    acc_test = float(correct / num_total)
                    logger.info(
                        f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, device: {self.devicename}")
        return acc_test

    def our_method_with_head(self, test_data, prompter_path):
        """Diversity-Aware Meta Visual Prompting (Head-Tuning Version).
        """
        logger.info(f"self.devicename: {self.devicename}")
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.model.reset_classifier(num_classes)
        self.model.get_classifier().to(self.devicename)
        if not self.args.wo_da:
            self.coarse_clustering(test_data)

        self.model.get_classifier().train()
        if self.args.wo_da:
            # prompter = deepcopy(prompter)
            optimizer = torch.optim.SGD([
                {'params': prompter.parameters(), 'lr': self.lr, 'momentum': 0.9,
                 'weight_decay': self.weight_decay},
                {'params': self.model.get_classifier().parameters(), 'lr': 0.1,
                 'momentum': 0.9, 'weight_decay': 0}
            ])
        else:
            prompter_gather, prompter_params_gather = [], []
            for i in range(self.num_coarse_classes):
                prompter_gather.append(
                    deepcopy(prompter)
                )
                prompter_params_gather.append(
                    {'params': prompter_gather[i].parameters(
                    ), 'lr': self.lr, 'momentum': 0.9, 'weight_decay': self.weight_decay}
                )
            prompter_params_gather.append(
                {'params': self.model.get_classifier().parameters(), 'lr': 0.1,
                 'momentum': 0.9, 'weight_decay': 0}
            )
            optimizer = torch.optim.SGD(prompter_params_gather)
        scheduler = cosine_lr(
            optimizer,
            self.lr,
            len(train_loader)*self.args.epochs//5,
            len(train_loader)*self.args.epochs
        )

        BEST_ACC_VAL = -np.inf
        if self.args.wo_da:
            best_prompter = deepcopy(prompter)
        else:
            best_prompter_gather = deepcopy(prompter_gather)

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
                image = sample["image"].to(self.devicename)
                label = sample["label"].to(self.devicename)
                # label[:int(self.args.batch_size/10)] = int(torch.randint(0, num_classes, (1,)).item())
                _, prompted_image = self.get_prompted_image(image, prompter=prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                logits = self.model(prompted_image)
                loss = self.loss_function(logits, label)
                optimizer.zero_grad()
                loss.backward()
                # logger.info(prompter.pad_up.grad)
                optimizer.step()
                if (i + 1) % 1 == 0:
                    logger.info(f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{i}/{len(train_loader)}], Training loss: {loss.item()}, device: {self.devicename}")
            # validate
            with torch.no_grad():
                num_total, correct = 0, 0
                for sample in val_loader:
                    image = sample["image"].to(self.devicename)
                    label = sample["label"].to(self.devicename)
                    _, prompted_image = self.get_prompted_image(image, prompter=prompter) \
                        if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                    logits = self.model(prompted_image)
                    pred = torch.argmax(logits, dim=-1)
                    correct += (pred == label).sum().item()
                    num_total += image.size(0)
                acc_val = float(correct / num_total)
                logger.info(
                    f"[Prompt Validating] Epoch: {epoch}, Val acc: {acc_val}, device: {self.devicename}")
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)

            # test
            # torch.save(best_prompter_gather,
            #     f"{self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(f"Saving best prompter gather to {self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # save model head
            # self.head = nn.Linear(self.num_features, num_classes)
            # torch.save(self.model.get_classifier(),
            #            f"{self.args.output_dir}/head_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(f"Saving model head to {self.args.output_dir}/head_{self.devicename}_epoch_{epoch}.pth")
            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    num_total, correct = 0, 0
                    for sample in test_loader:
                        image = sample["image"].to(self.devicename)
                        label = sample["label"].to(self.devicename)
                        _, prompted_image = self.get_prompted_image(image, prompter=best_prompter) \
                            if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=best_prompter_gather)
                        logits = self.model(prompted_image)
                        pred = torch.argmax(logits, dim=-1)
                        correct += (pred == label).sum().item()
                        num_total += image.size(0)
                    acc_test = float(correct / num_total)
                    # logger.info("[Prompt Testing] Epoch: {}, Test acc: {}".format(epoch, acc_test))
                    logger.info(f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, device: {self.devicename}")
        return acc_test

    def our_method_with_mul_head(self, test_data, prompter_path):
        """Multi-Head for unlearning.
        """
        logger.info(f"Start our_method_with_mul_head on {self.devicename}")
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.model.reset_classifier(num_classes)

        if not self.args.wo_da:
            self.coarse_clustering(test_data)
            # updated self.num_coarse_classes
            # updated self.prototype_gather

        # self.model.get_classifier().to(self.devicename)
        # generate multi-head
        self.head_list = self.model.get_multi_classifier(
            self.num_coarse_classes)

        self.head_list = [head.to(self.devicename) for head in self.head_list]
        # self.head_list.train()
        for head in self.head_list:
            head.train()

        if self.args.wo_da:
            # prompter = deepcopy(prompter)
            optimizer = torch.optim.SGD([
                {'params': prompter.parameters(), 'lr': self.lr, 'momentum': 0.9,
                 'weight_decay': self.weight_decay},
                {'params': self.model.get_classifier().parameters(), 'lr': 0.1,
                 'momentum': 0.9, 'weight_decay': 0}
            ])
        else:
            # head_params = [
            #     param for head in self.head_list for param in head.parameters()]

            prompter_gather, prompter_params_gather = [], []
            for i in range(self.num_coarse_classes):
                prompter_gather.append(deepcopy(prompter))
                prompter_params_gather.append({
                    'params': prompter_gather[i].parameters(),
                    'lr': self.lr,
                    'momentum': 0.9,
                    'weight_decay': self.weight_decay
                })
                prompter_params_gather.append({
                    'params': self.head_list[i].parameters(),
                    'lr': 0.1,
                    'momentum': 0.9,
                    'weight_decay': 0
                })

            # 将多头部分类器的参数添加到优化器配置中
            # prompter_params_gather.append({
            #     'params': head_params,
            #     'lr': 0.1,
            #     'momentum': 0.9,
            #     'weight_decay': 0
            # })

            optimizer = torch.optim.SGD(prompter_params_gather)

        scheduler = cosine_lr(
            optimizer,
            self.lr,
            len(train_loader) * self.args.epochs // 5,
            len(train_loader) * self.args.epochs
        )

        BEST_ACC_VAL = -np.inf
        if self.args.wo_da:
            best_prompter = deepcopy(prompter)
        else:
            best_prompter_gather = deepcopy(prompter_gather)

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                global_step = len(train_loader) * epoch + i
                scheduler(global_step)
                image = sample["image"].to(self.devicename)
                label = sample["label"].to(self.devicename)
                # label[:int(self.args.batch_size/10)] = int(torch.randint(0, num_classes, (1,)).item())
                logits, loss = self.infer(
                    prompter, prompter_gather, image, label)

                optimizer.zero_grad()
                loss.backward()
                # logger.info(prompter.pad_up.grad)
                optimizer.step()
                if (i + 1) % 1 == 0:
                    act_batch_size = image.size(0)
                    pred = torch.argmax(logits, dim=-1)
                    correct = (pred == label).sum().item()
                    acc_train = float(correct / act_batch_size)
                    logger.info(
                        f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{i}/{len(train_loader)}], Training loss: {loss.item()}, Training acc: {acc_train}, device: {self.devicename}"
                    )

            with torch.no_grad():
                num_total, correct = 0, 0
                for sample in train_loader:
                    image = sample["image"].to(self.devicename)
                    label = sample["label"].to(self.devicename)
                    # prompted_image = self.get_prompted_image(image, prompter=prompter) \
                    #     if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                    # logits = self.model(prompted_image)

                    logits, loss = self.infer(
                        prompter, prompter_gather, image, label)

                    pred = torch.argmax(logits, dim=-1)
                    correct += (pred == label).sum().item()
                    num_total += image.size(0)
                acc_val = float(correct / num_total)
                logger.info(
                    f"[Prompt Training] Epoch: {epoch}, Train acc: {acc_val}, device: {self.devicename}")
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)

            # validate
            with torch.no_grad():
                num_total, correct, loss_total = 0, 0, 0
                for sample in val_loader:
                    image = sample["image"].to(self.devicename)
                    label = sample["label"].to(self.devicename)
                    logits, loss = self.infer(
                        prompter, prompter_gather, image, label)
                    loss_total += loss.item()

                    pred = torch.argmax(logits, dim=-1)
                    correct += (pred == label).sum().item()
                    num_total += image.size(0)
                acc_val = float(correct / num_total)
                loss_val = float(loss_total / len(val_loader))
                logger.info(
                    f"[Prompt Validating] Epoch: {epoch}, Val acc: {acc_val}, Val loss: {loss_val}, device: {self.devicename}")
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)

            # test
            # torch.save(best_prompter_gather,
            #            f"{self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(
            #     f"Saving best prompter gather to {self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # save model head
            # self.head = nn.Linear(self.num_features, num_classes)
            # torch.save(self.head_list,
            #            f"{self.args.output_dir}/head_list_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(
            #     f"Saving model head to {self.args.output_dir}/head_list_{self.devicename}_epoch_{epoch}.pth")

            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    num_total, correct, loss_total = 0, 0, 0
                    for sample in test_loader:
                        image = sample["image"].to(self.devicename)
                        label = sample["label"].to(self.devicename)
                        logits, loss = self.infer(
                            prompter, prompter_gather, image, label)
                        loss_total += loss.item()

                        pred = torch.argmax(logits, dim=-1)
                        correct += (pred == label).sum().item()
                        num_total += image.size(0)
                    acc_test = float(correct / num_total)
                    loss_test = float(loss_total / len(test_loader))
                    logger.info(
                        f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, Test loss: {loss_test}, device: {self.devicename}")
        return 0

    def random_part(self, test_data, prompter_path):
        """Randomly partition the images into several groups.
        """
        logger.info(f"Start random partition on {self.devicename}")
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.model.reset_classifier(num_classes)

        self.coarse_clustering(test_data)
        self.head_list = self.model.get_multi_classifier(
            self.num_coarse_classes)

        self.head_list = [head.to(self.devicename) for head in self.head_list]
        for head in self.head_list:
            head.train()
        prompter_gather, prompter_params_gather = [], []
        for i in range(self.num_coarse_classes):
            prompter_gather.append(deepcopy(prompter))
            prompter_params_gather.append({
                'params': prompter_gather[i].parameters(),
                'lr': self.lr,
                'momentum': 0.9,
                'weight_decay': self.weight_decay
            })
            prompter_params_gather.append({
                'params': self.head_list[i].parameters(),
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 0
            })

        optimizer = torch.optim.SGD(prompter_params_gather)

        scheduler = cosine_lr(
            optimizer,
            self.lr,
            len(train_loader) * self.args.epochs // 5,
            len(train_loader) * self.args.epochs
        )

        BEST_ACC_VAL = -np.inf

        for epoch in range(self.args.epochs):
            # train
            for i, sample in enumerate(train_loader):
                # adjust learning rate
                self.train_rand(train_loader, prompter, prompter_gather, i, optimizer, scheduler, epoch, sample)
            with torch.no_grad():
                acc_train, loss_train = self.evaluation_rand(train_loader, prompter, prompter_gather)
                logger.info(
                    f"[Prompt Training] Epoch: {epoch}, Train acc: {acc_train}, Train loss: {loss_train}, device: {self.devicename}")
                if acc_train > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_train
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)

            # validate
            with torch.no_grad():
                acc_val, loss_val = self.evaluation_rand(val_loader, prompter, prompter_gather)
                logger.info(
                    f"[Prompt Validating] Epoch: {epoch}, Val acc: {acc_val}, Val loss: {loss_val}, device: {self.devicename}")
                if acc_val > BEST_ACC_VAL:
                    BEST_ACC_VAL = acc_val
                    if self.args.wo_da:
                        best_prompter = deepcopy(prompter)
                    else:
                        best_prompter_gather = deepcopy(prompter_gather)

            # test
            # torch.save(best_prompter_gather,
            #            f"{self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(
            #     f"Saving best prompter gather to {self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")
            # save model head
            # self.head = nn.Linear(self.num_features, num_classes)
            # torch.save(self.head_list,
            #            f"{self.args.output_dir}/head_list_{self.devicename}_epoch_{epoch}.pth")
            # logger.info(
            #     f"Saving model head to {self.args.output_dir}/head_list_{self.devicename}_epoch_{epoch}.pth")

            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    acc_test, loss_test = self.evaluation_rand(test_loader, prompter, prompter_gather)
                    logger.info(
                        f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, Test loss: {loss_test}, device: {self.devicename}")
        return 0

    def train_rand(self, train_loader, prompter, prompter_gather, step, optimizer, scheduler, epoch, batch):
        global_step = len(train_loader) * epoch + step
        scheduler(global_step)
        image = batch["image"].to(self.devicename)
        label = batch["label"].to(self.devicename)
                # label[:int(self.args.batch_size/10)] = int(torch.randint(0, num_classes, (1,)).item())
        logits, loss = self.infer_rand(
                    prompter, prompter_gather, image, label)

        optimizer.zero_grad()
        loss.backward()
                # logger.info(prompter.pad_up.grad)
        optimizer.step()
        act_batch_size = image.size(0)
        pred = torch.argmax(logits, dim=-1)
        correct = (pred == label).sum().item()
        acc_train = float(correct / act_batch_size)
        logger.info(
                    f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{step}/{len(train_loader)}], Training loss: {loss.item()}, Training acc: {acc_train}, device: {self.devicename}"
                )

    def evaluation_rand(self, test_loader, prompter, prompter_gather):
        num_total, correct, loss_total = 0, 0, 0
        for sample in test_loader:
            image = sample["image"].to(self.devicename)
            label = sample["label"].to(self.devicename)
            logits, loss = self.infer_rand(
                            prompter, prompter_gather, image, label)
            loss_total += loss.item()

            pred = torch.argmax(logits, dim=-1)
            correct += (pred == label).sum().item()
            num_total += image.size(0)
        acc_test = float(correct / num_total)
        loss_test = float(loss_total / len(test_loader))
        return acc_test,loss_test

    def infer(self, prompter, prompter_gather, image, label):
        indices, prompted_image = self.get_prompted_image(image, prompter=prompter) \
            if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
        reps = self.model.forward_features(prompted_image)
        # logits = self.head_list[indices](reps)

        logits = torch.stack([self.head_list[indices[idx]](reps[idx])
                              for idx in range(len(indices))], dim=0)

        loss = self.loss_function(logits, label)
        return logits, loss

    def infer_rand(self, prompter, prompter_gather, image, label):
        indices, prompted_image = self.get_prompted_image_rand(
            image, prompter_gather=prompter_gather, seed=2024)

        reps = self.model.forward_features(prompted_image)
        # logits = self.head_list[indices](reps)

        logits = torch.stack([self.head_list[indices[idx]](reps[idx])
                              for idx in range(len(indices))], dim=0)

        loss = self.loss_function(logits, label)
        return logits, loss

    def get_prompted_image_rand(self, image, prompter_gather, seed=2024):
        """Obtain the prompted batch images.
        """

        assert self.num_coarse_classes == len(prompter_gather)

        hash_list = [
            hash_tensor(image[idx], len(prompter_gather), seed=seed)
            for idx in range(image.size(0))
        ]

        indices = torch.tensor(hash_list).to(self.devicename)

        prompted_image = [
            prompter_gather[indices[idx]](image[idx].unsqueeze(0))
            for idx in range(image.size(0))
        ]

        return indices, torch.cat(prompted_image, dim=0)
