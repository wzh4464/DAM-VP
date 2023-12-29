import os
import sys
import time
from typing import Any
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import sklearn.cluster as cluster

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from models import prompters
import models.backbones.backbone_vit as bb
from data_utils import loader as data_loader
from utils.functional import set_seed
from utils.train_utils import cosine_lr
import utils.logging as logging

from arguments import Arguments


logger = logging.get_logger("dam-vp")


class Adapter(object):
    """A Gather of Our Task Adapting Methods.
    """

    def __init__(self, args: Arguments, model: bb.VisionTransformer):
        super(Adapter, self).__init__()
        self.args = args
        self.model: bb.VisionTransformer = model.eval()

        self.logger = logging.get_logger("dam-vp")

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.criterion = torch.nn.CrossEntropyLoss().to(args.device)  # 交叉熵损失函数

        self.device = args.device
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.logger.info(f"self.args.device: {self.args.device}")
        self.logger.info(f"self.rank: {self.rank}")

        # destination to this gpu -- .to(self.devicename)
        self.devicename = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"self.devicename: {self.devicename}")

        self.local_batch_size = args.batch_size // args.world_size

    def nums_of_learnable_params(self, model):
        n_parameters = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

    def loss_function(self, logits, target):
        """Loss function to predict GT target.
        """
        return self.criterion(logits, target)

    def load_prompter(self, prompter_path=None):
        """Load the trained visual prompter.
        """
        prompter = prompters.__dict__[self.args.prompt_method](
            self.args).to(self.devicename)
        if prompter_path is not None:
            checkpoint = torch.load(prompter_path)
            prompter.load_state_dict(checkpoint['state_dict'])
            logger.info(
                f"Loading meta-trained visual prompts from {prompter_path}")
        return prompter

    def get_active_neuron_index(self):
        """Search the most active neurons in the representaion.
        """
        set_seed(self.args.seed)
        with torch.no_grad():
            inPut = torch.randn(self.local_batch_size, 3, self.args.crop_size,
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
        if self.args.adapt_method != "prompt_wo_head":
            raise NotImplementedError
        # activity aware
        indices = self.indices[:num_classes]
        indices = torch.unique(indices)
        return output[:, indices]

    def get_prompted_image(self, image, prototype_gather=None, prompter=None, prompter_gather=None):
        """Obtain the prompted batch images.
        """
        if self.args.wo_da:
            assert prompter is not None
            return prompter(image)
        else:
            assert prototype_gather is not None
            assert prompter_gather is not None
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
            return torch.cat(prompted_image, dim=0)

    def coarse_clustering(self, data_loader):
        """Diversity-Aware Adaption on downstream data.
        We roughly divide the downstream task data into several partitions, each 
        partition presents a coarsely divided cluster. Different clusters correspond 
        to different prompters. 

        updated attributes:
            self.num_coarse_classes: number of coarsely divided clusters
            self.prototype_gather: prototypes of each cluster
        """
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
                rep_gather = rep if i < 1 else torch.cat([rep_gather, rep], dim=0)  # noqa: F821

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

    def damvp_method(self, test_data, prompter_path):
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
            len(train_loader) * self.args.epochs // 5,
            len(train_loader) * self.args.epochs
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
                prompted_image = self.get_prompted_image(image, prompter=prompter) \
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
                    prompted_image = self.get_prompted_image(image, prompter=prompter) \
                        if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
                    output = self.model.forward_features(prompted_image)
                    logits = self.rep2logit(output, num_classes)
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
            if epoch > 0 and (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    num_total, correct = 0, 0
                    for sample in test_loader:
                        image = sample["image"].to(self.devicename)
                        label = sample["label"].to(self.devicename)
                        prompted_image = self.get_prompted_image(image, prompter=best_prompter) \
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

    def damvp_method_with_head(self, test_data, prompter_path):
        """Diversity-Aware Meta Visual Prompting (Head-Tuning Version).
        """
        logger, train_loader, val_loader, test_loader, prompter \
            = self.init_with_head_not_propmted(test_data, prompter_path)

        self.model.get_classifier().train()  # unfreeze the head
        best_prompter, optimizer, scheduler = self.make_opt_and_bpr(
            len(train_loader), prompter)
        BEST_ACC_VAL = -np.inf
        for epoch in range(self.args.epochs):
            # train
            self.training_part(logger, train_loader, best_prompter,
                               optimizer, scheduler, epoch, prompt_here=True)
            # validate
            best_prompter = self.make_validation(
                logger, val_loader, best_prompter, BEST_ACC_VAL, epoch)
            # test
            if epoch > 0 and (epoch + 1) % 5 == 0:
                acc_test = self.make_test(
                    logger, test_loader, epoch, best_prompter)
        return acc_test

    def init_with_head_not_propmted(self, test_data, prompter_path):
        """define logger, train_loader, val_loader, test_loader, prompter and do coarse clustering.
        # ! To be removed
        """
        logger = self.logger
        logger.info(f"self.devicename: {self.devicename}")
        train_loader, val_loader, test_loader = test_data
        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.model.reset_classifier(num_classes)
        self.model.get_classifier().to(self.devicename)
        if not self.args.wo_da:
            self.coarse_clustering(test_data)
        return logger,train_loader,val_loader,test_loader,prompter
    
    def init_with_head(self, test_data, prompter_path):
        """define logger, train_loader, val_loader, test_loader, prompter and do coarse clustering.
        * test_data is prompted here
        """
        logger = self.logger
        logger.info(f"self.devicename: {self.devicename}")

        train_loader_len = len(test_data[0])

        prompter = self.load_prompter(prompter_path)
        
        begin_time_cluster = time.time()

        if not self.args.wo_da:
            self.coarse_clustering(test_data)
            # prototype_gather updated here

        best_prompter, optimizer, scheduler = self.make_opt_and_bpr(
            train_loader_len, prompter)
        
        for loader in test_data:
            for data_item in loader:
                image = data_item["image"].to(self.devicename)
                # data_item = self.get_prompted_image(data_item, self.prototype_gather, prompter_gather=best_prompter) if not self.args.wo_da else self.get_prompted_image(data_item, prompter=prompter)
                data_item["image"] = self.get_prompted_image(image, self.prototype_gather, prompter_gather=best_prompter) if not self.args.wo_da else self.get_prompted_image(image, prompter=prompter)
        
        logger.info(f"Prompting time: {time.time() - begin_time_cluster}")

        return logger, test_data[0], test_data[1], test_data[2], best_prompter, optimizer, scheduler
    
    def training_part(self, logger, train_loader, prompter, optimizer, scheduler, epoch, prompt_here=False):
        for i, sample in enumerate(train_loader):
            # adjust learning rate
            global_step = len(train_loader) * epoch + i
            scheduler(global_step)
            image = sample["image"].to(self.devicename)
            label = sample["label"].to(self.devicename)

            if prompt_here:
                # training_part_dam specific code
                prompted_image = self.get_prompted_image(image, prompter=prompter) if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter)
                logits = self.model(prompted_image)
            else:
                # training_part specific code
                logits = self.model(image)

            loss = self.loss_function(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                logger.info(
                    f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{i}/{len(train_loader)}], Training loss: {loss.item()}, device: {self.devicename}"
                )


    def make_test(self, logger, test_loader, epoch, best_prompter):
        with torch.no_grad():
            num_total, correct = 0, 0
            for sample in test_loader:
                image = sample["image"].to(self.devicename)
                label = sample["label"].to(self.devicename)
                prompted_image = self.get_prompted_image(image, prompter=best_prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=best_prompter)
                logits = self.model(prompted_image)
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == label).sum().item()
                num_total += image.size(0)
            acc_test = float(correct / num_total)
            logger.info(
                f"[Prompt Testing] Epoch: {epoch}, Test acc: {acc_test}, device: {self.devicename}")
        return acc_test

    def make_validation(self, logger, val_loader, prompter, BEST_ACC_VAL, epoch):
        with torch.no_grad():
            num_total, correct = 0, 0
            for sample in val_loader:
                image = sample["image"].to(self.devicename)
                label = sample["label"].to(self.devicename)
                prompted_image = self.get_prompted_image(image, prompter=prompter) \
                    if self.args.wo_da else self.get_prompted_image(image, self.prototype_gather, prompter_gather=prompter)
                logits = self.model(prompted_image)
                pred = torch.argmax(logits, dim=-1)
                correct += (pred == label).sum().item()
                num_total += image.size(0)
            acc_val = float(correct / num_total)
            logger.info(
                f"[Prompt Validating] Epoch: {epoch}, Val acc: {acc_val}, device: {self.devicename}")
            if acc_val > BEST_ACC_VAL:
                BEST_ACC_VAL = acc_val
                best_prompter = deepcopy(prompter)
        return best_prompter

    def make_opt_and_bpr(self, train_loader_len, _prompter):
        if self.args.wo_da:
            # prompter = deepcopy(prompter)
            optimizer = torch.optim.SGD([
                {'params': _prompter.parameters(), 'lr': self.lr, 'momemtum': 0.9,
                 'weight_decay': self.weight_decay},
                {'params': self.model.get_classifier().parameters(), 'lr': 0.1,
                 'momemtum': 0.9, 'weight_decay': 0}
            ])
            prompter = _prompter
        else:
            prompter, prompter_params = [], []
            for i in range(self.num_coarse_classes):
                prompter.append(
                    deepcopy(_prompter)
                )
                prompter_params.append(
                    {'params': prompter[i].parameters(
                    ), 'lr': self.lr, 'momemtum': 0.9, 'weight_decay': self.weight_decay}
                )
            prompter_params.append(
                {'params': self.model.get_classifier().parameters(), 'lr': 0.1,
                 'momemtum': 0.9, 'weight_decay': 0}
            )
            optimizer = torch.optim.SGD(prompter_params)
        scheduler = cosine_lr(
            optimizer,
            self.lr,
            train_loader_len * self.args.epochs // 5,
            train_loader_len * self.args.epochs
        )

        return prompter, optimizer, scheduler

    def our_method_with_head(self, test_data, prompter_path):
        """Diversity-Aware Meta Visual Prompting (Head-Tuning Version) revised aiming at eraseing.

        1. Split the cluster and train the prompter for each cluster.
        """
        # AGGREATE_METHOD_DICT = {
        #     "nearest": Aggregate.nearest,
        #     "average": Aggregate.average,
        #     "majority": Aggregate.majority,
        #     "gaussian": Aggregate.gaussian
        # }

        logger, train_loader, val_loader, test_loader, best_prompter, optimizer, scheduler \
            = self.init_with_head(test_data, prompter_path)
        
        self.model.get_classifier().train()
        BEST_ACC_VAL = -np.inf

        # logger out: *** our method with head ***
        logger.info("*** our method with head ***")

        # label with cluster result
        for epoch in range(self.args.epochs):
            # train
            self.training_part(logger, train_loader, best_prompter,
                               optimizer, scheduler, epoch, prompt_here=False)
            # validate
            best_prompter = self.make_validation(
                logger, val_loader, best_prompter, BEST_ACC_VAL, epoch)
            # test
            if epoch > 0 and (epoch + 1) % 5 == 0:
                acc_test = self.make_test(
                    logger, test_loader, epoch, best_prompter)
        return acc_test

class Aggregate(object):
    """Aggregate the prompted images to the original image.
    """

    @staticmethod
    def nearest(prompted_image, image):
        """Nearest Neighbor.
        """
        return prompted_image

    @staticmethod
    def average(prompted_image, image):
        """Average.
        """
        return (prompted_image + image) / 2

    @staticmethod
    def majority(prompted_image, image):
        """Majority.
        """
        return prompted_image

    @staticmethod
    def gaussian(prompted_image, image):
        """Gaussian.
        """
        return prompted_image