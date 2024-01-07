from argparse import Namespace
import os
import sys
import time
from typing import List
import numpy as np
from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import sklearn.cluster as clusterAgg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from models import prompters
from models.builder import get_current_device
import models.backbones.backbone_vit as bb
from data_utils import loader as data_loader
from utils.functional import set_seed
from utils.train_utils import cosine_lr
import utils.logging as logGing

from arguments import Arguments
from aggregation import AggregationStrategy, BaseAggregation, nearestAggregation
from cluster_and_rep import ClusterAndRepList, ProtoTypeList


logger = logGing.get_logger("dam-vp")


class Adapter(object):
    """A Gather of Our Task Adapting Methods.

    use strategy mode
    """

    def __init__(self, args: Arguments, model: bb.VisionTransformer, aggregation_strategy_list: List[AggregationStrategy]):
        super(Adapter, self).__init__()
        self.args: Namespace = args
        self.model: bb.VisionTransformer = model.eval()

        self.logger: logging.Logger = logGing.get_logger("dam-vp")

        self.lr: float = args.lr
        self.weight_decay = args.weight_decay
        self.criterion = nn.CrossEntropyLoss().to(args.device)  # 交叉熵损失函数

        self.device: torch.device = args.device
        self.rank: int = dist.get_rank() if dist.is_initialized() else 0

        # destination to this gpu -- .to(self.devicename)
        self.devicename = torch.device(
            f"cuda:{self.rank}" if torch.cuda.is_available() else get_current_device())
        self.logger.info(f"self.devicename: {self.devicename}")

        self.local_batch_size = args.batch_size // args.num_gpus if dist.is_initialized(
        ) else args.batch_size
        self.aggregation_strategy_list = aggregation_strategy_list
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.cluster_and_rep_list = None
        self.prototype_list = None
        self.aggregation_strategy = None
        self.aggregation_strategy_name = None
        self.num_coarse_classes = None
        self.prototype_gather = None
        self.indices = None

    def loss_function(self, logits, target) -> torch.Tensor:
        """Loss function to predict GT target.
        """
        return self.criterion(logits, target)

    def load_prompter(self, prompter_path=None) -> (prompters.PadPrompter | prompters.FixedPatchPrompter | prompters.RandomPatchPrompter):
        """Load the trained visual prompter.
        """
        prompter = prompters.__dict__[self.args.prompt_method](
            self.args).to(self.devicename)  # prompt_method: pad, fixed, random, default: prompter = padding(args)
        if prompter_path is not None:
            checkpoint = torch.load(
                prompter_path, map_location=self.devicename)
            prompter.load_state_dict(checkpoint['state_dict'])
            logger.info(
                f"Loading meta-trained visual prompts from {prompter_path}")
        return prompter

    def get_active_neuron_index(self) -> torch.Tensor:
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

    def rep2logit(self, output, num_classes) -> torch.Tensor:
        """Convert the output representations to logits. 
        把输出的表示转换为logits
        """
        if self.args.adapt_method != "prompt_wo_head":
            raise NotImplementedError
        # activity aware
        indices = self.indices[:num_classes]
        indices = torch.unique(indices)
        return output[:, indices]

    def get_nearest_prompted_image(self, image, prototype_gather=None, prompter=None, prompter_gather=None) -> torch.Tensor:
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

    def get_prompted_image_val(self, sample, prototype_gather=None, prompter=None, prompter_gather=None) -> torch.Tensor:
        """Obtain the prompted batch images with specified aggregation method."""
        if not self.args.wo_da:
            return self.get_prompted_image_w_da(
                prototype_gather, prompter_gather, sample
            )
        assert prompter is not None
        return prompter(sample["image"].to(self.devicename))

    def get_prompted_image_w_da(self, prototype_gather, prompter_gather, sample) -> torch.Tensor:
        assert prototype_gather is not None
        assert prompter_gather is not None
        with torch.no_grad():
            # rep_batch = self.model.forward_features(image)  # [N, emd_dim]
            return self.aggregation_strategy_list.get_prompted_images(sample, prototype_gather, prompter_gather, self)

    def coarse_clustering(self, train_loader) -> None:
        """Diversity-Aware Adaption on downstream data.
        We roughly divide the downstream task data into several partitions, each 
        partition presents a coarsely divided cluster. Different clusters correspond 
        to different prompters. 

        updated attributes:
            self.num_coarse_classes: number of coarsely divided clusters
            self.prototype_gather: prototypes of each cluster
        """
        threshold_dict: dict \
            = {
                "resnet50-1k": 21,
                "vit-b-1k": 31,
                "vit-b-22k": 10,
                "swin-b-22k": 20,
                "moco-v3-b-1k": 18
            }
        hc: clusterAgg.AgglomerativeClustering = \
            clusterAgg.AgglomerativeClustering(
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

        y_pred: np.ndarray = hc.fit(rep_gather.detach().cpu().numpy()).labels_
        y_pred: torch.Tensor = torch.from_numpy(y_pred).to(self.devicename)
        coarse_class_idx: torch.Tensor = torch.unique(y_pred)
        self.num_coarse_classes: int = len(coarse_class_idx)
        logger.info(
            f"Nums of coarsely divided categories for test dataset {self.args.test_dataset}: {len(coarse_class_idx)}"
        )

        prototype_gather: List[torch.Tensor] = []
        for i in range(len(coarse_class_idx)):
            pos: torch.Tensor = torch.where(y_pred == i)[0]
            prototype: torch.Tensor = rep_gather[pos].mean(0).unsqueeze(0)
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
                prompted_image = self.get_nearest_prompted_image(image, prompter=prompter) \
                    if self.args.wo_da else self.get_nearest_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
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
                    prompted_image = self.get_nearest_prompted_image(image, prompter=prompter) \
                        if self.args.wo_da else self.get_nearest_prompted_image(image, self.prototype_gather, prompter_gather=prompter_gather)
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
                        prompted_image = self.get_nearest_prompted_image(image, prompter=best_prompter) \
                            if self.args.wo_da else self.get_nearest_prompted_image(image, self.prototype_gather, prompter_gather=best_prompter_gather)
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
            = self.init_with_head_not_prompted(test_data, prompter_path)

        self.model.get_classifier().train()  # unfreeze the head
        best_prompter, optimizer, scheduler = self.make_opt_and_bpr(
            len(train_loader), prompter)
        BEST_ACC_VAL = -np.inf
        for epoch in range(self.args.epochs):
            # train
            self.training_part(logger, train_loader, best_prompter,
                               optimizer, scheduler, epoch)
            # validate
            best_prompter = self.make_validation(
                logger, val_loader, best_prompter, BEST_ACC_VAL, epoch)
            # test
            if epoch > 0 and (epoch + 1) % 5 == 0:
                acc_test = self.make_test(
                    logger, test_loader, epoch, best_prompter)
        return acc_test

    def init_with_head_not_prompted(self, test_data, prompter_path):
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
        return logger, train_loader, val_loader, test_loader, prompter

    def init_with_head(self, test_data: List[DataLoader], prompter_path: str) \
        -> tuple[logging.Logger, DataLoader, DataLoader, DataLoader,
                 prompters.PadPrompter | prompters.FixedPatchPrompter | prompters.RandomPatchPrompter,
                 torch.optim.SGD, cosine_lr]:
        """define logger, train_loader, val_loader, test_loader, prompter and do coarse clustering.
        * test_data is prompted here
        """
        logger: logging.Logger = self.logger
        logger.info(f"self.devicename: {self.devicename}")

        train_loader, val_loader, test_loader = test_data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        train_loader_len = len(train_loader)

        prompter = self.load_prompter(prompter_path)
        num_classes = data_loader._dataset_class_num(self.args.test_dataset)
        self.model.reset_classifier(num_classes)
        self.model.get_classifier().to(self.devicename)

        begin_time_cluster = time.time()

        if not self.args.wo_da:
            coarse_cocluster_path = f"{self.args.output_dir}/coarse_cluster_{self.args.test_dataset}_device_{self.devicename}.pth"
            coarse_renew: bool = False
            # coarse_renew: bool = True

            if (coarse_renew or not os.path.exists(coarse_cocluster_path)):
                self.coarse_clustering(train_loader)
                # save self.num_coarse_classes and self.prototype_gather

                torch.save({
                    "num_coarse_classes": self.num_coarse_classes,
                    "prototype_gather": self.prototype_gather
                }, coarse_cocluster_path)
            else:
                checkpoint = torch.load(coarse_cocluster_path)
                self.num_coarse_classes = checkpoint["num_coarse_classes"]
                self.prototype_gather = checkpoint["prototype_gather"]

        best_prompter, optimizer, scheduler = self.make_opt_and_bpr(
            train_loader_len, prompter)

        renew = False
        # renew = True
        self.cluster_and_rep_list: dict[str, ClusterAndRepList] = {
            "training": ClusterAndRepList(
                f"{self.args.output_dir}/train_cluster_mapping_{self.args.test_dataset}", train_loader, self, renew=renew),
            "validating": ClusterAndRepList(
                f"{self.args.output_dir}/val_cluster_mapping_{self.args.test_dataset}", val_loader, self, renew=renew),
            "testing": ClusterAndRepList(
                f"{self.args.output_dir}/test_cluster_mapping_{self.args.test_dataset}", test_loader, self, renew=renew)
        }
        # self.cluster_mappings["training"][0].cluster
        # tensor([ 73,  10,  74,   9,  33,  29,  32,  76,   4,  25,  75,  73,  42, 209,
        #         100,   2], device='cuda:0')

        self.prototype_list = {
            "training": ProtoTypeList(
                self.prototype_gather, self.cluster_and_rep_list["training"]),
            # "validating": ProtoTypeList(
            #     self.prototype_gather, self.cluster_mappings["validating"]),
            # "testing": ProtoTypeList(
            #     self.prototype_gather, self.cluster_mappings["testing"])
        }

        logger.info(f"Cluster time: {time.time() - begin_time_cluster}")
        # remove cache
        torch.cuda.empty_cache()

        return logger, train_loader, val_loader, test_loader, \
            best_prompter, optimizer, scheduler

    def get_prompted_image_train(self, batch_idx, sample, prompter_gather):
        """Obtain the prompted batch images using CUDA streams."""

        clusters = self.cluster_and_rep_list["training"][batch_idx].cluster
        images = sample["image"].to(self.devicename)

        # if streams is supported, use streams to accelerate
        if torch.cuda.is_available():
            streams = [torch.cuda.Stream() for _ in range(len(images))]
            prompted_images = []

            with torch.no_grad():
                for cluster, image, stream in zip(clusters, images, streams):
                    with torch.cuda.stream(stream):
                        # 应用 prompter_gather 并将结果添加到列表
                        prompted_image = prompter_gather[cluster](
                            image.unsqueeze(0))
                        prompted_images.append(prompted_image)

                torch.cuda.synchronize()  # 确保所有流完成处理
        else:
            # 如果不支持流，则按照原始方式处理
            with torch.no_grad():
                prompted_images = [
                    prompter_gather[cluster](image.unsqueeze(0))
                    for cluster, image in zip(clusters, images)
                ]

        # 将所有处理后的图像合并成一个批次
        return torch.cat(prompted_images, dim=0)

    def training_part(self, logger, train_loader, prompter, optimizer, scheduler, epoch, renew: bool = False) -> bool:
        for i, sample in enumerate(train_loader):
            # adjust learning rate
            global_step = len(train_loader) * epoch + i
            scheduler(global_step)

            path = f"{self.args.output_dir}/prompter_{self.args.test_dataset}_epoch_{epoch}_device_{self.devicename}.pth"

            if not renew and os.path.exists(path):
                if os.path.exists(f"{self.args.output_dir}/prompter_{self.args.test_dataset}_epoch_{epoch+1}_device_{self.devicename}.pth"):
                    logger.info(f"Skipping epoch {epoch}...")
                else:
                    checkpoint = torch.load(
                        path, map_location=self.devicename)
                    # load the prompter (List nn.Module)
                    prompter_state_dicts = checkpoint['prompter_state_dicts']

                    prompter = [prompters.__dict__[self.args.prompt_method](
                        self.args).to(self.devicename) for _ in range(self.num_coarse_classes)]

                    for idx, p in enumerate(prompter):
                        p.load_state_dict(
                            prompter_state_dicts[f'prompter_{idx}'])

                    optimizer.load_state_dict(checkpoint['optimizer'])

                    logger.info(
                        f"Loading meta-trained visual prompts from {path}")

                return True
            prompted_image = self.get_prompted_image_train(i, sample, prompter)
            logits = self.model(prompted_image)
            loss = self.loss_function(
                logits, sample["label"].to(self.devicename))
            optimizer.zero_grad()  # zero_grad: clear the gradient of all optimized torch.Tensor
            loss.backward()  # calculated: gradient of prompter
            optimizer.step()  # updated: prompter

            logger.info(
                f"[Prompt Finetuning] Epoch: [{epoch}/{self.args.epochs}], Step: [{i}/{len(train_loader)}], Training loss: {loss.item()}, device: {self.devicename}"
            )

        logger.info(f"[Prompt Finetuning] Epoch: {epoch} finished.")
        # save the prompter
        prompter_state_dicts = {
            f'prompter_{idx}': p.state_dict() for idx, p in enumerate(prompter)}
        torch.save({
            'epoch': epoch,
            'prompter_state_dicts': prompter_state_dicts,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
        }, path)

        return False

    def evaluate(self, logger: logging.Logger, data_loader: DataLoader, prompter: list[
            prompters.PadPrompter | prompters.FixedPatchPrompter | prompters.RandomPatchPrompter], mode: str, epoch: int, base_agg=None) -> float:
        num_total, correct = 0, 0
        # begin_time = time.time()
        for i, sample in enumerate(data_loader):
            # sample is a dict
            # add an attribute "epoch" to it as i
            # sample["batch"] = i
            # image = sample["image"].to(self.devicename)
            label = sample["label"].to(self.devicename)
            # logger.info(
            #     f"type of data_loader: {type(data_loader)} in evaluate")
            if mode == 'testing':
                testingAggregator = self.aggregation_strategy
                testingAggregator.update_from_base(base_agg)
                testingAggregator.get_actual_batch_size(sample)
                pred = testingAggregator.get_prediction(i)
            elif mode == 'validating':
                validationAggregator = nearestAggregation()
                validationAggregator.update(
                    prompter, self.model, self.devicename, self
                )
                pred = validationAggregator.get_prediction(i)
            else:
                raise NotImplementedError
            correct += (pred == label).sum().item()
            num_total += sample["image"].size(0)
        return float(correct / num_total)

    def make_test(self, logger, test_loader, epoch, best_prompter) -> float:
        self.logger.info("Testing")
        base_agg = BaseAggregation()
        base_agg.update(
            self.cluster_and_rep_list["testing"], self.prototype_list["training"], self.model,
            best_prompter, test_loader, self.local_batch_size, self.devicename, len(test_loader.dataset.classes), mode="testing",
            logger=logger, out_path=self.args.output_dir)
        for strategy in self.aggregation_strategy_list:
            logger.info(f"Testing with {strategy.__class__.__name__}")
            begin_time = time.time()
            strategy.update_from_base(base_agg)
            self.aggregation_strategy = strategy
            self.aggregation_strategy_name = strategy.__class__.__name__
            acc_test = self.evaluate(
                logger, test_loader, best_prompter, 'testing', epoch, base_agg)
            logger.info(
                f"[Testing]Test acc: {acc_test}, Time consuming {time.time() - begin_time} for {self.aggregation_strategy_name} in device {self.devicename}")
        return acc_test

    def make_validation(self, logger, val_loader, prompter, best_prompter, best_acc_val, epoch) -> tuple[float, prompters.PadPrompter | prompters.FixedPatchPrompter | prompters.RandomPatchPrompter]:
        logger.info(f"[Prompt Finetuning] Validating: epoch {epoch}")
        correct = 0
        for ind, sample in enumerate(val_loader):
            prompted_images = self.get_prompted_image_train(
                ind, sample, prompter)
            pred = torch.argmax(self.model(prompted_images), dim=-1)
            correct += (pred ==
                        sample["label"].to(self.devicename)).sum().item()
        num_total = sample["image"].size(0)
        acc_val = float(correct / num_total)
        if acc_val > best_acc_val:
            best_acc_val = acc_val
            best_prompter = deepcopy(prompter)
        return best_acc_val, best_prompter

    def make_opt_and_bpr(self, train_loader_len, _prompter) -> tuple[prompters.PadPrompter, torch.optim.SGD, cosine_lr]:
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

        logger, train_loader, val_loader, test_loader, best_prompter_gather, optimizer, scheduler \
            = self.init_with_head(test_data, prompter_path)

        prompter_gather: List[prompters.PadPrompter | prompters.FixedPatchPrompter |
                              prompters.RandomPatchPrompter] = best_prompter_gather[:]

        self.model.get_classifier().train()
        BEST_ACC_VAL = -np.inf

        # logger out: *** our method with head ***
        logger.info("*** our method with head ***")

        # label with cluster result
        for epoch in range(self.args.epochs):
            # train
            skip = self.training_part(logger, train_loader, prompter_gather,
                                      optimizer, scheduler, epoch)

            if skip:
                continue

            # validate
            [BEST_ACC_VAL, best_prompter_gather] = self.make_validation(
                logger, val_loader, prompter_gather, best_prompter_gather, BEST_ACC_VAL, epoch)

            # test
            # epoch = self.args.epochs - 1
            if epoch == self.args.epochs - 1:
                acc_test = self.make_test(
                    logger, test_loader, epoch, best_prompter_gather)
                break
        return acc_test

    def save_checkpoint(self, path, test_data, prompters_path):
        logger, train_loader, val_loader, test_loader, best_prompter, optimizer, scheduler \
            = self.init_with_head(test_data, prompters_path)
        self.training_part(logger, train_loader, best_prompter,
                           optimizer, scheduler, 0, renew=True)
        torch.save({
            "prompters": best_prompter.state_dict()
        }, os.path.join(path, f"device_{self.devicename}.pth"))

        try:
            torch.load(os.path.join(path, f"device_{self.devicename}.pth"))
        except:
            logger.info("checkpoint save failed")
        else:
            logger.info("checkpoint save success")
