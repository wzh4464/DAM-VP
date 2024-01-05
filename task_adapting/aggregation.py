'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Saturday, 6th January 2024 12:23:00 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from abc import ABC, abstractmethod
import torch
# stream
from torch.cuda import Stream

import logging


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.

    """
    # @abstractmethod
    # def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
    #     """
    #     Retrieves the prompted images based on the given representations, prototypes, image, and prompter.

    #     Args:
    #         rep_batch (torch.Tensor): Tensor of shape [N, D] representing the batch of representations.
    #         prototype_gather (torch.Tensor): Tensor of shape [M, D] representing the gathered prototypes.
    #         image (torch.Tensor): Tensor representing the input image.
    #         prompter_gather (torch.Tensor): Tensor of shape [M, D] representing the gathered prompters.

    #     Returns:
    #         torch.Tensor: Tensor representing the prompted images.

    #     """
    #     pass

    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_prediction(self, sample, prompter, model, device, num_classes, adapter):
        """
        Retrieves the prediction based on the given prompted images, model, label, and loss function.

        Args:
            prompted_images (torch.Tensor): Tensor representing the prompted images.
            model (torch.nn.Module): The model to use for the forward pass.
            label (torch.Tensor): Tensor representing the label.
            loss (torch.nn.Module): The loss function to use.

        Returns:
            torch.Tensor: Tensor representing the prediction.

        """
        pass


def calculate_distance_matrix(rep_batch, prototype_gather):
    """
    Calculates the distance matrix between rep_batch and prototype_gather.

    Args:
        rep_batch (torch.Tensor): Tensor of shape [N, D] representing the batch of representations.
        prototype_gather (torch.Tensor): Tensor of shape [M, D] representing the gathered prototypes.

    Returns:
        torch.Tensor: Tensor of shape [N, M] representing the distance matrix.

    """

    rep_batch_sum = (rep_batch**2).sum(dim=-1, keepdims=True)  # [N, 1]
    prototype_gather_sum = (
        prototype_gather**2).sum(dim=-1, keepdims=True).T  # [1, M]
    return torch.sqrt(
        rep_batch_sum
        + prototype_gather_sum
        - 2 * torch.mm(rep_batch, prototype_gather.T)
    )


def get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter):
    """Gets all prompted images for each prototype.

    @return prompted_images: [P, B, C, H, W], P is the number of prototypes
    """
    # get all prompted images for each image
    # prompted_images
    # first dimension: prototype ind
    # second dimension: batch
    # other dimension: image
    batch_size = adapter.local_batch_size
    logger = adapter.logger
    # logger out: batch_size
    logger.info(f"batch_size: {str(batch_size)}")
    logger.info(f"rank: {str(adapter.rank)}")
    logger.info(f"local_batch_size: {str(adapter.local_batch_size)}")
    image = sample["image"].to(adapter.devicename)

    for i in range(prototype_gather.shape[0]):
        prompted_images = [
            prompter_gather[i](image[idx].unsqueeze(0))
            for idx in range(batch_size)
        ]
        prompted_images = torch.cat(
            prompted_images, dim=0).to(adapter.devicename)
        if i == 0:
            prompted_images_all = prompted_images.unsqueeze(0)
        else:
            prompted_images_all = torch.cat(
                [prompted_images_all, prompted_images.unsqueeze(0)], dim=0)

    logger.info(f"prompted_images_all.shape: {str(prompted_images_all.shape)}")
    return prompted_images_all


class nearestAggregation(AggregationStrategy):
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
        # 具体实现 A 的 get_prompted_images
        """Nearest Neighbor.

        @return prompted_images: [B, C, H, W]
        """
        # rep_batch = sample["rep_batch"].to(adapter.devicename)
        # image = sample["image"].to(adapter.devicename)
        # distance_matrix = calculate_distance_matrix(
        #     rep_batch, prototype_gather)
        # indices = torch.argmin(distance_matrix, dim=-1)  # [B]
        batch_size = adapter.local_batch_size
        prompted_images = [
            # prompter_gather[indices[idx]](image[idx].unsqueeze(0))
            prompter_gather[sample["prototype_indices"][idx]](
                sample["image"][idx].unsqueeze(0))
            for idx in range(batch_size)
        ]
        prompted_images = torch.cat(
            prompted_images, dim=0).to(adapter.devicename)
        return prompted_images

    def get_prediction(self, sample, prompter, model, device, num_classes, adapter):
        """Nearest Neighbor 的 get_prediction

        @return loss: [1]
        """
        prompted_images = adapter.get_prompted_image_train(
            sample["epoch"], sample, prompter)
        return torch.argmax(adapter.model(prompted_images), dim=-1)


class averageAggregation(AggregationStrategy):
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
        """Average Fusion 的 get_prompted_images

        @return prompted_images: [B, P, C, H, W], P is the number of prototypes
        """
        return get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter)

    def get_prediction(self, sample, prompter, model, device, num_classes, adapter):
        """Average Fusion 的 get_prediction

        @return prediction: [B] (B is the batch size)
        """
        self.logger.info("Average Fusion")
        cluster_num = len(prompter)
        logits_sum = torch.zeros(
            sample["image"].shape[0], num_classes).to(device)
        image = sample["image"].to(device)
        for i in range(cluster_num):
            prompted_images = prompter[i](image)
            logits = model.forward(prompted_images)
            # logits = rep2logit(output, num_classes)

            logits_sum += logits
            del prompted_images

        # 计算平均值
        average_logits = logits_sum / cluster_num
        prediction = torch.argmax(average_logits, dim=0)
        return prediction


class majorityAggregation(AggregationStrategy):
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
        """Majority Voting 的 get_prompted_images

        @return prompted_images: [B, P, C, H, W], P is the number of prototypes
        """
        return get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter)

    def get_prediction(self, sample, prompter, model, device, num_classes, adapter):
        """Majority Voting 的 get_prediction

        @return prediction: [B] (B is the batch size)
        """
        # get all losses for each prompted image
        # losses
        # first dimension: batch
        # second dimension: prototype ind
        # other dimension: loss

        cluster_num = len(prompter)
        counts = None
        image = sample["image"].to(device)
        for i in range(cluster_num):
            prompted_images = prompter[i](image)
            logits = model(prompted_images)
            # self.logger.info(f"logits.shape: {str(logits.shape)}")
            # counts += logits
            if counts is None:
                counts = logits
            else:
                counts += logits
            del prompted_images

        return torch.argmax(counts, dim=-1)


class gaussianAggregation(AggregationStrategy):
    def get_prediction(self, sample, prompter, model, device, num_classes, adapter):
        """Gaussian Aggregation 的 get_prediction

        @return prediction: [B] (B is the batch size)
        """
        logging.info("Gaussian Aggregation")
        time_start = torch.cuda.Event(enable_timing=True)
        time_end = torch.cuda.Event(enable_timing=True)

        time_start.record()
        cluster_list = adapter.cluster_list["training"]
        counts = None
        image = sample["image"].to(device)
        # 对每个cluster计算一次norm，而不是在循环中重复计算
        norms = [torch.norm(cluster.sigma) ** 2 if cluster.sigma is not torch.nan else torch.nan for cluster in cluster_list]


        if torch.cuda.is_available():
            cluster_num = len(cluster_list)
            streams = [Stream() for _ in range(cluster_num)]

            with torch.no_grad():
                for i, stream in enumerate(streams):
                    with torch.cuda.stream(stream):
                        if cluster_list[i].sigma is torch.nan:
                            # give a warning
                            # self.logger.warning("sigma is nan")
                            continue

                        prompted_images = prompter[i](image)
                        image_rep = model.forward_features(prompted_images)
                        logits = model.head(image_rep)

                        # 计算高斯权重
                        diff = image_rep - cluster_list[i].prototype
                        # assert device correct
                        assert diff.device == device
                        weights = torch.exp(-torch.sum(diff ** 2, dim=1) / (2 * norms[i]))
                        # 加权预测
                        # counts += weights.unsqueeze(1) * logits
                        if counts is None:
                            counts = weights.unsqueeze(1) * logits
                        else:
                            counts.add_(weights.unsqueeze(1) * logits)

            for stream in streams:
                stream.synchronize()
        else:
            # cpu
            for i in range(len(cluster_list)):
                if cluster_list[i].sigma is torch.nan:
                    # give a warning
                    # self.logger.warning("sigma is nan")
                    continue

                prompted_images = prompter[i](image)
                image_rep = model.forward_features(prompted_images)
                logits = model.head(image_rep)

                # 计算高斯权重
                diff = image_rep - cluster_list[i].prototype
                # assert device correct
                assert diff.device == device
                weights = torch.exp(-torch.sum(diff ** 2, dim=1) / (2 * norms[i]))
                # 加权预测
                # counts += weights.unsqueeze(1) * logits
                if counts is None:
                    counts = weights.unsqueeze(1) * logits
                else:
                    counts.add_(weights.unsqueeze(1) * logits)

        # 得到最终预测
        prediction = torch.argmax(counts, dim=1)
        time_end.record()


        logging.info(f"gaussianAggregation time: {time_start.elapsed_time(time_end)}")

        return prediction
