'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Saturday, 6th January 2024 1:37:45 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from abc import ABC, abstractmethod
import torch

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
    def update(self, prompter, model, device, adapter):
        pass

    @abstractmethod
    def get_prediction(self, sample, num_classes):
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


# def get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter):
#     """Gets all prompted images for each prototype.

#     @return prompted_images: [P, B, C, H, W], P is the number of prototypes
#     """
#     # get all prompted images for each image
#     # prompted_images
#     # first dimension: prototype ind
#     # second dimension: batch
#     # other dimension: image
#     batch_size = adapter.local_batch_size
#     logger = adapter.logger
#     # logger out: batch_size
#     logger.info(f"batch_size: {str(batch_size)}")
#     logger.info(f"rank: {str(adapter.rank)}")
#     logger.info(f"local_batch_size: {str(adapter.local_batch_size)}")
#     image = sample["image"].to(adapter.devicename)

#     for i in range(prototype_gather.shape[0]):
#         prompted_images = [
#             prompter_gather[i](image[idx].unsqueeze(0))
#             for idx in range(batch_size)
#         ]
#         prompted_images = torch.cat(
#             prompted_images, dim=0).to(adapter.devicename)
#         if i == 0:
#             prompted_images_all = prompted_images.unsqueeze(0)
#         else:
#             prompted_images_all = torch.cat(
#                 [prompted_images_all, prompted_images.unsqueeze(0)], dim=0)

#     logger.info(f"prompted_images_all.shape: {str(prompted_images_all.shape)}")
#     return prompted_images_all


# class nearestAggregation(AggregationStrategy):
#     def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
#         # 具体实现 A 的 get_prompted_images
#         """Nearest Neighbor.

#         @return prompted_images: [B, C, H, W]
#         """
#         # rep_batch = sample["rep_batch"].to(adapter.devicename)
#         # image = sample["image"].to(adapter.devicename)
#         # distance_matrix = calculate_distance_matrix(
#         #     rep_batch, prototype_gather)
#         # indices = torch.argmin(distance_matrix, dim=-1)  # [B]
#         batch_size = adapter.local_batch_size
#         prompted_images = [
#             # prompter_gather[indices[idx]](image[idx].unsqueeze(0))
#             prompter_gather[sample["prototype_indices"][idx]](
#                 sample["image"][idx].unsqueeze(0))
#             for idx in range(batch_size)
#         ]
#         prompted_images = torch.cat(
#             prompted_images, dim=0).to(adapter.devicename)
#         return prompted_images

#     def get_prediction(self, sample, prompter, model, device, num_classes, adapter):
#         """Nearest Neighbor 的 get_prediction

#         @return loss: [1]
#         """
#         prompted_images = adapter.get_prompted_image_train(
#             sample["epoch"], sample, prompter)
#         return torch.argmax(adapter.model(prompted_images), dim=-1)


class BaseAggregation(AggregationStrategy):
    def __init__(self) -> None:
        super().__init__()
        self.prompter = None
        self.model = None
        self.device = None
        self.adapter = None
        self.precomputed_reps = None
        self.precomputed_logits = None
        self.cluster_norms = None

    def update(self, prompter, model, device, adapter):
        self.prompter = prompter
        self.model = model
        self.device = device
        self.adapter = adapter
        self.precomputed_reps = self.precompute_reps()
        self.precomputed_logits = self.precompute_logits()
        self.cluster_norms = [torch.norm(cluster.sigma) ** 2 if cluster.sigma is not torch.nan else torch.nan for cluster in adapter.cluster_list["training"]]


    def precompute_reps(self):
        # 提前计算所有clusters的representation
        cluster_list = self.adapter.cluster_list["training"]
        rep_dict = {}
        for i, cluster in enumerate(cluster_list):
            image = cluster.sample_image.to(self.device)
            prompted_images = self.prompter[i](image)
            image_rep = self.model.forward_features(prompted_images)
            rep_dict[i] = image_rep
        return rep_dict

    def precompute_logits(self):
        # 提前计算所有clusters的logits
        self.precompute_logits = {}
        for i, rep in self.precomputed_reps.items():
            logits = self.model.head(rep)
            self.precompute_logits[i] = logits

    def get_prediction(self, sample, num_classes):
        logging.info(f"{self.aggregation_method} Aggregation")

        cluster_list = self.adapter.cluster_list["training"]
        counts = None

        if torch.cuda.is_available():
            with torch.no_grad():
                for i, cluster in enumerate(cluster_list):
                    counts = self.process_cluster(i, cluster, counts)

            torch.cuda.synchronize()
        else:
            # CPU处理
            for i, cluster in enumerate(cluster_list):
                counts = self.process_cluster(i, cluster, counts)

        # 得到最终预测
        prediction = torch.argmax(
            counts, dim=1) if counts is not None else None

        return prediction

    def process_cluster(self, index, cluster, counts):
        logits = self.precomputed_logits[index]

        if self.aggregation_method == "gaussian":
            # 计算高斯权重
            diff = self.precomputed_reps[index] - cluster.prototype
            norms = self.cluster_norms
            weights = torch.exp(-torch.sum(diff ** 2,
                                dim=1) / (2 * norms[index]))
            logits_weighted = weights.unsqueeze(1) * logits

        if self.aggregation_method == "majority":
            logits_weighted = logits

        if self.aggregation_method == "nearest":
            # 计算距离矩阵
            distance_matrix = calculate_distance_matrix(
                self.precomputed_reps[index], cluster.prototype)
            # 计算最近邻
            indices = torch.argmin(distance_matrix, dim=-1)
            logits_weighted = logits[indices]

        else:  # 默认为 majority
            raise NotImplementedError

        # 加权预测
        if counts is None:
            counts = logits_weighted


class nearestAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "nearest"

    def update(self, prompter, model, device, adapter):
        # super().__init__(prompter, model, device, adapter)
        super().update(prompter, model, device, adapter)

    def get_prediction(self, sample, num_classes):
        logging.info(f"{self.aggregation_method} Aggregation")

        # 其他逻辑与 BaseAggregation 相同
        return super().get_prediction(sample, num_classes)


class gaussianAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "gaussian"

    def get_prediction(self, sample, num_classes):
        return super().get_prediction(sample, num_classes)


class majorityAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "majority"

    def get_prediction(self, sample, num_classes):
        super().get_prediction(sample, num_classes)
