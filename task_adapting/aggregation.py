'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Saturday, 6th January 2024 5:49:24 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from abc import ABC, abstractmethod
import time
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
    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode):
        # base_agg.update(self.cluster_and_rep_list["testing"], self.prototype_list["training"], self.model, best_prompter, data_loader, self.local_batch_size, self.devicename)
        pass

    @abstractmethod
    def update_from_base(self, base_agg):
        pass

    @abstractmethod
    def get_prediction(self, index, data_item):
        """
        Retrieves the prediction based on multiple aggregation strategies.

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
        self.aggregation_method = None
        self.cluster_and_rep_list = None
        self.prototype_list = None
        self.batch_size = None
        self.num_class = None
        self.mode = None

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode):
        self.cluster_and_rep_list = cluster_and_rep_list
        self.prototype_list = prototype_list
        self.batch_size = batch_size
        self.device = device
        self.num_class = num_class

        self.model = model
        self.prompter = prompter
        self.data_loader = data_loader
        self.mode = mode

        self.sigma_list = self.load_sigmas()
        self.distance_tensor = self.precompute_distances()
        self.logits_tensor = self.precompute_logits()

    def update_from_base(self, base_agg):
        self.distance_tensor = base_agg.distance_tensor
        self.logits_tensor = base_agg.logits_tensor
        self.sigma_list = base_agg.sigma_list

    def load_sigmas(self) -> list[torch.Tensor]:
        """load sigmas from prototype_list

        @return sigma_list: [
            torch.Tensor: [P]
        ]
        """

        sigma_list = []
        for prototype in self.prototype_list:
            if prototype.sigma is torch.nan:
                sigma_list.append(torch.nan)
            else:
                sigma_list.append(torch.norm(prototype.sigma))
        return sigma_list

    def precompute_distances(self) -> list[torch.Tensor]:
        """提前计算所有batch的到各个prototype的距离矩阵

        @return distance_matrix_list: [
            torch.Tensor: [N, B, P]
        ]

        N: number of batches
        B: batch size
        P: number of prototypes
        """
        begin_dist_time = time.time()
        distance_matrix = torch.zeros(
            (self.batch_size, len(self.prototype_list))).to(self.device)
        for batch_idx, cluster_and_rep in enumerate(self.cluster_and_rep_list):
            rep = cluster_and_rep.rep
            prototype_gather = self.prototype_list.prototype_gather
            distance_matrix[batch_idx] = calculate_distance_matrix(
                rep, prototype_gather)
        end_dist_time = time.time()
        logging.info(
            f"Time for calculating distance matrix: {end_dist_time - begin_dist_time}")
        return distance_matrix

    def precompute_logits(self) -> list[torch.Tensor]:
        """提前计算所有batch在不同prompter下的logits

        @return logits_tensor: [N, B, P, C]
        """
        begin_logits_time = time.time()
        logits_tensor = torch.zeros(
            (self.batch_size, len(self.prototype_list), self.num_class)).to(self.device)
        for batch_idx, cluster_and_rep in enumerate(self.cluster_and_rep_list):
            rep = cluster_and_rep.rep
            for prototype_idx, prompter in enumerate(self.prompter):
                prompted_images = prompter(rep)
                logits_tensor[batch_idx, prototype_idx] = self.model(
                    prompted_images)[:, :self.num_class]
        end_logits_time = time.time()
        logging.info(
            f"Time for calculating logits: {end_logits_time - begin_logits_time}")
        return logits_tensor

    def get_prediction(self, index, data_item):
        """根据不同的聚合策略，返回预测结果

        @return prediction: [B]
        """
        pass

    def cal_prediction(self, weight, logits):
        """根据weight和logits计算预测结果

        @return prediction: [B]
        """
        return torch.argmax(torch.sum(weight.unsqueeze(-1) * logits, dim=1), dim=-1)


class nearestAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "nearest"

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode):
        # super().__init__(prompter, model, device, adapter)
        super().update(self, cluster_and_rep_list, prototype_list, model,
                       prompter, data_loader, batch_size, device, num_class, mode)

    def update_from_base(self, base_agg):
        return super().update_from_base(base_agg)

    def get_prediction(self, index, data_item):
        logging.info(f"{self.aggregation_method} Aggregation")

        dist_matrix = self.distance_tensor[index]  # [B, P]
        weight = torch.softmax(-dist_matrix, dim=-1)  # [B, P]
        return super().cal_prediction(weight, self.logits_tensor[index])


class gaussianAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "gaussian"

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode):
        super().update(self, cluster_and_rep_list, prototype_list, model,
                       prompter, data_loader, batch_size, device, num_class, mode)

    def update_from_base(self, base_agg):
        return super().update_from_base(base_agg)

    def get_prediction(self, index, data_item):
        logging.info(f"{self.aggregation_method} Aggregation")

        # weight[i][j] = exp(-dist_matrix[i][j] / sigma[j])

        dist_matrix = self.distance_tensor[index].to(self.device)  # [B, P]
        sigma = torch.stack(self.sigma_list).to(self.device)  # [P]
        weight = torch.exp(-dist_matrix / sigma)  # [B, P]
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        return super().cal_prediction(weight, self.logits_tensor[index])


class majorityAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "majority"

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode):
        super().update(self, cluster_and_rep_list, prototype_list, model,
                       prompter, data_loader, batch_size, device, num_class, mode)

    def update_from_base(self, base_agg):
        return super().update_from_base(base_agg)

    def get_prediction(self, index, data_item):
        weight = torch.ones(
            (self.batch_size, len(self.prototype_list))).to(self.device) / len(self.prototype_list)
        return super().cal_prediction(weight, self.logits_tensor[index])
