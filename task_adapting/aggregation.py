'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Sunday, 7th January 2024 9:47:16 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from abc import ABC, abstractmethod
import os
import time
import torch


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

    @abstractmethod
    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode, logger, out_path):
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

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode, logger, out_path):
        self.cluster_and_rep_list = cluster_and_rep_list
        self.prototype_list = prototype_list
        self.batch_size = batch_size
        self.device = device
        self.num_class = num_class
        self.logger = logger
        self.out_path = out_path

        self.model = model
        self.prompter = prompter
        self.data_loader = data_loader
        self.mode = mode

        with torch.no_grad():
            self.sigma_list = self.load_sigmas()
            self.distance_tensor = self.precompute_distances()
            self.logits_tensor = self.precompute_logits()

    def update_from_base(self, base_agg):
        self.distance_tensor = base_agg.distance_tensor
        self.logits_tensor = base_agg.logits_tensor
        self.sigma_list = base_agg.sigma_list
        self.logger = base_agg.logger

    def load_sigmas(self, renew=False) -> list[torch.Tensor]:
        """load sigmas from prototype_list

        @return sigma_list: [
            torch.Tensor: [P]
        ]
        """

        path = f"{self.out_path}/sigma_list_{self.device}.pth"
        if not renew and os.path.exists(path):
            self.logger.info(f"Loading sigma list from {path}")
            return torch.load(path)

        sigma_list = []
        for prototype in self.prototype_list:
            if prototype.sigma is torch.nan:
                sigma_list.append(torch.nan)
            else:
                sigma_list.append(torch.norm(prototype.sigma))
        return sigma_list

    def precompute_distances(self, renew=False) -> list[torch.Tensor]:
        """提前计算所有batch的到各个prototype的距离矩阵

        @return distance_matrix_list: [
            torch.Tensor: [N, B, P]
        ]

        N: number of batches
        B: batch size
        P: number of prototypes
        """

        path = f"{self.out_path}/distance_matrix_{self.device}.pth"
        if not renew and os.path.exists(path):
            self.logger.info(f"Loading distance matrix from {path}")
            return torch.load(path)

        begin_dist_time = time.time()
        # 初始化一个空的列表来存储每个批次的距离矩阵
        distance_matrix_list = []
        for cluster_and_rep, data in zip(self.cluster_and_rep_list, self.data_loader):
            rep = cluster_and_rep.rep
            prototype_gather = self.prototype_list.prototype_gather
            # 获取当前批次的实际大小
            actual_batch_size = data['image'].size(0)
            # 根据实际大小创建距离矩阵
            batch_distance_matrix = torch.zeros(
                (actual_batch_size, len(self.prototype_list))).to(self.device)
            batch_distance_matrix = calculate_distance_matrix(
                rep, prototype_gather)
            # 将计算得到的距离矩阵添加到列表中
            distance_matrix_list.append(batch_distance_matrix)

            del rep, prototype_gather, batch_distance_matrix

        end_dist_time = time.time()
        self.logger.info(
            f"Time for calculating distance matrix: {end_dist_time - begin_dist_time} for device {self.device}")
        # 将列表转换为张量
        distance_matrix = torch.cat(distance_matrix_list, dim=0)
        torch.save(distance_matrix,
                   f"{self.out_path}/distance_matrix_{self.device}.pth")
        del distance_matrix_list
        return distance_matrix

    def precompute_logits(self, renew=False) -> list[torch.Tensor]:
        """提前计算所有batch在不同prompter下的logits

        @return logits_tensor: [N, B, P, C]
        """

        path = f"{self.out_path}/logits_tensor_{self.device}.pth"
        if not renew and os.path.exists(path):
            self.logger.info(f"Loading logits tensor from {path}")
            return torch.load(path)

        begin_logits_time = time.time()
        # 初始化一个空的列表来存储每个批次的logits
        logits_list = []
        for data in self.data_loader:
            image = data['image'].to(self.device)
            # 创建当前批次的logits张量
            for prompter in self.prompter:
                prompted_images = prompter(image)
                batch_logits_tensor = self.model(prompted_images)[
                    :, :self.num_class]
                del prompted_images
            # 将当前批次的logits添加到列表中
            logits_list.append(batch_logits_tensor)
        end_logits_time = time.time()
        self.logger.info(
            f"Time for calculating logits: {end_logits_time - begin_logits_time} for device {self.device}")
        # 将列表中的logits合并为一个张量
        logits_tensor = torch.cat(logits_list, dim=0)
        torch.save(logits_tensor,
                   f"{self.out_path}/logits_tensor_{self.device}.pth")
        del logits_list

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

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode, logger, out_path):
        # super().__init__(prompter, model, device, adapter)
        super().update(self, cluster_and_rep_list, prototype_list, model,
                       prompter, data_loader, batch_size, device, num_class, mode)

    def update_from_base(self, base_agg):
        return super().update_from_base(base_agg)

    def get_prediction(self, index, data_item):
        self.logger.info(f"{self.aggregation_method} Aggregation")
        begin_time = time.time()
        dist_matrix = self.distance_tensor[index].to(self.device)  # [B, P]
        weight = torch.softmax(-dist_matrix, dim=-1)  # [B, P]
        res = super().cal_prediction(
            weight, self.logits_tensor[index].to(self.device))
        self.logger.info(
            f"Time for calculating prediction: {time.time() - begin_time} by nearest aggregation for device {self.device}")

        return res


class gaussianAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "gaussian"

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode, logger, out_path):
        super().update(self, cluster_and_rep_list, prototype_list, model,
                       prompter, data_loader, batch_size, device, num_class, mode)

    def update_from_base(self, base_agg):
        return super().update_from_base(base_agg)

    def get_prediction(self, index, data_item):
        self.logger.info(f"{self.aggregation_method} Aggregation")

        # weight[i][j] = exp(-dist_matrix[i][j] / sigma[j])
        begin_time = time.time()
        dist_matrix = self.distance_tensor[index].to(
            self.device).to(self.device)  # [B, P]
        sigma = torch.stack(self.sigma_list).to(self.device)  # [P]
        weight = torch.exp(-dist_matrix / sigma)  # [B, P]
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        res = super().cal_prediction(
            weight, self.logits_tensor[index].to(self.device))
        self.logger.info(
            f"Time for calculating prediction: {time.time() - begin_time} by gaussian aggregation for device {self.device}")

        return res


class majorityAggregation(BaseAggregation):
    def __init__(self) -> None:
        super().__init__()
        self.aggregation_method = "majority"

    def update(self, cluster_and_rep_list, prototype_list, model, prompter, data_loader, batch_size, device, num_class, mode, logger, out_path):
        super().update(self, cluster_and_rep_list, prototype_list, model,
                       prompter, data_loader, batch_size, device, num_class, mode)

    def update_from_base(self, base_agg):
        return super().update_from_base(base_agg)

    def get_prediction(self, index, data_item):

        self.logger.info(f"{self.aggregation_method} Aggregation")
        begin_time = time.time()
        weight = torch.ones(
            (self.batch_size, len(self.prototype_list))).to(self.device) / len(self.prototype_list)
        res = super().cal_prediction(
            weight, self.logits_tensor[index].to(self.device))
        self.logger.info(
            f"Time for calculating prediction: {time.time() - begin_time} by majority aggregation for device {self.device}")

        return res
