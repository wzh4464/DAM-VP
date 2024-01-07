'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Sunday, 7th January 2024 1:06:52 pm
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
    def get_prediction(self, index):
        """
        Retrieves the prediction based on multiple aggregation strategies.

        Returns:
            torch.Tensor: Tensor representing the prediction.

        """
        pass

    @abstractmethod
    def get_actual_batch_size(self, sample):
        """
        Retrieves the actual batch size of the given sample.

        Args:
            sample (dict): Dictionary containing the sample.

        Returns:
            int: Integer representing the actual batch size.

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

        self.num_cluster = len(self.prototype_list)

        with torch.no_grad():
            self.sigma_list = self.load_sigmas()
            self.distance_list = self.precompute_distances()
            self.logits_list = self.precompute_logits()

    def update_from_base(self, base_agg):
        self.distance_list = base_agg.distance_list
        self.logits_list = base_agg.logits_list
        self.sigma_list = base_agg.sigma_list
        self.logger = base_agg.logger
        self.batch_size = base_agg.batch_size
        self.device = base_agg.device
        self.num_class = base_agg.num_class
        self.num_cluster = base_agg.num_cluster

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
        # renew = True
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

        return self.info_and_save(
            'Time for calculating distance matrix: ',
            begin_dist_time,
            distance_matrix_list,
            '/distance_matrix_',
        )

    def precompute_logits(self, renew=False) -> list[torch.Tensor]:
        """提前计算所有batch在不同prompter下的logits

        @return logits_tensor: [N, B, P, C]
        """
        # renew = True
        path = f"{self.out_path}/logits_tensor_{self.device}.pth"
        if not renew and os.path.exists(path):
            self.logger.info(f"Loading logits tensor from {path}")
            return torch.load(path)

        begin_logits_time = time.time()
        # 初始化一个空的列表来存储每个批次的logits
        logits_list = []
        batch_num = len(self.cluster_and_rep_list)
        prototype_num = len(self.prototype_list)
        class_num = self.num_class
        for data in self.data_loader:
            image = data['image'].to(self.device)
            actual_batch_size = image.size(0)
            # 根据实际大小创建logits
            # [P, B, C]
            batch_logits = torch.zeros((prototype_num, actual_batch_size, class_num)).to(self.device)
            for i, prompter in enumerate(self.prompter):
                # [P, B, C]
                batch_logits[i] = self.model(prompter(image))[:, :class_num]
            # 将计算得到的logits添加到列表中
            logits_list.append(batch_logits.permute(1, 0, 2)) # [B, P, C]

            del image, batch_logits

        return self.info_and_save(
            'Time for calculating logits: ',
            begin_logits_time,
            logits_list,
            '/logits_tensor_',
        )

    def info_and_save(self, str_time_for, begin_time, save_list, save_name):
        end_dist_time = time.time()
        self.logger.info(
            f"{str_time_for}{end_dist_time - begin_time} for device {self.device}")
        
        # save list, list item is tensor
        torch.save(save_list, f"{self.out_path}{save_name}{self.device}.pth")

        return save_list

    def get_prediction(self, index):
        """根据不同的聚合策略，返回预测结果

        @return prediction: [B]
        """
        pass

    def cal_prediction(self, weight, logits):
        """根据weight和logits计算预测结果

        @return prediction: [B]
        """
        return torch.argmax(torch.sum(weight.unsqueeze(-1) * logits, dim=1), dim=-1)

    def get_actual_batch_size(self, sample):
        self.actual_batch_size = sample['image'].size(0)
        return self.actual_batch_size


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

    def get_prediction(self, index):
        # self.logger.info(f"{self.aggregation_method} Aggregation")
        begin_time = time.time()
        dist_matrix = self.distance_list[index].to(self.device)  # [B, P]
        weight = torch.softmax(-dist_matrix, dim=-1)  # [B, P]
        res = super().cal_prediction(
            weight, self.logits_list[index].to(self.device))
        # self.logger.info(
        #     f"Time for calculating prediction: {time.time() - begin_time} by nearest aggregation for device {self.device}")

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

    def get_prediction(self, index):
        # self.logger.info(f"{self.aggregation_method} Aggregation")

        # weight[i][j] = exp(-dist_matrix[i][j] / sigma[j])
        begin_time = time.time()
        dist_matrix = self.distance_list[index].to(
            self.device).to(self.device)  # [B, P]
        sigma = torch.stack(self.sigma_list).to(self.device)  # [P]
        weight = torch.exp(-dist_matrix / sigma)  # [B, P]
        weight = weight / torch.sum(weight, dim=-1, keepdim=True)
        res = super().cal_prediction(
            weight, self.logits_list[index].to(self.device))
        # self.logger.info(
        #     f"Time for calculating prediction: {time.time() - begin_time} by gaussian aggregation for device {self.device}")

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

    def get_prediction(self, index):

        # self.logger.info(f"{self.aggregation_method} Aggregation")
        num_cluster = self.num_cluster
        begin_time = time.time()
        actual_batch_size = self.actual_batch_size

        weight = torch.ones(
            (actual_batch_size, num_cluster)).to(self.device) / num_cluster

        res = super().cal_prediction(
            weight, self.logits_list[index].to(self.device))


        # self.logger.info(
        #     f"Time for calculating prediction: {time.time() - begin_time} by majority aggregation for device {self.device}")

        return res
