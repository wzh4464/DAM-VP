'''
File: /cluster_and_rep.py
Created Date: Monday January 1st 2024
Author: Zihan
-----
Last Modified: Friday, 5th January 2024 11:28:34 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''
import torch
import os
import logging


class ClusterAndRep:
    class ClusterAndRepItem:
        def __init__(self, cluster, rep):
            self.cluster = cluster
            self.rep = rep

    def __init__(self, image, adapter):
        self.rep, self.cluster = self.get_rep_and_cluster(
            image, adapter.prototype_gather, adapter.model, adapter.devicename)
        self.index = 0

    def get_rep_and_cluster(self, image, prototype_gather, model, device):
        """Get representation and cluster index of the image.
        """
        with torch.no_grad():
            rep = model.forward_features(image)
            rep_sum = (rep**2).sum(dim=-1, keepdims=True)
            prototype_gather_sum = (
                prototype_gather**2).sum(dim=-1, keepdims=True).T
            distance_matrix = torch.sqrt(
                rep_sum + prototype_gather_sum - 2 * torch.mm(rep, prototype_gather.T))
            cluster_idx = torch.argmin(distance_matrix, dim=-1)
        return rep.to(device), cluster_idx.to(device)

    def __len__(self):
        return len(self.cluster)

    def __getitem__(self, idx):
        # 确保idx在有效范围内
        if idx < 0 or idx >= self.__len__():
            raise IndexError("Index out of range")
        return self.ClusterAndRepItem(self.cluster[idx], self.rep[idx])

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        result = self[self.index]
        self.index += 1
        return result


class ClusterAndRepList:
    def __init__(self, path, dataset, adapter, renew=False):
        self.index = 0
        if not renew and os.path.exists(f"{path}_devicename_{adapter.devicename}.pth"):
            self.cluster_and_rep_list = torch.load(
                f"{path}_devicename_{adapter.devicename}.pth")
        else:
            self.cluster_and_rep_list = [
                ClusterAndRep(data_item["image"].to(
                    adapter.devicename), adapter)
                for data_item in dataset
            ]
            # torch.save(self.cluster_and_rep_list, os.path.join(
            #     path, f"_devicename_{adapter.devicename}.pth"))
            torch.save(
                self.cluster_and_rep_list,
                f"{path}_devicename_{adapter.devicename}.pth",
            )

    def __getitem__(self, batch_index):
        return self.cluster_and_rep_list[batch_index]

    def __len__(self):
        return len(self.cluster_and_rep_list)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.__len__():
            raise StopIteration
        result = self[self.index]
        self.index += 1
        return result

    def get_all_reps(self):
        return torch.cat([cluster_and_rep.rep for cluster_and_rep in self.cluster_and_rep_list], dim=0)

    def get_all_clusters(self):
        return torch.cat([cluster_and_rep.cluster for cluster_and_rep in self.cluster_and_rep_list], dim=0)


class ProtoType:
    """Clusters from diversity aware (in sense of prototype) clustering.

    Args:
        prototype: center of clusters
        label: label of clusters
        items: [
                    {
                        "batch_index": int,
                        "index_in_batch": int
                    }
                ]
        sigma: standard deviation of clusters
    """

    def __init__(self, prototype: torch.Tensor, label: int, cluster_and_rep_list: ClusterAndRepList):
        self.prototype = prototype
        self.label = label
        self.items = []
        self.sigma = 0
        self.update(cluster_and_rep_list)

    def update(self, cluster_and_rep_list: ClusterAndRepList):
        # 假设cluster_and_rep_list可以直接提供一个合并后的所有批次的cluster数据
        # [total_batches * batch_size]
        all_clusters = cluster_and_rep_list.get_all_clusters()
        # [total_batches * batch_size, rep_dim]
        all_reps = cluster_and_rep_list.get_all_reps()

        # 找到匹配self.label的索引
        matching_indices = (all_clusters == self.label).nonzero().squeeze()

        if matching_indices.numel() < 2:
            # 如果匹配的数量小于2，则不计算标准差
            self.sigma = torch.nan
        else:
            # 取出匹配的rep数据
            matching_reps = all_reps[matching_indices]
            # 计算标准差
            self.sigma = torch.std(matching_reps, dim=0)
            assert self.sigma.shape == self.prototype.shape
            logging.info(
                f"number of matching indices: {matching_indices.numel()} for label {self.label}")
            logging.info(
                f"simga: {self.sigma.shape} for label {self.label}")


# protypes[i] = ProtoType(prototype, label, cluster_and_rep_list : ClusterAndRepList)
# prototype_gather = []
# for i in range(len(coarse_class_idx)):
#     pos = torch.where(y_pred == i)[0]
#     prototype = rep_gather[pos].mean(0).unsqueeze(0)
#     prototype_gather.append(prototype)


class ProtoTypeList:
    def __init__(self, prototype_gather, cluster_and_rep_list: ClusterAndRepList):
        self.prototype_gather = prototype_gather
        self.prototype_list = [
            ProtoType(prototype, label, cluster_and_rep_list)
            for label, prototype in enumerate(prototype_gather)
        ]

    def __getitem__(self, label):
        return self.prototype_list[label]

    def __len__(self):
        return len(self.prototype_list)
