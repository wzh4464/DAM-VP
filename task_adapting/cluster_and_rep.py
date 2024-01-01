'''
File: /cluster_and_rep.py
Created Date: Monday January 1st 2024
Author: Zihan
-----
Last Modified: Monday, 1st January 2024 5:11:02 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''
import torch
import os


class ClusterAndRep:
    def __init__(self, image, adapter):
        self.rep, self.cluster = self.get_rep_and_cluster(
            image, adapter.prototype_gather, adapter.model)

    def get_rep_and_cluster(self, image, prototype_gather, model):
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
        return rep, cluster_idx


class ClusterAndRepList:
    def __init__(self, path, dataset, adapter):
        if os.path.exists(path):
            self.cluster_and_rep_list = torch.load(path)
        else:
            self.cluster_and_rep_list = [
                ClusterAndRep(data_item["image"].to(
                    adapter.devicename), adapter)
                for data_item in dataset
            ]
            torch.save(self.cluster_and_rep_list, path)

    def __getitem__(self, batch_index):
        return self.cluster_and_rep_list[batch_index]

    def __len__(self):
        return len(self.cluster_and_rep_list)

    def save(self, path):
        torch.save(self.cluster_and_rep_list, path)


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
        """Update items and sigma.

        Args:
            cluster_and_rep_list (ClusterAndRepList): [description]
        """
        self.items = []
        for batch_index, cluster_and_rep in enumerate(cluster_and_rep_list):
            # cluster_and_rep is a batch
            self.items.extend(
                {"batch_index": batch_index, "index_in_batch": index_in_batch}
                for index_in_batch, cluster_and_rep in enumerate(cluster_and_rep)
                if cluster_and_rep.cluster == self.label
            )
        self.sigma = torch.std(
            torch.stack([cluster_and_rep.rep for cluster_and_rep in self.items]), dim=0)

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
