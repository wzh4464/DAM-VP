'''
File: /cluster_and_rep.py
Created Date: Monday January 1st 2024
Author: Zihan
-----
Last Modified: Monday, 1st January 2024 4:18:10 pm
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
        self.rep, self.cluster = adapter.get_rep_and_cluster(
            image, adapter.prototype_gather)


class ClusterAndRepList:
    def __init__(self, path, dataset, adapter):
        if os.path.exists(path):
            self.cluster_and_rep_list = torch.load(path)
        else:
            self.cluster_and_rep_list = [
                ClusterAndRep(data_item["image"].to(adapter.devicename), adapter)
                for data_item in dataset
            ]
            torch.save(self.cluster_and_rep_list, path)

    def __getitem__(self, batch_index):
        return self.cluster_and_rep_list[batch_index]

    def __len__(self):
        return len(self.cluster_and_rep_list)

    def save(self, path):
        torch.save(self.cluster_and_rep_list, path)
