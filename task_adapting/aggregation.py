'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Friday, 29th December 2023 8:54:21 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from abc import ABC, abstractmethod
import torch


class AggregationStrategy(ABC):
    """
    Abstract base class for aggregation strategies.

    """
    @abstractmethod
    def get_prompted_images(self, rep_batch, prototype_gather, image, prompter_gather):
        """
        Retrieves the prompted images based on the given representations, prototypes, image, and prompter.

        Args:
            rep_batch (torch.Tensor): Tensor of shape [N, D] representing the batch of representations.
            prototype_gather (torch.Tensor): Tensor of shape [M, D] representing the gathered prototypes.
            image (torch.Tensor): Tensor representing the input image.
            prompter_gather (torch.Tensor): Tensor of shape [M, D] representing the gathered prompters.

        Returns:
            torch.Tensor: Tensor representing the prompted images.

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


class nearestAggregation(AggregationStrategy):
    def get_prompted_images(self, rep_batch, prototype_gather, image, prompter_gather):
        # 具体实现 A 的 get_prompted_images
        """Nearest Neighbor.

        @return prompted_images: [B, C, H, W]
        """
        distance_matrix = calculate_distance_matrix(
            rep_batch, prototype_gather)
        indices = torch.argmin(distance_matrix, dim=-1)  # [B]
        batch_size = indices.shape[0]
        prompted_images = [
            prompter_gather[indices[idx]](image[idx].unsqueeze(0))
            for idx in range(batch_size)
        ]
        prompted_images = torch.cat(prompted_images, dim=0)
        return prompted_images


class averageAggregation(AggregationStrategy):
    def get_prompted_images(self, rep_batch, prototype_gather, image, prompter_gather):
        # 具体实现 B 的 get_prompted_images
        raise NotImplementedError


class majorityAggregation(AggregationStrategy):
    def get_prompted_images(self, rep_batch, prototype_gather, image, prompter_gather):
        """Majority Voting 的 get_prompted_images

        @return prompted_images: [B, P, C, H, W], P is the number of prototypes
        """
        # get all prompted images for each image
        # prompted_images
        # first dimension: batch
        # second dimension: prototype ind
        # other dimension: image
        batch_size = rep_batch.shape[0]

        return [
            [
                prompter_gather[j](image[i].unsqueeze(0))
                for j in range(len(prompter_gather))
            ]  # [P, C, H, W]
            for i in range(batch_size)
        ]  # [B, P, C, H, W]


class gaussianAggregation(AggregationStrategy):
    def get_prompted_images(self, rep_batch, prototype_gather, image, prompter_gather):
        # 具体实现 D 的 get_prompted_images
        pass
