'''
File: /aggregation.py
Created Date: Friday, December 29th, 2023
Author: Zihan
-----
Last Modified: Saturday, 30th December 2023 9:54:52 pm
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
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
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

    @abstractmethod
    def get_prediction(self, prompted_images, adapter):
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
        prompted_images = torch.cat(prompted_images, dim=0).to(adapter.devicename)
        if i == 0:
            prompted_images_all = prompted_images.unsqueeze(0)
        else:
            prompted_images_all = torch.cat([prompted_images_all, prompted_images.unsqueeze(0)], dim=0)

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
            prompter_gather[sample["prototype_indices"][idx]](sample["image"][idx].unsqueeze(0))
            for idx in range(batch_size)
        ]
        prompted_images = torch.cat(prompted_images, dim=0).to(adapter.devicename)
        return prompted_images

    def get_prediction(self, prompted_images, adapter):
        """Nearest Neighbor 的 get_prediction

        @return loss: [1]
        """
        return torch.argmax(adapter.model(prompted_images))

class averageAggregation(AggregationStrategy):
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
        # 具体实现 B 的 get_prompted_images
        return get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter)
    def get_prediction(self, prompted_images, adapter):
        """Average Aggregation 的 get_prediction

        @return prediction: [B] (B is the batch size)
        """
        # get all losses for each prompted image
        # losses
        # first dimension: batch
        # second dimension: prototype ind
        # other dimension: loss
        

class majorityAggregation(AggregationStrategy):
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
        """Majority Voting 的 get_prompted_images

        @return prompted_images: [B, P, C, H, W], P is the number of prototypes
        """
        return get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter)
    
    def get_prediction(self, prompted_images, adapter):
        """Majority Voting 的 get_prediction

        @return prediction: [B] (B is the batch size)
        """
        # get all losses for each prompted image
        # losses
        # first dimension: batch
        # second dimension: prototype ind
        # other dimension: loss
        print(torch.cuda.memory_allocated()) 
        # counts [B, P]
        counts = torch.zeros([prompted_images.shape[1], prompted_images.shape[0]]).to(adapter.devicename)
        for i in range(prompted_images.shape[0]):
            counts += adapter.model.forward_features(prompted_images[i])
        
        return torch.argmax(counts, dim=-1)

class gaussianAggregation(AggregationStrategy):
    def get_prompted_images(self, sample, prototype_gather, prompter_gather, adapter):
        # 具体实现 D 的 get_prompted_images
        return get_all_prototyped_prompted_images(sample, prototype_gather, prompter_gather, adapter)