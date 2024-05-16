###
 # File: /diversities.py
 # Created Date: Thursday, May 16th 2024
 # Author: Zihan
 # -----
 # Last Modified: Thursday, 16th May 2024 11:38:04 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

from __future__ import print_function
import os
import sys
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
import timm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from arguments import Arguments
from utils.functional import set_seed
from models import *
from data_utils import loader as data_loader
import lpips

class util_of_lpips():
    def __init__(self, custom_model, device):
        """Learned Perceptual Image Patch Similarity, LPIPS with custom model (ViT-B/22k).
        
        args:
            custom_model: nn.Module, custom model for feature extraction
            device: torch.device, device to run the model on
        """
        self.loss_fn = custom_model.to(device)
        self.device = device
        self.scaling_layer = lpips.LPIPS().scaling_layer.to(device)

    def calc_lpips(self, img_batch1, img_batch2):
        """LPIPS distance calculator with custom model.
        
        args:
            img_batch1 : tensor
            img_batch2 : tensor
        """
        img_batch1 = self.scaling_layer(img_batch1.to(self.device))
        img_batch2 = self.scaling_layer(img_batch2.to(self.device))
        feats0 = self.loss_fn(img_batch1)
        feats1 = self.loss_fn(img_batch2)
        diffs = [(f0 - f1) ** 2 for f0, f1 in zip(feats0, feats1)]
        res = [torch.mean(diff) for diff in diffs]
        res = sum(res)
        return res

@torch.no_grad()
def main(test_dataset):
    """Task Adaption on the downstream dataset.
    """

    # load datasets for diversity calculation
    minis_test = data_loader.construct_train_loader(args, test_dataset)

    # Load the pretrained ViT model
    vit_model = timm.create_model('vit_base_patch16_224_in21k', pretrained=True, num_classes=0).to(args.device)

    # introduce LPIPS with custom model
    lpips_func = util_of_lpips(custom_model=vit_model, device=args.device)

    # randomly select and obtain the diversity using average lpips
    dist_total = 0
    num_total = 0
    for i, sample in enumerate(minis_test):
        image = sample["image"].to(args.device)
        order = torch.randperm(image.size(0))
        image_shuffled = image[order]

        dist = lpips_func.calc_lpips(image.detach(), image_shuffled.detach())
        dist = dist[dist != 0]
        dist_total += dist.sum()
        num_total += dist.size(0)
        print(dist.sum().item()/dist.size(0))
        if i == 1000 // args.batch_size - 1:
            break

    dist_total /= num_total
    print("Data diversity score of {}: {}".format(test_dataset, dist_total))

if __name__ == '__main__':
    args = Arguments(stage='task_adapting').parser().parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu')

    # basic configuration
    set_seed(args.seed)

    # main loop
    dataset_list = ['cifar10', 'cifar100', 'svhn', 'gtsrb']
    for test_dataset in dataset_list:
        main(test_dataset)
