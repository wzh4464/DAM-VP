'''
File: /test_torch.py
Created Date: Saturday December 30th 2023
Author: Zihan
-----
Last Modified: Monday, 8th January 2024 2:48:31 pm
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

from argparse import Namespace
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from models import prompters


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train(rank, world_size):
    setup(rank, world_size)

    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载数据集
    train_set = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, sampler=train_sampler)

    # 定义模型
    model = nn.Sequential(nn.Conv2d(3, 32, 3, 1), nn.ReLU(
    ), nn.Flatten(), nn.Linear(32 * 30 * 30, 10)).cuda(rank)
    model = DDP(model, device_ids=[rank])

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 训练模型
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            print(f"{torch.cuda.memory_allocated(), rank}")
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    cleanup()


def test_save_torch_list():
    """Test save torch list.
    """
    a = torch.tensor([1, 2, 3])
    b = torch.tensor([4, 5, 6])
    c = torch.tensor([7, 8, 9])
    torch.save([a, b, c], 'test.pt')
    print(torch.load('test.pt'))


def test_softmax():
    """Test softmax.
    """
    dist_matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(torch.softmax(dist_matrix, dim=-1))
    weight = torch.zeros_like(dist_matrix)
    weight[torch.arange(
        dist_matrix.size(0)), torch.argmin(dist_matrix, dim=-1)] = 1
    print(weight)


def read_and_show_prompter():
    # 加载数据
    path = 'result/cifar10_aggregation_fix'
    checkpoints = torch.load(
        os.path.join(path, 'best_prompter_gather_cuda:2_epoch_0.pth'))

    # which is saved as:
    # torch.save(best_prompter_gather,
    #     f"{self.args.output_dir}/best_prompter_gather_{self.devicename}_epoch_{epoch}.pth")

    # load the prompter (List nn.Module)
    prompter_state_dicts = checkpoints['prompter_state_dicts']
    num_coarse_classes = len(prompter_state_dicts)
    
    args = Namespace()
    args.prompt_size = 30
    args.crop_size = 224

    # prompt_method = 'paddding'

    prompter = [prompters.PadPrompter(args) for _ in range(num_coarse_classes)]

    for idx, p in enumerate(prompter):
        p.load_state_dict(prompter_state_dicts[f'prompter_{idx}'])
        p.cuda()
        p.eval()

    # compare the prompter
    import matplotlib.pyplot as plt
    import numpy as np
    x = torch.zeros(1, 3, args.crop_size, args.crop_size).cuda()

    for idx, p in enumerate(prompter):
        prompt_vis = p.show_prompter_image(
            os.path.join(path, f'prompter_{idx}.png'), x)

        #


if __name__ == "__main__":
    read_and_show_prompter()
