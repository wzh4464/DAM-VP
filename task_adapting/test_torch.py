'''
File: /test_torch.py
Created Date: Saturday December 30th 2023
Author: Zihan
-----
Last Modified: Sunday, 7th January 2024 11:36:12 am
Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
-----
HISTORY:
Date      		By   	Comments
----------		------	---------------------------------------------------------
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import os


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


if __name__ == "__main__":
    test_save_torch_list()
