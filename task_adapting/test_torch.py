'''
File: /test_torch.py
Created Date: Saturday December 30th 2023
Author: Zihan
-----
Last Modified: Monday, 8th January 2024 8:39:49 pm
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
# from torchvision import datasets, transforms
import os
import sys
import numpy as np

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


def compare_two_prompter_list(file1, file2):
    """Compare two prompter list.
    """
    prompter_list1 = torch.load(file1, map_location='cpu')
    prompter_list2 = torch.load(file2, map_location='cpu')

    args = Namespace()
    args.prompt_size = 30
    args.crop_size = 224

    # compare the prompter
    x = torch.zeros(1, 3, args.crop_size, args.crop_size)
    diff_list = []

    for i, p1, p2 in zip(range(len(prompter_list1)), prompter_list1, prompter_list2):
        prompted1 = p1(x)
        prompted2 = p2(x)
        prompted1 = prompted1.squeeze(0).permute(1, 2, 0).detach().numpy()
        prompted2 = prompted2.squeeze(0).permute(1, 2, 0).detach().numpy()
        diff = np.max(np.abs(prompted1 - prompted2))
        print(f"prompter {i} diff: {diff}")
        diff_list.append(diff)

    print(f"max diff: {max(diff_list)}")


def prompters_std(path, file):
    # 加载数据
    prompter_list = torch.load(
        os.path.join(path, file), map_location='cpu')

    args = Namespace()
    args.prompt_size = 30
    args.crop_size = 224

    # compare the prompter
    x = torch.zeros(1, 3, args.crop_size, args.crop_size)

    prompted_x_list = []
    for i, p in enumerate(prompter_list):
        prompted_x = p(x)
        prompted_x_list.append(prompted_x)
        if i == 0:
            print(f"prompted_x shape: {prompted_x.shape}")

    prompted_x_tensor = torch.cat(prompted_x_list, dim=0)
    print(f"prompted_x_tensor shape: {prompted_x_tensor.shape}")  # "torch.Size([2, 30, 224, 224])

    std = torch.std(prompted_x_tensor, dim=0)
    std_norm = torch.norm(std, dim=(1, 2))
    print(f"std_norm is {std_norm}")



def read_and_show_prompter_old(path, file):
    # path = 'result/reset_head'
    checkpoints = torch.load(
        os.path.join(path, file), map_location='cpu')

    args = Namespace()
    args.prompt_size = 30
    args.crop_size = 224

    prompter_state_dicts = checkpoints['prompter_state_dicts']
    key_prefix = 'prompter_'

    prompter_list = []

    for idx in range(len(prompter_state_dicts)):
        prompter = prompters.PadPrompter(args)
        prompter.load_state_dict(prompter_state_dicts[key_prefix + str(idx)])
        prompter_list.append(prompter)

    x = torch.zeros(1, 3, args.crop_size, args.crop_size)

    p1 = prompter_list[0]
    p2 = prompter_list[1]

    prompted1 = p1(x)
    prompted2 = p2(x)

    diff = prompted1 - prompted2

    max_diff = torch.max(diff)

    print(max_diff)


if __name__ == "__main__":
    prompters_std("result/cifar10_without_head_origin",
                  "best_prompter_gather_cuda:0_epoch_19.pth")
