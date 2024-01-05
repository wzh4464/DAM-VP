###
 # File: /batch.sh
 # Created Date: Saturday December 2nd 2023
 # Author: Zihan
 # -----
 # Last Modified: Friday, 5th January 2024 9:15:07 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export MASTER_ADDR=localhost
export MASTER_PORT=1234
export WORLD_SIZE=3
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=200

# 确保 output_dir 存在
output_dir="result/cifar10_gaussian_pth_saved_distributed"
mkdir -p $output_dir

# 运行分布式 Python 脚本
/home/zihan/dataset/new_dam/bin/python -m torchrun \
    task_adapting/main.py \
    --output_dir $output_dir \
    --batch_size 96 \
    --base_dir ~/dataset \
    --pretrained_model vit-b-22k \
    --adapt_method prompt_w_head \
    --test_dataset cifar10 \
    --epochs 50 \
    --lr 0.5 \
    --weight_decay 1e-4 \
    --checkpoint_dir checkpoints/vit-b-22k-w-head.pth \
    --dataset cifar10 \
    --num_gpus 3 \
    --distributed \
    --num_workers 3 2> $output_dir/error.log
