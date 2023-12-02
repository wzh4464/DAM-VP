###
 # File: /train.sh
 # Created Date: Sunday December 3rd 2023
 # Author: Zihan
 # -----
 # Last Modified: Sunday, 3rd December 2023 12:16:54 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

#!/bin/bash

# 设置环境变量
export PYTHONPATH=~/miniconda3/envs/dam_vp/bin/python
export CUDA_VISIBLE_DEVICES=0  # 只使用一块 GPU
export MASTER_ADDR=localhost   # 可以保留，对于单卡不是必需
export MASTER_PORT=1234        # 可以保留，对于单卡不是必需
# 以下环境变量在单卡运行中不再需要
# export WORLD_SIZE=3
# export RANK=0
# export LOCAL_RANK=0

# 确保 output_dir 存在
output_dir="result/cifar100_22k_wo_meta"
mkdir -p $output_dir

# 运行单卡 Python 脚本
~/miniconda3/envs/dam_vp/bin/python task_adapting/main.py \
    --output_dir $output_dir \
    --batch_size 32 \
    --wo_da \
    --base_dir dataset \
    --pretrained_model vit-b-22k \
    --adapt_method prompt_w_head \
    --test_dataset cifar100 \
    --epochs 50 \
    --lr 0.5 \
    --weight_decay 1e-4 \
    --checkpoint_dir checkpoints/vit-b-22k-wo-head.pth \
    --dataset cifar100 \
    --num_gpus 1 \  # 更新 GPU 数量
    --distributed 2> $output_dir/error.txt
