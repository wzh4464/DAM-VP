###
 # File: /batch.sh
 # Created Date: Saturday December 2nd 2023
 # Author: Zihan
 # -----
 # Last Modified: Sunday, 3rd December 2023 12:34:19 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

#!/bin/bash

# 设置环境变量
export PYTHONPATH=~/miniconda3/envs/dam_vp/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2
export MASTER_ADDR=localhost
export MASTER_PORT=1234
export WORLD_SIZE=3
export RANK=0
export LOCAL_RANK=0

# 确保 output_dir 存在
output_dir="result/cifar100_22k_w_meta"
mkdir -p $output_dir

# 运行分布式 Python 脚本
~/miniconda3/envs/dam_vp/bin/python -m torch.distributed.launch \
    --nproc_per_node=3 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=1234 \
    task_adapting/main.py \
    --output_dir $output_dir \
    --batch_size 96 \
    --base_dir ~/dataset \
    --pretrained_model vit-b-22k \
    --adapt_method prompt_w_head \
    --test_dataset cifar100 \
    --epochs 50 \
    --lr 0.5 \
    --weight_decay 1e-4 \
    --checkpoint_dir checkpoints/vit-b-22k-wo-head.pth \
    --dataset cifar100 \
    --num_gpus 3 \
    --distributed 2> $output_dir/error.txt
