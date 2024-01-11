#!/bin/bash
#SBATCH --job-name=cifar100-single      # 作业名称
#SBATCH --nodes=1                      # 使用的节点数
#SBATCH --ntasks-per-node=1            # 每个节点的任务数
#SBATCH --gres=gpu:1                   # 每个节点使用的 GPU 数量
#SBATCH --output=result/hpc_log/cifar100_output_%j.txt # 标准输出和错误的文件名
#SBATCH --time=24:00:00                # 预计作业运行时间
#SBATCH --mem=102400                      # 内存要求
#SBATCH --partition=gpu-v100s-test           # 选择 partition


module load cuda/11.6.0

/home/zihanwu7/miniconda3/envs/dam-vp/bin/python \
    /home/zihanwu7/DAM-VP/task_adapting/main.py \
    --output_dir result/original_cifar100_1k_without_head_128_single \
    --batch_size 128 \
    --base_dir ~/dataset \
    --pretrained_model vit-b-1k \
    --adapt_method prompt_wo_head \
    --epochs 50 \
    --lr 0.5 \
    --weight_decay 1e-4 \
    --dataset cifar100 \
    --checkpoint_dir checkpoints/vit-b-1k-wo-head.pth \
    --num_gpus 1 \
    --num_workers 1
