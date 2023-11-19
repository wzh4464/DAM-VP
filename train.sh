###
 # File: /train.sh
 # Created Date: Sunday November 19th 2023
 # Author: Zihan
 # -----
 # Last Modified: Sunday, 19th November 2023 10:32:23 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# "args": [
#                 "-m",
#                 "torch.distributed.launch",
#                 "task_adapting/main.py",
#                 "--nproc_per_node=3",
#                 "--nnodes=1",
#                 "--node_rank=0",
#                 "--master_addr=localhost",
#                 "--master_port=1234",
#                 "--base_dir",
#                 "../dataset",
#                 "--pretrained_model",
#                 "vit-b-1k",
#                 "--adapt_method",
#                 "prompt_wo_head",
#                 "--test_dataset",
#                 "oxford-flowers",
#                 "--epochs",
#                 "50",
#                 "--lr",
#                 "0.5",
#                 "--weight_decay",
#                 "1e-4",
#                 "--checkpoint_dir",
#                 "checkpoints/vit-b-1k-wo-head.pth",
#                 "--dataset",
#                 "vtab-caltech101",
#                 "--num_gpus",
#                 "3"
#             ],
                # "CUDA_VISIBLE_DEVICES": "0,1,2",
                # "MASTER_ADDR": "localhost",
                # "MASTER_PORT": "1234",
                # "WORLD_SIZE": "3",
                # "RANK": "0",
                # "LOCAL_RANK": "0",
CUDA_VISIBLE_DEVICES=0,1,2 \
MASTER_ADDR=localhost \
MASTER_PORT=1234 \
WORLD_SIZE=3 \
RANK=0 \
/home/zihan/miniconda3/envs/dam_vp/bin/torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=1234 \
    task_adapting/main.py \
    --base_dir ../dataset \
    --pretrained_model vit-b-1k \
    --adapt_method prompt_wo_head \
    --test_dataset oxford-flowers \
    --epochs 50 \
    --lr 0.5 \
    --weight_decay 1e-4 \
    --checkpoint_dir checkpoints/vit-b-1k-wo-head.pth \
    --dataset vtab-caltech101 \
    --num_gpus 3
