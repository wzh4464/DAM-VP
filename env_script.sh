###
 # File: /env_script.sh
 # Created Date: Friday October 6th 2023
 # Author: Zihan
 # -----
 # Last Modified: Sunday, 12th November 2023 10:50:24 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# install miniconda

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc

# create conda env
~/miniconda3/bin/conda update -n base -c defaults conda -y
~/miniconda3/bin/conda create -n dam_vp python=3.9 -y
~/miniconda3/bin/conda activate dam_vp
~/miniconda3/envs/dam_vp/bin/python -m pip install --upgrade pip
~/miniconda3/envs/dam_vp/bin/pip install -r requirements.txt

# # download dataset
# mkdir data
# cd data
# mkdir FGVC
# cd FGVC
# apt update && apt upgrade -y
# apt install unzip
# unzip FGVC/fgvc_splits.zip -d FGVC
# echo \* >> .gitignore
# mv cub CUB_200_2011
# mv oxfordflower OxfoldFlower
# mv stanfordcars Stanford-cars
# mv stanforddogs Stanford-dogs
# cd ../..

## now tree
# .
# |-- LICENSE
# |-- README.md
# |-- arguments.py
# |-- data
# |   `-- FGVC
# |-- data_utils
# |   |-- datasets
# |   |-- loader.py
# |   |-- transforms.py
# |   |-- vtab_datasets
# |   `-- vtab_prep.py
# |-- env_script.sh
# |-- environment.yaml
# |-- launch.py
# |-- meta-training
# |   |-- main_clip.py
# |   |-- main_hf.py
# |   |-- main_ht.py
# |   |-- meta_clip.py
# |   |-- meta_hf.py
# |   `-- meta_ht.py
# |-- models
# |   |-- backbones
# |   |-- builder.py
# |   |-- model_zoo
# |   `-- prompters.py
# |-- requirements.txt
# |-- task_adapting
# |   |-- adapter.py
# |   |-- adapter_clip.py
# |   |-- diversities.py
# |   |-- main.py
# |   `-- main_clip.py
# `-- utils
#     |-- distributed.py
#     |-- file_io.py
#     |-- functional.py
#     |-- io_utils.py
#     |-- logging.py
#     |-- train_utils.py
#     `-- vis_utils.py

# CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1

# if prompting on vit-b-1k
# python main_hf.py --base_dir /your/path/to/dataset/ --pretrained_model vit-b-1k --meta_lr 0.5 --update_lr 0.5 --update_step 4 --meta_step_size 0.5 --test_dataset oxford-flowers
# ~/miniconda3/envs/dam_vp/bin/python meta-training/main_hf.py --base_dir /workspace/data --pretrained_model vit-b-22k --meta_lr 1.0 --update_lr 1.0 --update_step 4 --meta_step_size 0.5 --weight_decay 1e-4  --test_dataset oxford-flowers
# tar uncompress fgvc-aircraft-2013b.tar.gz to FGVC
# tar -zxvf fgvc-aircraft-2013b.tar.gz -C FGVC

# backbone
onedrive --synchronize --single-directory pth_prompt
onedrive --synchronize --single-directory data
onedrive --synchronize --single-directory backbone_pth
