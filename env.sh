###
 # File: /script.sh
 # Created Date: Friday January 12th 2024
 # Author: Zihan
 # -----
 # Last Modified: Saturday, 13th January 2024 4:31:50 pm
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

git clone https://github.com/wzh4464/DAM-VP.git

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
rm Miniconda3-latest-Linux-x86_64.sh

cd DAM-VP
git checkout runpod
# bash conda setting
sh -c '. /workspace/miniconda3/etc/profile.d/conda.sh'
source ~/.bashrc
conda init bash
source ~/.bashrc

conda create -n lmeraser -y
conda activate lmeraser

# conda install from environment.yaml
CONDA_ALWAYS_YES="true" conda env update --file environment.yaml

apt update && apt install tmux rsync cadaver unzip -y

curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# az login

az login --use-device-code

mkdir -p models/checkpoints

# az storage blob download --account-name lmeraser --container-name hance --name models/checkpoints/vit_base_p16_224_in22k.pth --file models/checkpoints/vit_base_p16_224_in22k.pth
# az storage blob download --account-name lmeraser --container-name hance --name models/checkpoints/swin_base_patch4_window7_224_22k.pth --file models/checkpoints/swin_base_patch4_window7_224_22k.pth

mkdir -p task_adapting/checkpoints

# az storage blob download --account-name lmeraser --container-name hance --name task_adapting/checkpoints/vit-b-22k-w-head.pth --file task_adapting/checkpoints/vit-b-22k-w-head.pth
# az storage blob download --account-name lmeraser --container-name hance --name task_adapting/checkpoints/swin-b-22k-w-head.pth --file task_adapting/checkpoints/swin-b-22k-w-head.pth

mkdir -p ../dataset

# az storage blob download --account-name lmeraser --container-name hance --name dataset/cifar100.zip --file ../dataset/cifar100.zip

# python -m pip install --upgrade pip
# pip install tqdm simplejson termcolor iopath pandas tkintertable

# git config --global user.name runpod_n
# git config --global user.email 32484940+wzh4464@users.noreply.github.com

### download dataset

## cifar10

bash download_az_file.sh torchvision_dataset/cifar-10-batches-py ~/dataset/torchvision_dataset/cifar-10-batches-py