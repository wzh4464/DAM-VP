###
 # File: /script.sh
 # Created Date: Friday January 12th 2024
 # Author: Zihan
 # -----
 # Last Modified: Saturday, 13th January 2024 12:16:48 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

git clone https://github.com/wzh4464/DAM-VP.git

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
bash Miniconda3-latest-Linux-x86_64.sh -b && \
rm Miniconda3-latest-Linux-x86_64.sh

cd DAM-VP
git checkout runpod
# bash conda setting
sh -c '. /root/miniconda3/etc/profile.d/conda.sh ; conda activate base ; conda init bash'
source ~/.bashrc

conda create -n lmeraser
conda activate lmeraser

# conda install from environment.yaml
conda env update --file environment.yaml

apt install tmux -y

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# az login

