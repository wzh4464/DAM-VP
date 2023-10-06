###
 # File: /env_script.sh
 # Created Date: Friday October 6th 2023
 # Author: Zihan
 # -----
 # Last Modified: Friday, 6th October 2023 1:29:13 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

# # install miniconda

# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# ~/miniconda3/bin/conda init bash
# source ~/.bashrc

# # create conda env
# ~/miniconda3/bin/conda update -n base -c defaults conda -y
# ~/miniconda3/bin/conda create -n dam_vp python=3.8 -y
# ~/miniconda3/bin/conda activate dam_vp
# ~/miniconda3/envs/dam_vp/bin/python -m pip install --upgrade pip
# ~/miniconda3/envs/dam_vp/bin/pip install -r requirements.txt

# download dataset
