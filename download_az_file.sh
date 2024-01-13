###
 # File: /DAM-VP/download_az_file.sh
 # Created Date: Saturday January 13th 2024
 # Author: Zihan
 # -----
 # Last Modified: Saturday, 13th January 2024 11:50:39 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

#!/bin/bash

# 检查参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_blob_folder_path> <destination_local_folder>"
    exit 1
fi

# 设置变量
SOURCE_FOLDER_PATH=$1 # Blob存储中的文件夹路径
DESTINATION_FOLDER=$2 # 本地目标文件夹路径
ACCOUNT_NAME="lmeraser" # 替换为您的存储账户名
CONTAINER_NAME="hance"  # 替换为您的容器名

# 创建本地目录
mkdir -p "$DESTINATION_FOLDER"

# 列出并下载所有blob
az storage blob list --account-name $ACCOUNT_NAME --container-name $CONTAINER_NAME --prefix "$SOURCE_FOLDER_PATH" --output tsv --query "[].{name:name}" |
while IFS=$'\t' read -r blobname; do
    # 创建本地子目录
    local_path="$DESTINATION_FOLDER/${blobname#$SOURCE_FOLDER_PATH}"
    local_dir=$(dirname "$local_path")
    mkdir -p "$local_dir"

    # 下载blob
    echo "Downloading $blobname to $local_path"
    az storage blob download --account-name $ACCOUNT_NAME --container-name $CONTAINER_NAME --name "$blobname" --file "$local_path"
done
