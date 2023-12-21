###
 # File: /idle.sh
 # Created Date: Sunday December 3rd 2023
 # Author: Zihan
 # -----
 # Last Modified: Sunday, 3rd December 2023 12:32:30 am
 # Modified By: the developer formerly known as Zihan at <wzh4464@gmail.com>
 # -----
 # HISTORY:
 # Date      		By   	Comments
 # ----------		------	---------------------------------------------------------
###

#!/bin/bash

# 连续低占用次数计数
low_usage_count=0

# 检测 GPU 占用的函数
check_gpu_usage() {
    local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # 检查每个 GPU 的占用率是否都小于 20%
    for usage in $gpu_usage; do
        if [ $usage -ge 20 ]; then
            # 如果任一 GPU 占用率 ≥ 20%，则重置计数器并返回 1
            low_usage_count=0
            return 1
        fi
    done

    # 所有 GPU 占用率都小于 20%，计数加 1
    ((low_usage_count++))

    # 如果连续三次都低于 20%，返回 0
    if [ $low_usage_count -ge 3 ]; then
        return 0
    else
        return 1
    fi
}

# 主循环
while true; do
    if check_gpu_usage; then
        echo "GPU 占用率连续三次低于 20%，运行 batch.sh..."
        /home/zihan/DAM-VP/batch.sh
        break
    else
        echo "等待 GPU 占用率降低..."
    fi

    # 每分钟检查一次
    sleep 60
done
