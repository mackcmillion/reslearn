#!/bin/bash

echo "Started computation for resnet_110..."

stdbuf -oL python /root/reslearn/main.py --experiment_name=resnet_110 --dataset=cifar10 --model=cifar10-resnet-110 \
 --data_path=/root/data --summary_path=/root/summaries --checkpoint_path=/root/checkpoints >/root/logs/resnet_110.log

echo "Finished computation for resnet_110."