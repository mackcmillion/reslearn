#!/bin/bash

mkdir /root/logs

echo "Started computation for resnet_32..."

stdbuf -oL python /root/reslearn/main.py --experiment_name=resnet_32 --dataset=cifar10 --model=cifar10-resnet-32 \
 --data_path=/root/data --summary_path=/root/summaries --checkpoint_path=/root/checkpoints >/root/logs/resnet_32.log

echo "Finished computation for resnet_32."