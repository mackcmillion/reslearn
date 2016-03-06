#!/bin/bash

mkdir /root/logs

echo "Started computation for resnet_56..."

stdbuf -oL python /root/reslearn/main.py --experiment_name=resnet_56 --dataset=cifar10 --model=cifar10-resnet-56 \
 --data_path=/root/data --summary_path=/root/summaries --checkpoint_path=/root/checkpoints >/root/logs/resnet_56.log

echo "Finished computation for resnet_56."