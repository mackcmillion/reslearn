#!/bin/bash

mkdir /root/logs

echo "Started computation for resnet_44..."

stdbuf -oL python /root/reslearn/main.py --experiment_name=resnet_44 --dataset=cifar10 --model=cifar10-resnet-44 \
 --data_path=/root/data --summary_path=/root/summaries --checkpoint_path=/root/checkpoints >/root/logs/resnet_44.log

echo "Finished computation for resnet_44."