#!/bin/bash

mkdir /home/max/logs

echo "Started computation for resnet_110..."

stdbuf -oL python /home/max/reslearn/main.py --experiment_name=resnet_110 --dataset=cifar10 --model=cifar10-resnet-110 \
 --data_path=/home/max/data --summary_path=/home/max/summaries --checkpoint_path=/home/max/checkpoints >/home/max/logs/resnet_110.log

echo "Finished computation for resnet_110."