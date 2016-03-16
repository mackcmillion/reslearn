#!/bin/bash

mkdir /home/max/logs

echo "Started training for resnet_20..."

stdbuf -oL python /home/max/reslearn/main.py --experiment_name=resnet_20 --dataset=cifar10 --model=cifar10-resnet-20 \
 --train \
 --data_path=/home/max/data --summary_path=/home/max/summaries --checkpoint_path=/home/max/checkpoints >/home/max/logs/resnet_20.log

echo "Finished training for resnet_20."

echo "Started evaluation for resnet_20."

stdbuf -oL python /home/max/reslearn/main.py --experiment_name=resnet_20 --dataset=cifar10 --model=cifar10-resnet-20 \
 --eval --eval_interval_secs=1 \
 --data_path=/home/max/data --summary_path=/home/max/summaries --checkpoint_path=/home/max/checkpoints >/home/max/logs/resnet_20_eval.log

echo "Finished evaluation for resnet_20."