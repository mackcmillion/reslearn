#!/bin/bash

EXPERIMENT_NAME="resnet_110"
DATASET="cifar10"
MODEL="cifar10-resnet-110"

DATA_PATH="/home/max/data"
SUMMARY_PATH="/home/max/summaries"
CHECKPOINT_PATH="/home/max/checkpoints"

mkdir /home/max/logs

echo "Started training for $EXPERIMENT_NAME..."
stdbuf -oL python /home/max/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --train --learning_rate_decay_strategy=1 --initial_learning_rate=0.01 \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 >/home/max/logs/${EXPERIMENT_NAME}.log
echo "Finished training for $EXPERIMENT_NAME."

echo "Started evaluation for $EXPERIMENT_NAME."
stdbuf -oL python /home/max/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --eval --eval_interval_secs=1 \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 >/home/max/logs/${EXPERIMENT_NAME}_eval.log
echo "Finished evaluation for $EXPERIMENT_NAME."