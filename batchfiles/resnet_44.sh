#!/bin/bash

EXPERIMENT_NAME="resnet_44"
DATASET="cifar10"
MODEL="cifar10-resnet-44"

DATA_PATH="/mnt/data"
SUMMARY_PATH="/mnt/summaries"
CHECKPOINT_PATH="/mnt/checkpoints"

mkdir /mnt/logs

echo "Started training for $EXPERIMENT_NAME..."
stdbuf -oL python /mnt/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --train \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 >/mnt/logs/${EXPERIMENT_NAME}.log
echo "Finished training for $EXPERIMENT_NAME."

echo "Started evaluation for $EXPERIMENT_NAME."
stdbuf -oL python /mnt/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --eval --eval_interval_secs=1 \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 >/mnt/logs/${EXPERIMENT_NAME}_eval.log
echo "Finished evaluation for $EXPERIMENT_NAME."