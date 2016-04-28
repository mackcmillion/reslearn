#!/usr/bin/env bash

EXPERIMENT_NAME="yelp_resnet_18_eval_once"
DATASET="yelp"
MODEL="resnet-18"

DATA_PATH="/home/max/data"
SUMMARY_PATH="/home/max/summaries"
CHECKPOINT_PATH="/home/max/checkpoints"

echo "Started evaluation for $EXPERIMENT_NAME."
stdbuf -oL python /home/max/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --eval --eval_interval_secs=1 \
 --num_consuming_threads=4 --min_frac_examples_in_queue=0.001 \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 >/home/max/logs/${EXPERIMENT_NAME}_eval.log
echo "Finished evaluation for $EXPERIMENT_NAME."