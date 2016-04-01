#!/usr/bin/env bash

EXPERIMENT_NAME="yelp_resnet_20_adam_80000"
DATASET="yelp-small"
MODEL="cifar10-resnet-20"

DATA_PATH="/home/max/data/yelp"
SUMMARY_PATH="/home/max/summaries"
CHECKPOINT_PATH="/home/max/checkpoints"
TRAINING_IMAGE_PATH="${DATA_PATH}/train_photos"
TRAINING_PHOTO_TO_BIZ_ID_PATH="${DATA_PATH}/train_photo_to_biz_ids.csv"
BIZ_ID_LABEL_PATH="${DATA_PATH}/train.csv"
TRAINING_SET="${DATA_PATH}/labelmap_train"
VALIDATION_SET="${DATA_PATH}/labelmap_val"
MEAN_STDDEV_PATH="${DATA_PATH}/mean_stddev"

mkdir /home/max/logs

echo "Started training for $EXPERIMENT_NAME..."
stdbuf -oL python /home/max/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --train \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 --yelp_training_image_path=${TRAINING_IMAGE_PATH} --yelp_training_photo_biz_id_path=${TRAINING_PHOTO_TO_BIZ_ID_PATH} \
 --yelp_biz_id_label_path=${BIZ_ID_LABEL_PATH} \
 --yelp_training_set=${TRAINING_SET} --yelp_validation_set=${VALIDATION_SET} \
 --yelp_mean_stddev_path=${MEAN_STDDEV_PATH} \
 >/home/max/logs/${EXPERIMENT_NAME}.log
echo "Finished training for $EXPERIMENT_NAME."

echo "Started evaluation for $EXPERIMENT_NAME."
stdbuf -oL python /home/max/reslearn/main.py --experiment_name=${EXPERIMENT_NAME} \
 --dataset=${DATASET} --model=${MODEL} \
 --eval --eval_interval_secs=1 \
 --data_path=${DATA_PATH} --summary_path=${SUMMARY_PATH} --checkpoint_path=${CHECKPOINT_PATH} \
 --yelp_training_image_path=${TRAINING_IMAGE_PATH} --yelp_training_photo_biz_id_path=${TRAINING_PHOTO_TO_BIZ_ID_PATH} \
 --yelp_biz_id_label_path=${BIZ_ID_LABEL_PATH} \
 --yelp_training_set=${TRAINING_SET} --yelp_validation_set=${VALIDATION_SET} \
 --yelp_mean_stddev_path=${MEAN_STDDEV_PATH} \
 >/home/max/logs/${EXPERIMENT_NAME}_eval.log
echo "Finished evaluation for $EXPERIMENT_NAME."