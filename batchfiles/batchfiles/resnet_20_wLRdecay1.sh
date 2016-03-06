#!/usr/bin/env bash

#!/bin/bash

echo "Started computation for resnet_20 with learning rate decay strategy 1..."

stdbuf -oL python /root/reslearn/main.py --experiment_name=resnet_20_wLRdecay1 --dataset=cifar10 --model=cifar10-resnet-20 \
 --data_path=/root/data --summary_path=/root/summaries --checkpoint_path=/root/checkpoints >/root/logs/resnet_20_wLRdecay1.log

echo "Finished computation for resnet_20 with learning rate decay strategy 1."