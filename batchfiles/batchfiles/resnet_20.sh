#!/bin/bash

#SBATCH -o /home/hpc/pr63so/ga67zux2/ba/resnet_20.out
#SBATCH -D /home/hpc/pr63so/ga67zux2/ba/reslearn
#SBATCH -J resnet20
#SBATCH --get-user-env
#SBATCH --partition=snb
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END
#SBATCH --mail-user=mumme@in.tum.de
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --nodelist=mac-snb[1-10]

python /home/hpc/pr63so/ga67zux2/ba/reslearn/main.py --experiment_name=resnet_20 --model=cifar10-resnet-20 \
 --data_path=/home/hpc/pr63so/ga67zux2/ba/data --summary_path=/home/hpc/pr63so/ga67zux2/ba/summaries \
 --checkpoint_path=/home/hpc/pr63so/ga67zux2/ba/checkpoints