#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --tasks-per-node=16
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=serc
#SBATCH --output=%a.log
export DATADIR=/scratch/users/robcking/cs229_data
module load pytorch/1.11.0_py39
python3 train.py $@