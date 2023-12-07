#!/bin/bash

# Initialize modules
source /etc/profile

# Loading modules
module purge
module load anaconda/Python-ML-2023b


echo "My SLURM_ARRAY_TASK_ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

python3 experiment.py $LLSUB_RANK $LLSUB_SIZE

