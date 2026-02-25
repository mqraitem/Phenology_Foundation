#!/bin/bash
# Activate your environment

#$ -P ivc-ml
#$ -l gpus=1
#$ -pe omp 4
#$ -j y
#$ -l h_rt=48:00:00
#$ -l gpu_c=8.6

export PATH=/projectnb/ivc-ml/mqraitem/miniconda3/bin:$PATH
source activate geo

# Run your commands
python train_prithvi_conv3d_crops_multiscale.py $args
