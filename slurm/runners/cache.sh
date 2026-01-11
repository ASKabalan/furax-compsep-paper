#!/bin/bash

SBATCH_ARGS="--account=apc --partition gpu_v100 --gpus 1"
SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --qos=qos_gpu_a100-dev"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --qos=qos_gpu_h100-dev"


sbatch $SBATCH_ARGS \
       --job-name=FX_cache \
       $SLURM_SCRIPT r_analysis cache -r kmeans fgbuster -ird RESULTS -mi 1000 --no-tex
