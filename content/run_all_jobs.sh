#!/bin/bash
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --job-name=BENCH-N-a100 01-bench.slurm
sbatch --account=nih@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GRID-N-a100 02-distributed-gridding.slurm
