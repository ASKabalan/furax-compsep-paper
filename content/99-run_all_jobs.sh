#!/bin/bash

# Run Benchmarks
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 01-bench_bcp.py -n 4 8 16 32 64 128 256 512 1024 -s -l
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --job-name=BENCH_CLUS-N-a100 99-slurm_runner.slurm 01-bench_clusters.py -n 32 64 128 256 512 -cl 10 20 50 100 200 500 1000 

# Validations models (with and without noise)
sbatch --account=nih@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GRID_valid-N-a100 99-slurm_runner.slurm 02-validation-model.py -n 64 -ns 50
sbatch --account=nih@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GRID_noise-N-a100 99-slurm_runner.slurm 03-noise-model.py -n 64 -ns 50 -nr 0.2

# c1d1s1 MODELS
sbatch --account=nih@a100 --nodes=1 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 128 -ns 50 -nr 0.2 -tag c1d1s1 -m GAL020_U -i LiteBIRD