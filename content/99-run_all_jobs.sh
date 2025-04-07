#!/bin/bash

# =============================================================================
# Run Benchmarks
# =============================================================================
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_BCP-N-a100 99-slurm_runner.slurm 01-bench_bcp.py -n 4 8 16 32 64 128 256 512 1024 -s -l
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=BENCH_CLUS-N-a100 99-slurm_runner.slurm 01-bench_clusters.py -n 32 64 128 256 512 -cl 10 20 50 100 200 500 1000 

# =============================================================================
# Validations models (with and without noise)
# =============================================================================
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GRID_valid-N-a100 99-slurm_runner.slurm 02-validation-model.py -n 64 -m GAL020
# with 20% noise
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GRID_noise_20-N-a100 99-slurm_runner.slurm 03-noise-model.py -n 64 -ns 200 -nr 1.0 -m GAL020
# with 50% noise
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GRID_noise_50-N-a100 99-slurm_runner.slurm 03-noise-model.py -n 64 -ns 200 -nr 0.5 -m GAL020

# =============================================================================
# c1d1s1 MODELS
# =============================================================================
# Zone 1 Upper mask of GAL020
sbatch --account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GAL020_U_GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 64 -ns 200 -nr 1.0 -tag c1d1s1 -m GAL020_U -i LiteBIRD
# Zone 2 Lower mask of GAL020
sbatch --account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GAL020_L_GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 64 -ns 200 -nr 1.0 -tag c1d1s1 -m GAL020_L -i LiteBIRD
# Zone 3 Upper mask of GAL040 - GAL020
sbatch --account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GAL040_U_GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 64 -ns 200 -nr 1.0 -tag c1d1s1 -m GAL040_U -i LiteBIRD
# Zone 4 Lower mask of GAL040 - GAL020
sbatch --account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GAL040_L_GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 64 -ns 200 -nr 1.0 -tag c1d1s1 -m GAL040_L -i LiteBIRD
# Zone 5 Upper mask of GAL060 - GAL040
sbatch --account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GAL060_U_GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 64 -ns 200 -nr 1.0 -tag c1d1s1 -m GAL060_U -i LiteBIRD
# Zone 6 Lower mask of GAL060 - GAL040
sbatch --account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100 --job-name=GAL060_L_GRID_noise-N-a100 99-slurm_runner.slurm 04-distributed-gridding.py -n 64 -ns 200 -nr 1.0 -tag c1d1s1 -m GAL060_L -i LiteBIRD
# =============================================================================
# c1d1s1 MODELS
# =============================================================================
# Zone 1 Low galactic represented by GAL020
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GAL020_PTEP_noise-N-a100 99-slurm_runner.slurm 05-PTEP-model.py  -n 64 -ns 200 -nr 1.0 -ud 64 8 4  -tag c1d1s1 -m GAL020 -i LiteBIRD
# Zone 2 Medium galactic represented by GAL040 - GAL020
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GAL040_PTEP_noise-N-a100 99-slurm_runner.slurm 05-PTEP-model.py  -n 64 -ns 200 -nr 1.0 -ud 64 4 2  -tag c1d1s1 -m GAL040 -i LiteBIRD
# Zone 3 High galactic represented by GAL060 - GAL040
sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=GAL060_PTEP_noise-N-a100 99-slurm_runner.slurm 05-PTEP-model.py  -n 64 -ns 200 -nr 1.0 -ud 64 0 2  -tag c1d1s1 -m GAL060 -i LiteBIRD