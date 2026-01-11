#!/bin/bash

SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"
SBATCH_ARGS_MULTI="--account=nih@a100 --nodes=1 --gres=gpu:4 --tasks-per-node=4 -C a100"
# =============================================================================
# Run Benchmarks
# =============================================================================
sbatch $SBATCH_ARGS --job-name=BENCH_BCP-N-a100 $SLURM_SCRIPT bench-bcp -n 4 8 16 32 64 128 256 512 1024 -s -l --jax-solver optax_lbfgs_zoom --fgbuster-solver TNC --noise 0.0
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT bench-clusters -n 32 64 128 256 512 -cl 10 20 50 100 200 500 1000 --jax-solver optax_lbfgs_zoom --fgbuster-solver TNC --noise 0.0

# =============================================================================
# Validations models (with and without noise)
# =============================================================================
sbatch $SBATCH_ARGS --job-name=GRID_valid-N-a100 $SLURM_SCRIPT validation-model -n 64 -m GAL020
# with 20% noise
sbatch $SBATCH_ARGS_MULTI --job-name=GRID_noise_20-N-a100 $SLURM_SCRIPT noise-model -n 64 -ns 20 -nr 0.2 -m GAL020
# with 100% noise
sbatch $SBATCH_ARGS_MULTI --job-name=GRID_noise_100-N-a100 $SLURM_SCRIPT noise-model -n 64 -ns 100 -nr 1.0 -m GAL020
