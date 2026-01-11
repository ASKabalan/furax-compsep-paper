SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --exclusive"
# =============================================================================
# Run Benchmarks
# =============================================================================
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT bench-clusters -n 32 64 128 256 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --fgbuster-solver TNC  --noise 1.0
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT bench-clusters -n 32 64 128 256 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver active_set --noise 1.0
sbatch $SBATCH_ARGS --job-name=BENCH_CLUS-N-a100 $SLURM_SCRIPT bench-clusters -n 32 64 128 256 -cl 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 --jax-solver scipy_tnc --noise 1.0

# =============================================================================
# Validations models (with and without noise)
# =============================================================================
sbatch $SBATCH_ARGS --job-name=GRID_valid-N-a100 $SLURM_SCRIPT validation-model -n 64 -m GAL020
# with 20% noise
sbatch $SBATCH_ARGS_MULTI --job-name=GRID_noise_20-N-a100 $SLURM_SCRIPT noise-model -n 64 -ns 20 -nr 0.2 -m GAL020
# with 100% noise
sbatch $SBATCH_ARGS_MULTI --job-name=GRID_noise_100-N-a100 $SLURM_SCRIPT noise-model -n 64 -ns 100 -nr 1.0 -m GAL020
