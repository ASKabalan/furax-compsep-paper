#!/bin/bash

SBATCH_ARGS="--account=nih@a100 --nodes=4 --gres=gpu:8 --tasks-per-node=8 -C a100"

job_ids=()

# =============================================================================
# grid_search MODELS on both parts UPPER and LOWER
# =============================================================================
# Zone 1 mask of GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=GAL020_GRID_noise-N-a100 \
       $SLURM_SCRIPT distributed-gridding -n 64 -ns 100 -nr 1.0 \
       -tag c1d1s1 -m GAL020 -i LiteBIRD -cond \
       -ss SEARCH_SPACE.yml -s active_set -mi 1000 -o RESULTS/GRID_SEARCH)
job_ids+=($jid)
# Zone 2 mask of GAL040 - GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=GAL040_GRID_noise-N-a100 \
       $SLURM_SCRIPT distributed-gridding -n 64 -ns 100 -nr 1.0 \
       -tag c1d1s1 -m GAL040 -i LiteBIRD -cond \
       -ss SEARCH_SPACE.yml -s active_set -mi 1000 -o RESULTS/GRID_SEARCH)
job_ids+=($jid)
# Zone 3 mask of GAL060 - GAL0
jid=$(sbatch $SBATCH_ARGS --job-name=GAL060_GRID_noise-N-a100 \
       $SLURM_SCRIPT distributed-gridding -n 64 -ns 100 -nr 1.0 \
       -tag c1d1s1 -m GAL060 -i LiteBIRD -cond \
       -ss SEARCH_SPACE.yml -s active_set -mi 1000 -o RESULTS/GRID_SEARCH)
job_ids+=($jid)
# Zone 4 mask of GALACTIC
jid=$(sbatch $SBATCH_ARGS --job-name=GALACTIC_GRID_noise-N-a100 \
       $SLURM_SCRIPT distributed-gridding -n 64 -ns 100 -nr 1.0 \
       -tag c1d1s1 -m GALACTIC -i LiteBIRD -cond \
       -ss SEARCH_SPACE.yml -s active_set -mi 1000 -o RESULTS/GRID_SEARCH)
job_ids+=($jid)

# =============================================================================
# CACHE_RUNS
# =============================================================================

deps=$(IFS=:; echo "${job_ids[*]}")

jid=$(sbatch --dependency=afterany:$deps \
        $SBATCH_ARGS \
       --job-name=CACHE_GRIDDING \
       $SLURM_SCRIPT r_analysis cache -r compsep -ird RESULTS/GRID_SEARCH -mi 1000 -n 64 -i LiteBIRD -s scipy_tnc)



# =============================================================================
# SNAPSHOT RUNS
# =============================================================================

sbatch --dependency=afterok:$jid \
        $SBATCH_ARGS \
       --job-name=PTEP_SNAP
         $SLURM_SCRIPT r_analysis snap -n 64 -i LiteBIRD -ird RESULTS/MULTIRES -r ptep -o SNAPSHOT
