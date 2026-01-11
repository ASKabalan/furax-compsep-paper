#!/bin/bash

SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"

job_ids=()


# =============================================================================
# MultiRes MODELS (1)
# =============================================================================
# Zone 1 Low galactic represented by GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL020_M1 \
       $SLURM_SCRIPT ptep-model  -n 64 -ns 100 -nr 1.0 -ud 64 0 2 \
       -tag c1d1s1 -m GAL020 -i LiteBIRD -cond -o -mi 1000 -s active_set RESULTS/MULTIRES)
job_ids+=($jid)
# Zone 2 Medium galactic represented by GAL040 - GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL040_M1 \
       $SLURM_SCRIPT ptep-model  -n 64 -ns 100 -nr 1.0 -ud 64 4 2 \
       -tag c1d1s1 -m GAL040 -i LiteBIRD -cond -o -mi 1000 -s active_set RESULTS/MULTIRES)
job_ids+=($jid)
# Zone 3 High galactic represented by GAL060 - GAL040
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL060_M1 \
       $SLURM_SCRIPT ptep-model  -n 64 -ns 100 -nr 1.0 -ud 64 8 4 \
       -tag c1d1s1 -m GAL060 -i LiteBIRD -cond -o -mi 1000 -s active_set RESULTS/MULTIRES)
job_ids+=($jid)
# =============================================================================
# MultiRes MODELS (2)
# =============================================================================
# Zone 1 Low galactic represented by GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL020_M2 \
       $SLURM_SCRIPT ptep-model  -n 64 -ns 100 -nr 1.0 -ud 64 2 2 \
       -tag c1d1s1 -m GAL020 -i LiteBIRD -cond -o -mi 1000 -s active_set RESULTS/MULTIRES)
job_ids+=($jid)
# Zone 2 Medium galactic represented by GAL040 - GAL020
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL040_M2 \
       $SLURM_SCRIPT ptep-model  -n 64 -ns 100 -nr 1.0 -ud 64 8 2 \
       -tag c1d1s1 -m GAL040 -i LiteBIRD -cond -o -mi 1000 -s active_set RESULTS/MULTIRES)
job_ids+=($jid)
# Zone 3 High galactic represented by GAL060 - GAL040
jid=$(sbatch $SBATCH_ARGS --job-name=PTEP_GAL060_M2 \
       $SLURM_SCRIPT ptep-model  -n 64 -ns 100 -nr 1.0 -ud 64 16 4 \
       -tag c1d1s1 -m GAL060 -i LiteBIRD -cond -o -mi 1000 -s active_set RESULTS/MULTIRES)
job_ids+=($jid)

# =============================================================================
# CACHE_RUNS
# =============================================================================


deps=$(IFS=:; echo "${job_ids[*]}")

jid=$(sbatch --dependency=afterany:$deps \
        $SBATCH_ARGS \
       --job-name=CACHE_PTEP \
       $SLURM_SCRIPT r_analysis cache -r ptep -ird RESULTS/MULTIRES -mi 1000 -n 64 -i LiteBIRD -s scipy_tnc)
