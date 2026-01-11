#!/bin/bash
# Corrected Bash loop over BS values

# Collect all kmeans job IDs here
job_ids=()
BATCH_PARAMS='--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100'
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

### =============================================================================
# KMEANS Patch MODELS with varying T_DUST_PATCH
# =============================================================================

for VARYING_PATCH in 50 100 200 300 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
do
    echo "Running KMeans with Patch = $VARYING_PATCH"

    B_DUST_PATCH=$VARYING_PATCH
    T_DUST_PATCH=500
    B_SYNC_PATCH=500

    jid=$(sbatch $BATCH_PARAMS --job-name=KMEANS_$VARYING_PATCH$-h100 $SLURM_SCRIPT kmeans-model -n 64 -ns 100 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o RESULTS/KMEANS)
    job_ids+=("$jid")
    # Zone 2 mask of GAL040 - GAL020
    jid=$(sbatch $BATCH_PARAMS --job-name=KMEANS_$VARYING_PATCH$-h100 $SLURM_SCRIPT kmeans-model -n 64 -ns 100 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o RESULTS/KMEANS)
    job_ids+=("$jid")
    # Zone 3 mask of GAL060 - GAL040
    jid=$(sbatch $BATCH_PARAMS --job-name=KMEANS_$VARYING_PATCH$-h100 $SLURM_SCRIPT kmeans-model -n 64 -ns 100 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o RESULTS/KMEANS)
    job_ids+=("$jid")
done

# =============================================================================
# CACHE KMEANS RESIUDALS
# =============================================================================
# Build colon-separated list of job IDs: id1:id2:id3:...
deps=$(IFS=:; echo "${job_ids[*]}")

sbatch --dependency=afterany:$deps \
       $BATCH_PARAMS \
       --job-name=KM_varying_patch_h100 \
       $SLURM_SCRIPT r_analysis cache -r kmeans -ird RESULTS/KMEANS -mi 2000 -s scipy_tnc
