#!/bin/bash
# Set 3: BD4000_TD10_BSXX (Varying BS, small range)

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=4000
T_DUST_PATCH=10
OUTPUT="RESULTS/KMEANS/BD4000_TD10_BSXX"

# Loop over BS values
for VARYING_PATCH in 10 20 30 40 50 60 70 80 90 100 120 150 170 200 300 400
do
    echo "Running KMeans with B_SYNC Patch = $VARYING_PATCH"
    B_SYNC_PATCH=$VARYING_PATCH

    # GAL020
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL040
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL060
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
done

# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S3 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD${T_DUST_PATCH}_BS(\d+)" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD
