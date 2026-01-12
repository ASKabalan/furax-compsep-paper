#!/bin/bash
# Set 1: BD10000_TD500_BSXXX (Varying BS)

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=10000
T_DUST_PATCH=500
OUTPUT="RESULTS/KMEANS/BD10000_TD500_BSXXX"

# Loop over BS values
for VARYING_PATCH in 10 50 100 150 200 250 300 450 550 600 650 700 750 800 850 900 950 1000 2000 3000 4000 5000 6000
do
    echo "Running KMeans with B_SYNC Patch = $VARYING_PATCH"
    B_SYNC_PATCH=$VARYING_PATCH

    # GAL020
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL040
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL060
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
done

# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S1 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD${T_DUST_PATCH}_BS(\d+)" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD
