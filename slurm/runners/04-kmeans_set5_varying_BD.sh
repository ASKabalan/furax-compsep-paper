#!/bin/bash
# Set 5: BDXXX_TD500_BS500 (Varying BD)

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
T_DUST_PATCH=500
B_SYNC_PATCH=500
OUTPUT="RESULTS/KMEANS/BDXXX_TD500_BS500"

# Loop over BD values
for VARYING_PATCH in 50 100 200 300 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
do
    echo "Running KMeans with B_DUST Patch = $VARYING_PATCH"
    B_DUST_PATCH=$VARYING_PATCH

    # GAL020
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL040
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL060
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
done

# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S5 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD(\d+)_TD${T_DUST_PATCH}_BS${B_SYNC_PATCH}" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD
