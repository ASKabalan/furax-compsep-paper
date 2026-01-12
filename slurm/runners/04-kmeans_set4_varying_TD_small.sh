#!/bin/bash
# Set 4: BD4000_TDXX_BS10 (Varying TD, small range)

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=4000
B_SYNC_PATCH=10
OUTPUT="RESULTS/KMEANS/BD4000_TDXX_BS10"

# Loop over TD values
for VARYING_PATCH in 10 20 30 40 50 60 70 80 90 100 120 150 170 200 300 400
do
    echo "Running KMeans with T_DUST Patch = $VARYING_PATCH"
    T_DUST_PATCH=$VARYING_PATCH

    # GAL020
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S4_TD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL040
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S4_TD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL060
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S4_TD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
done

# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S4 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD(\d+)_BS${B_SYNC_PATCH}" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD
