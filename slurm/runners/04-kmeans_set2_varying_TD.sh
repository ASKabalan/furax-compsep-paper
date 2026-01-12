#!/bin/bash
# Set 2: BD10000_TDXXX_BS500 (Varying TD)

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=10000
B_SYNC_PATCH=500
OUTPUT="RESULTS/KMEANS/BD10000_TDXXX_BS500"

# Loop over TD values
for VARYING_PATCH in 10 50 100 150 200 250 300 450 550 600 650 700 750 800 850 900 950 1000 2000 3000 4000 5000 6000
do
    echo "Running KMeans with T_DUST Patch = $VARYING_PATCH"
    T_DUST_PATCH=$VARYING_PATCH

    # GAL020
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S2_TD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL040
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S2_TD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
    # GAL060
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S2_TD${VARYING_PATCH} $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT)
    job_ids+=("$jid")
done

# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S2 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD(\\d+)_BS${B_SYNC_PATCH}" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD
