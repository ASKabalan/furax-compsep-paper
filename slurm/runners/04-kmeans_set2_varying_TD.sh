#!/bin/bash
# Set 2: BD10000_TDXXX_BS500 (Varying TD) - MISSING RUNS ONLY

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=10000
B_SYNC_PATCH=500
OUTPUT="RESULTS/KMEANS/BD10000_TDXXX_BS500"

echo "Running missing Set 2 jobs..."

# TD=5000
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S2_TD5000 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH 5000 $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# TD=6000
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S2_TD6000 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH 6000 $B_SYNC_PATCH -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")


# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S2 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD(\d+)_BS${B_SYNC_PATCH}" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD