#!/bin/bash
# Set 3: BD4000_TD10_BSXX (Varying BS, small range) - MISSING RUNS ONLY

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=4000
T_DUST_PATCH=10
OUTPUT="RESULTS/KMEANS/BD4000_TD10_BSXX"

echo "Running missing Set 3 jobs..."

# BS 40 to 100 for GAL040
for bs in 40 50 60 70 90 100; do
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS$bs $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $bs -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
done

# BS=120
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS120 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 120 -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS120 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 120 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=150, 170
for bs in 150 170; do
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS$bs $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH $bs -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
done

# BS=200
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS200 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 200 -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS200 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 200 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=300
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS300 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 300 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=400
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S3_BS400 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 400 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")


# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S3 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD${T_DUST_PATCH}_BS(\d+)" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD