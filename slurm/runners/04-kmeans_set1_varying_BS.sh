#!/bin/bash
# Set 1: BD10000_TD500_BSXXX (Varying BS) - MISSING RUNS ONLY

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
B_DUST_PATCH=10000
T_DUST_PATCH=500
OUTPUT="RESULTS/KMEANS/BD10000_TD500_BSXXX"

echo "Running missing Set 1 jobs..."

# BS=100
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS100 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 100 -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS100 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 100 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=150
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS150 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 150 -tag c1d1s1 -m GAL040 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=200
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS200 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 200 -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS200 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 200 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=250
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS250 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 250 -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS250 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 250 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=300
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS300 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 300 -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BS=1000
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S1_BS1000 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $B_DUST_PATCH $T_DUST_PATCH 1000 -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")


# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S1 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD${B_DUST_PATCH}_TD${T_DUST_PATCH}_BS(\d+)" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD