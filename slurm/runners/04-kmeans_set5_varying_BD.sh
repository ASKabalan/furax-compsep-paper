#!/bin/bash
# Set 5: BDXXX_TD500_BS500 (Varying BD) - MISSING RUNS ONLY

job_ids=()
BATCH_PARAMS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# Fixed Parameters
T_DUST_PATCH=500
B_SYNC_PATCH=500
OUTPUT="RESULTS/KMEANS/BDXXX_TD500_BS500"

echo "Running missing Set 5 jobs..."

# BD=50, 100, 200, 300, 1000, 1500, 2000 for GAL020 and GAL060
for bd in 50 100 200 300 1000 1500 2000; do
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD$bd $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $bd $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD$bd $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $bd $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
done

# BD=500 for GAL060
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD500 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc 500 $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")

# BD=2500, 3000, 3500, 4000 for GAL060
for bd in 2500 3000 3500 4000; do
    jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD$bd $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc $bd $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL060 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")
done

# BD=5500 for GAL020
jid=$(sbatch $BATCH_PARAMS --job-name=KM_S5_BD5500 $SLURM_SCRIPT $OUTPUT kmeans-model -n 64 -ns 10 -nr 1.0 -pc 5500 $T_DUST_PATCH $B_SYNC_PATCH -tag c1d1s1 -m GAL020 -i LiteBIRD -s active_set  -o $OUTPUT); job_ids+=("$jid")


# Snapshot Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
sbatch --dependency=afterany:$deps $BATCH_PARAMS --job-name=SNAP_S5 \
    $SLURM_SCRIPT $OUTPUT r_analysis snap -r "kmeans_BD(\d+)_TD${T_DUST_PATCH}_BS${B_SYNC_PATCH}" -ird $OUTPUT -mi 2000 -s active_set -n 64 -i LiteBIRD