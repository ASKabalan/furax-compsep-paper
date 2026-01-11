#!/bin/bash

# bash validate_fg.sh 1.0 c1d1s1
noise=$1
td_sp=20.0

SKY=$2

# Collect all kmeans job IDs here
job_ids=()

SBATCH_ARGS="--account=apc --partition gpu_v100 --gpus 1"
SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"

# ---------- GAL020 ----------


jid=$(sbatch $SBATCH_ARGS --job-name=FG_20_$noise_$SKY \
     $SLURM_SCRIPT fgbuster-model -n 64 -ns 1 -nr $noise \
     -pc 10000 500 500 -tag $SKY -m GAL020 -i LiteBIRD \
     -sp 1.54 $td_sp -3.0 -mi 2000 -o RESULTS)
job_ids+=("$jid")


# ---------- GAL040 ----------


jid=$(sbatch $SBATCH_ARGS --job-name=FG_40_$noise_$SKY \
     $SLURM_SCRIPT fgbuster-model -n 64 -ns 1 -nr $noise \
     -pc 10000 500 500 -tag $SKY -m GAL040 -i LiteBIRD \
     -sp 1.54 $td_sp -3.0 -mi 2000 -o RESULTS)
job_ids+=("$jid")


# ---------- GAL060 ----------

jid=$(sbatch $SBATCH_ARGS --job-name=FG_60_$noise_$SKY \
     $SLURM_SCRIPT fgbuster-model -n 64 -ns 1 -nr $noise \
     -pc 10000 500 500 -tag $SKY -m GAL060 -i LiteBIRD \
     -sp 1.54 $td_sp -3.0 -mi 2000 -o RESULTS)
job_ids+=("$jid")


# ---------- Final analysis job depending on ALL previous jobs ----------

# Build colon-separated list of job IDs: id1:id2:id3:...
deps=$(IFS=:; echo "${job_ids[*]}")

noise_text=Noiseless
noise_percent=0
if (( $(echo "$noise > 0.0" | bc -l) )); then
    noise_text=Noisy
    noise_percent=100
fi

sbatch --dependency=afterany:$deps \
        $SBATCH_ARGS \
       --job-name=FG_validate_$noise_$SKY \
       $SLURM_SCRIPT r_analysis validate -r fgbuster_${noise_percent}_$SKY -t "FGBuster $noise_text $SKY" \
       -ird RESULTS --noise-ratio $noise --no-tex --scales 1e-4 1e-5
