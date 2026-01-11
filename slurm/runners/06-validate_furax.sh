#!/bin/bash
PYTHONUNBUFFERED=1
# bash validate_fg.sh 1.0 c1d1s1
noise=$1
td_sp=20.0

SKY=$2
SOLVER=$3

# Collect all kmeans job IDs here
job_ids=()

SBATCH_ARGS="--account=apc --partition gpu_v100 --gpus 1"
SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --qos=qos_gpu_a100-dev"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# ---------- GAL020 ----------


jid=$(sbatch $SBATCH_ARGS --job-name=FX_20_$noise_$SKY_$SOLVER \
     $SLURM_SCRIPT kmeans-model -n 64 -ns 1 -nr $noise \
     -pc 10000 500 500 -tag $SKY -m GAL020 -i LiteBIRD \
     -sp 1.54 $td_sp -3.0 -mi 2000 -o RESULTS -s $SOLVER)
job_ids+=("$jid")


# ---------- GAL040 ----------


jid=$(sbatch $SBATCH_ARGS --job-name=FX_40_$noise_$SKY_$SOLVER \
     $SLURM_SCRIPT kmeans-model -n 64 -ns 1 -nr $noise \
     -pc 10000 500 500 -tag $SKY -m GAL040 -i LiteBIRD \
     -sp 1.54 $td_sp -3.0 -mi 2000 -o RESULTS -s $SOLVER)
job_ids+=("$jid")


# ---------- GAL060 ----------

jid=$(sbatch $SBATCH_ARGS --job-name=FX_60_$noise_$SKY_$SOLVER \
     $SLURM_SCRIPT kmeans-model -n 64 -ns 1 -nr $noise \
     -pc 10000 500 500 -tag $SKY -m GAL060 -i LiteBIRD \
     -sp 1.54 $td_sp -3.0 -mi 2000 -o RESULTS -s $SOLVER)
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
       --job-name=FX_validate_$noise_$SKY_$SOLVER \
       $SLURM_SCRIPT r_analysis validate -r kmeans_noise${noise_percent}_${SKY} -t "FURAX $noise_text $SKY" \
       -ird RESULTS --noise-ratio $noise --no-tex --scales 1e-4 1e-5


sbatch --dependency=afterany:$deps \
        $SBATCH_ARGS \
       --job-name=FX_cache_$noise_$SKY_$SOLVER \
       $SLURM_SCRIPT r_analysis cache -r kmeans -ird RESULTS -mi 2000 --no-tex -s scipy_tnc
