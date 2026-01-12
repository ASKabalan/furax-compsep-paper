#!/bin/bash
SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --qos=qos_gpu_h100-dev"



PTEP='ptep'
kmeans_varying_patch='kmeans_c1d1s1_BD(\d+)_TD500_BS500'
FGBUSTER='fgbuster_c1d1s1_BD10000_TD500_BS500'
KMEANS_FIXED='kmeans_c1d1s1_FIXED_active_set kmeans_c1d1s1_FIXED_adam_condFalse kmeans_c1d1s1_FIXED_scipy_tnc kmeans_c1d1s1_FIXED_adam_condTrue'

jid=$(sbatch $SBATCH_ARGS \
       --job-name=SNAP \
       $SLURM_SCRIPT RESULT/PLOT r_analysis snap -r $PTEP $kmeans_varying_patch $FGBUSTER $KMEANS_FIXED -ird RESULTS --no-tex -o SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)

sbatch  --dependency=afterany:$jid \
        $SBATCH_ARGS \
       --job-name=PLOT \
       $SLURM_SCRIPT RESULT/PLOT r_analysis plot -r $PTEP $kmeans_varying_patch $FGBUSTER $KMEANS_FIXED  -ird RESULTS -a --snapshot SNAPSHOT --no-tex

sbatch $SBATCH_ARGS \
       --job-name=VALIDATE \
       $SLURM_SCRIPT RESULT/PLOT r_analysis validate -r $PTEP $kmeans_varying_patch $FGBUSTER $KMEANS_FIXED -ird RESULTS --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5
