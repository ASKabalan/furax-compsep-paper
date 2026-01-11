#!/bin/bash
SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --qos=qos_gpu_h100-dev"


C1D1S1_NOISELESS_SCIPY="kmeans_c1d1s1_noise0_scipy fgbuster_c1d1s1_0"
C1D1S1_NOISY_SCIPY="kmeans_c1d1s1_noise100_scipy fgbuster_c1d1s1_100"
C1D0S0_NOISELESS_SCIPY="kmeans_c1d0s0_noise0_scipy fgbuster_c1d0s0_0"
C1D0S0_NOISY_SCIPY="kmeans_c1d0s0_noise100_scipy fgbuster_c1d0s0_100"

C1D1S1_NOISELESS_ADAM="kmeans_c1d1s1_noise0_adam_condFalse kmeans_c1d1s1_noise0_adam_condTrue"
C1D1S1_NOISY_ADAM="kmeans_c1d1s1_noise100_adam_condFalse kmeans_c1d1s1_noise100_adam_condTrue"
C1D0S0_NOISELESS_ADAM="kmeans_c1d0s0_noise0_adam_condFalse kmeans_c1d0s0_noise0_adam_condTrue"
C1D0S0_NOISY_ADAM="kmeans_c1d0s0_noise100_adam_condFalse kmeans_c1d0s0_noise100_adam_condTrue"

D1S1_NOISELESS_ADAM_TITLE=("FURAX c1d1s1 ADAM Noiseless" "FURAX c1d1s1 ADAM Noiseless Conditioned")
D1S1_NOISY_ADAM_TITLE=("FURAX c1d1s1 ADAM Noisy" "FURAX c1d1s1 ADAM Noisy Conditioned")
D0S0_NOISELESS_ADAM_TITLE=("FURAX c1d0s0 ADAM Noiseless" "FURAX c1d0s0 ADAM Noiseless Conditioned")
D0S0_NOISY_ADAM_TITLE=("FURAX c1d0s0 ADAM Noisy" "FURAX c1d0s0 ADAM Noisy Conditioned")

D1S1_NOISELESS_SCIPY_TITLE=("FURAX c1d1s1 Scipy TNC Noiseless" "FGBuster c1d1s1 Noiseless")
D1S1_NOISY_SCIPY_TITLE=("FURAX c1d1s1 Scipy TNC Noisy" "FGBuster c1d1s1 Noisy")
D0S0_NOISELESS_SCIPY_TITLE=("FURAX c1d0s0 Scipy TNC Noiseless" "FGBuster c1d0s0 Noiseless")
D0S0_NOISY_SCIPY_TITLE=("FURAX c1d0s0 Scipy TNC Noisy" "FGBuster c1d0s0 Noisy")

jid=$(sbatch $SBATCH_ARGS \
       --job-name=SNAP \
       $SLURM_SCRIPT r_analysis snap -r $C1D1S1_NOISY_ADAM -ird RESULTS --no-tex -o SNAPSHOT)

sbatch  --dependency=afterany:$jid \
        $SBATCH_ARGS \
       --job-name=PLOT \
       $SLURM_SCRIPT r_analysis plot -r $C1D1S1_NOISY_ADAM -t "${D1S1_NOISY_ADAM_TITLE[@]}" -ird RESULTS -a --snapshot SNAPSHOT --no-tex
