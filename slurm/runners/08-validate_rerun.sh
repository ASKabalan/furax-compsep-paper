SBATCH_ARGS="--account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100"
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --qos=qos_gpu_h100-dev"

SCALES="1e-5"

RUNS_NOISY="fgbuster_100_c1d1s1 kmeans_noise100_c1d1s1 fgbuster_100_c1d0s0 kmeans_noise100_c1d0s0"
NAMES_NOISY='"FGBuster_Noisy_c1d1s1" "FURAX_Noisy_c1d1s1" "FGBuster_Noisy_c1d0s0" "FURAX_Noisy_c1d0s0"'

RUNS_NOISELESS="fgbuster_0_c1d1s1 kmeans_noise0_c1d1s1 fgbuster_0_c1d0s0 kmeans_noise0_c1d0s0"
NAMES_NOISELESS='"FGBuster Noiseless c1d1s1" "FURAX_Noiseless_c1d1s1" "FGBuster Noiseless c1d0s0" "FURAX_Noiseless_c1d0s0"'

sbatch $SBATCH_ARGS \
       --job-name=VALIDATE_NOISY \
       $SLURM_SCRIPT r_analysis validate -r $RUNS_NOISY -t $NAMES_NOISY  \
       -ird RESULTS  --noise-ratio 1.0 --no-tex --scales $SCALES

sbatch $SBATCH_ARGS \
       --job-name=VALIDATE_NOISELESS \
       $SLURM_SCRIPT r_analysis validate -r $RUNS_NOISELESS -t $NAMES_NOISELESS  \
       -ird RESULTS  --noise-ratio 0.0 --no-tex --scales $SCALES
