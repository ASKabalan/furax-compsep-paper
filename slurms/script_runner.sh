#!/bin/bash

# Corrected Bash loop over BS values
for BS in 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500 6000 6500 7000 7500 8000 8500 9000 9500 10000
do
    echo "Running KMeans with BS = $BS"
    sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=KMEANS_$BS$-a100 99-slurm_runner.slurm kmeans-model -n 64 -ns 100 -nr 1.0 -pc "$BS" 500 500 -tag c1d1s1 -m GAL020 -i LiteBIRD
    # Zone 2 mask of GAL040 - GAL020
    sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=KMEANS_$BS$-a100 99-slurm_runner.slurm kmeans-model -n 64 -ns 100 -nr 1.0 -pc "$BS" 500 500 -tag c1d1s1 -m GAL040 -i LiteBIRD
    # Zone 3 mask of GAL060 - GAL040
    sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100 --job-name=KMEANS_$BS$-a100 99-slurm_runner.slurm kmeans-model -n 64 -ns 100 -nr 1.0 -pc "$BS" 500 500 -tag c1d1s1 -m GAL060 -i LiteBIRD
done
