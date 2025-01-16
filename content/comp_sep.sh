#!/bin/bash
sbatch --account=tkc@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --job-name=NLL-N-h100 comp_sep.slurm