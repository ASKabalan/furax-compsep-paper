#!/bin/bash
##############################################################################################################################
# USAGE:sbatch --account=nih@a100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C a100  
##############################################################################################################################
#SBATCH --job-name=Likelihoods   # nom du job
#SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 du noeud 8-GPU)
#SBATCH --hint=nomultithread         # hyperthreading desactive
#SBATCH --time=05:00:00              # temps d'execution maximum demande (HH:MM:SS)
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH --output=%x_%N.out # nom du fichier de sortie
#SBATCH --error=%x_%N.err  # nom du fichier d'erreur (ici commun avec la sortie)
##SBATCH --exclusive                  # ressources dediees
# Nettoyage des modules charges en interactif et herites par defaut
num_nodes=$SLURM_JOB_NUM_NODES
num_gpu_per_node=$SLURM_NTASKS_PER_NODE
OUTPUT_FOLDER_ARGS=1
# Calculate the number of GPUs
nb_gpus=$(( num_nodes * num_gpu_per_node))

module purge

echo "Job partition: $SLURM_JOB_PARTITION"
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition

if [[ "$SLURM_JOB_PARTITION" == "gpu_p5" ]]; then
    module load arch/a100
    source /gpfsdswork/projects/rech/tkc/commun/venv/wass/bin/activate
    gpu_name=a100
elif [[ "$SLURM_JOB_PARTITION" == "gpu_p6" ]]; then
    module arch/h100
    source /gpfsdswork/projects/rech/tkc/commun/venv/h100/bin/activate
    gpu_name=h100
else
    source /gpfsdswork/projects/rech/tkc/commun/venv/v100/bin/activate
    gpu_name=v100
fi

echo "The number of nodes allocated for this job is: $num_nodes"
echo "The number of GPUs allocated for this job is: $nb_gpus"

export EQX_ON_ERROR=nan
export OUTPUT_FOLDER_ARGS=1

function plaunch() {
    if [ $# -lt 1 ]; then
        echo "Usage: profile_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="prof_traces/$gpu_name/$nb_gpus/$script_name"
    local report_dir="out_prof/$gpu_name/$nb_gpus/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="prof_traces/$gpu_name/$nb_gpus/$script_name/$args"
        report_dir="out_prof/$gpu_name/$nb_gpus/$script_name/$args"
    fi

    mkdir -p "$output_dir"
    mkdir -p "$report_dir"

    srun nsys profile -t cuda,nvtx,osrt,mpi -o "$report_dir/report_rank%q{SLURM_PROCID}" python "$@" > "$output_dir/$script_name.err" 2> "$output_dir/$script_name.out" || true
}

function slaunch() {
    if [ $# -lt 1 ]; then
        echo "Usage: run_python <python_script> [arguments for the script]"
        return 1
    fi

    local script_name=$(basename "$1" .py)
    local output_dir="traces/$gpu_name/$nb_gpus/$script_name"

    if [ $OUTPUT_FOLDER_ARGS -eq 1 ]; then
        local args=$(echo "${@:2}" | tr ' ' '_')
        # Remove characters '/' and '-' from folder name
        args=$(echo "$args" | tr -d '/-')
        output_dir="traces/$gpu_name/$nb_gpus/$script_name/$args"
    fi

    mkdir -p "$output_dir"

    srun python "$@" > "$output_dir/$script_name.out" 2> "$output_dir/$script_name.err" || true
}


# Echo des commandes lancees
set -x

slaunch "$@"

