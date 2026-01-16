#!/bin/bash
##############################################################################################################################
# USAGE: sbatch myscript.sh <RUN_NAME> python <script.py> [args...]
# EXAMPLE: sbatch myscript.sh my_experiment_v1 python train.py --lr 0.01
##############################################################################################################################
#SBATCH --job-name=Likelihoods
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1 
#SBATCH --tasks-per-node=1  
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --output=%x_%N.out
#SBATCH --error=%x_%N.err
#SBATCH --parsable

# Nettoyage des modules charges en interactif et herites par defaut
num_nodes=$SLURM_JOB_NUM_NODES
num_gpu_per_node=$SLURM_NTASKS_PER_NODE
nb_gpus=$(( num_nodes * num_gpu_per_node ))

module purge

echo "Job partition: $SLURM_JOB_PARTITION"

if [[ "$SLURM_JOB_PARTITION" == "gpu_p5" ]]; then
    module load arch/a100
    source /gpfsdswork/projects/rech/tkc/commun/venv/a100/bin/activate
    gpu_name=a100
elif [[ "$SLURM_JOB_PARTITION" == "gpu_p6" ]]; then
    module load arch/h100 
    source /gpfsdswork/projects/rech/tkc/commun/venv/h100/bin/activate
    gpu_name=h100
else
    source /gpfsdswork/projects/rech/tkc/commun/venv/v100/bin/activate
    gpu_name=v100
fi

echo "The number of nodes allocated for this job is: $num_nodes"
echo "The number of GPUs allocated for this job is: $nb_gpus"

export SEND_EMAIL=1
export JAX_COMPILATION_CACHE_DIR=$WORK/jax_cache
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0

# Helper function to get the actual script name (e.g., 'main' from 'python main.py')
function get_script_name() {
    if [[ "$1" == "python" || "$1" == "python3" ]]; then
        basename "$2" .py
    else
        basename "$1" .py
    fi
}

# Helper to turn arguments into a filename-safe string
# Truncates long strings and appends a hash for uniqueness/safety
function get_args_slug() {
    local full_slug=$(echo "$*" | tr -s ' ./-' '_')
    
    # Check if slug is longer than 100 characters
    if [ ${#full_slug} -le 100 ]; then
        echo "$full_slug"
    else
        # Take the first 50 chars for readability + 8 chars of MD5 hash for uniqueness
        local hash=$(echo "$*" | md5sum | cut -c1-8)
        echo "${full_slug:0:50}_${hash}"
    fi
}

function plaunch() {
    if [ $# -lt 2 ]; then
        echo "Usage: plaunch <RUN_NAME> <command> [arguments...]"
        return 1
    fi

    # 1. Extract the Run Name (first arg)
    local run_name=$1
    shift 

    # 2. Identify script name and create a slug from arguments
    local script_name=$(get_script_name "$@")
    local args_slug=$(get_args_slug "$@")
    
    # 3. Define directories (RUN_NAME is the folder)
    local output_dir="$run_name/prof_traces"
    local report_dir="$run_name/out_prof"

    mkdir -p "$output_dir"
    mkdir -p "$report_dir"

    local out_file="$output_dir/${args_slug}.out"
    local err_file="$output_dir/${args_slug}.err"
    local report_file="$report_dir/report_${args_slug}_rank%q{SLURM_PROCID}"
    
    # Log full command for traceability since filename might be truncated
    echo "Full Command: $@" > "$out_file"
    echo "----------------------------------------" >> "$out_file"

    # 4. Run NSYS
    srun nsys profile -t cuda,nvtx,osrt,mpi -o "$report_file" "$@" >> "$out_file" 2>> "$err_file" || true
}

function slaunch() {
    if [ $# -lt 2 ]; then
        echo "Usage: slaunch <RUN_NAME> <command> [arguments...]"
        return 1
    fi

    # 1. Extract the Run Name (first arg)
    local run_name=$1
    shift 

    # 2. Identify script name and create a slug from arguments
    local script_name=$(get_script_name "$@")
    local args_slug=$(get_args_slug "$@")

    # 3. Define directories (RUN_NAME is the folder)
    local output_dir="$run_name/traces"
    mkdir -p "$output_dir"

    local out_file="$output_dir/${args_slug}.out"
    local err_file="$output_dir/${args_slug}.err"
    
    # Log full command for traceability since filename might be truncated
    echo "Full Command: $@" > "$out_file"
    echo "----------------------------------------" >> "$out_file"

    # 4. Run the job
    srun "$@" >> "$out_file" 2>> "$err_file"
    local rc=$?

    # ---- conditional email ----
    if [ "$SEND_EMAIL" -eq 1 ] && [ -n "$EMAIL" ]; then
        {
            echo "SLURM job $SLURM_JOB_ID finished."
            echo "Job name: $SLURM_JOB_NAME"
            echo "Run label: $run_name"
            echo "Args: $@"
            echo "Exit code: $rc"
            echo "Output directory: $output_dir"
            echo
            echo "Attaching:"
            echo "  $out_file"
            echo "  $err_file"
            echo

            uuencode "$out_file" "${args_slug}.out"
            uuencode "$err_file" "${args_slug}.err"
        } | mail -s "[$SLURM_JOB_NAME] $run_name (rc=$rc)" "$EMAIL"
    fi
}

# Echo des commandes lancees
set -x

slaunch "$@"
