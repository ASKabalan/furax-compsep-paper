#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"
OUTPUT_DIR="RESULTS/MINIMIZE/TOPK"
mkdir -p $OUTPUT_DIR

job_ids=()
masks=("GAL020" "GAL040" "GAL060")
# k=None (default ~0.1), then 0.1, 0.2, 0.4, 0.6, 0.8
# We'll use "default" to denote the run without -top_k arg
ks=("0.1" "0.2" "0.4" "0.6" "0.8")

td_sp=20.0
tag="c1d1s1"

for mask in "${masks[@]}"; do
    mp=${mask#GAL} # e.g. 020

    for k in "${ks[@]}"; do
    
        top_k_arg="-top_k $k"
        k_name="$k"

        job_name="FX_${mp}_100_AS_${k_name}"
        
        # sbatch output: "Submitted batch job 12345"
        output=$(sbatch $SBATCH_ARGS --job-name="$job_name" \
             $SLURM_SCRIPT $OUTPUT_DIR kmeans-model -n 64 -ns 1 -nr 1.0 \
             -pc 10000 500 500 -tag $tag -m $mask -i LiteBIRD \
             -sp 1.54 $td_sp -3.0 -mi 2000 -o $OUTPUT_DIR \
             -s active_set $top_k_arg)
             
        jid=$(echo "$output" | awk '{print $4}')
        if [ -n "$jid" ]; then
            job_ids+=("$jid")
            echo "Submitted $job_name: $jid"
        else
            echo "Failed to submit $job_name"
        fi
    done
done

# Final Analysis
deps=$(IFS=:; echo "${job_ids[*]}")
if [ -z "$deps" ]; then
    echo "No jobs submitted."
    exit 1
fi

echo "Submitting analysis job depending on: $deps"

sbatch --dependency=afterany:$deps \
       $SBATCH_ARGS \
       --job-name=FX_validate_top_k \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis validate \
       -r "kmeans_topk(\d+)" \
       -ird $OUTPUT_DIR \
       --noise-ratio 1.0 \
       --no-tex \
       --scales 1e-4 1e-5
       

snap_id=$(sbatch --dependency=afterany:$deps \
       $SBATCH_ARGS \
       --job-name=FX_SNAP_top_k \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap \
       -r "kmeans_topk(\d+)" \
       -ird $OUTPUT_DIR \
       --no-tex \
       -s scipy_tnc \
       -o $OUTPUT_DIR/SNAPSHOT \
       -mi 2000 \
       -n 64 \
       -i LiteBIRD)

sbatch --dependency=afterany:$snap_id \
       $SBATCH_ARGS \
       --job-name=FX_PLOT_top_k \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis plot \
       -r "kmeans_topk(\d+)" \
       -ird $OUTPUT_DIR \
       -a \
       --snapshot $OUTPUT_DIR/SNAPSHOT \
       --no-tex 
       

