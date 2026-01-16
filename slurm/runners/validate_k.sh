#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100 --parsable"
OUTPUT_DIR="RESULTS/MINIMIZE/BENCHMARK"
mkdir -p $OUTPUT_DIR

job_ids=()
tag="c1d1s1"
mask="ALL-GALACTIC"

# Common arguments for all runs
COMMON_ARGS="-n 64 -ns 1 -nr 1.0 -pc 10000 500 500 -tag $tag -m $mask -i LiteBIRD -sp 1.54 20.0 -3.0 -mi 2000 -o $OUTPUT_DIR"
max_syst_iter=2000
# =============================================================================
# 1. Active Set Variants (3x2x6 = 36 runs)
# =============================================================================
as_solvers=("active_set_sgd" "active_set_adaw" "active_set_adabelief")
linesearches=("zoom" "backtracking")
ks=("0.0" "0.1" "0.2" "0.4" "0.6" "0.8")

for solver in "${as_solvers[@]}"; do
    for ls in "${linesearches[@]}"; do
        for k in "${ks[@]}"; do
            
            # Keywords: kmeans, topk, active_set, solver, linesearch
            run_name="kmeans_topk${k}_${solver}_ls${ls}"
            job_name="AS_${solver:11:3}_${ls:0:1}_${k}" 
            
            echo "Submitting $run_name"
            jid=$(sbatch $SBATCH_ARGS --job-name="$job_name" \
                 $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
                 -s $solver -ls $ls -top_k $k --name $run_name)
                 
            job_ids+=("$jid")
        done
    done
done

# =============================================================================
# 2. Optax L-BFGS Variants (1x2x1 = 2 runs)
# =============================================================================
lbfgs_linesearches=("zoom" "backtracking")

for ls in "${lbfgs_linesearches[@]}"; do
    # Only conditioned runs
    flag="-cond"
    cname="cond"
    
    run_name="kmeans_lbfgs_ls${ls}_${cname}"
    job_name="LB_${ls:0:1}_${cname}"
    
    echo "Submitting $run_name"
    jid=$(sbatch $SBATCH_ARGS --job-name="$job_name" \
            $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
            -s optax_lbfgs -ls $ls $flag --name $run_name)
            
    job_ids+=("$jid")
done

# =============================================================================
# 3. SciPy TNC (1 run)
# =============================================================================
run_name="kmeans_scipy_tnc"
job_name="TNC"

echo "Submitting $run_name"
jid=$(sbatch $SBATCH_ARGS --job-name="$job_name" \
     $SLURM_SCRIPT $OUTPUT_DIR kmeans-model $COMMON_ARGS \
     -s scipy_tnc --name $run_name)
     
job_ids+=("$jid")

# =============================================================================
# 4. Analysis
# =============================================================================
deps=$(IFS=:; echo "${job_ids[*]}")
if [ -z "$deps" ]; then
    echo "No jobs submitted."
    exit 1
fi

echo "Submitting analysis jobs depending on: $deps"

# --- Set 1: Active Set (Split by Solver) ---

# Loop over solvers to create separate analysis jobs for each
for s in sgd adaw adabelief; do
    SOLVER_PATS=""
    for ls in zoom backtracking; do
        # Pattern: matches folder tokens e.g. "topk0.1" -> capture "0.1"
        # Folder tokens: kmeans, topk0.1, active, set, sgd, lszoom
        pat="kmeans_topk(\d+\.\d+)_active_set_${s}_ls${ls}"
        SOLVER_PATS="$SOLVER_PATS $pat"
    done
    
    # Short name for job labels
    s_short=${s:0:3}
    if [ "$s" == "adabelief" ]; then s_short="ada"; fi

    # Validate
    sbatch --dependency=afterany:$deps \
           $SBATCH_ARGS \
           --job-name=FX_val_${s_short} \
           $SLURM_SCRIPT $OUTPUT_DIR r_analysis validate \
           -r $SOLVER_PATS \
           -ird $OUTPUT_DIR \
           --noise-ratio 1.0 \
           --no-tex \
           --scales 1e-4 1e-5

    # Snapshot
    snap_id=$(sbatch --dependency=afterany:$deps \
           $SBATCH_ARGS \
           --job-name=FX_snap_${s_short} \
           $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap \
           -r $SOLVER_PATS \
           -ird $OUTPUT_DIR \
           --no-tex \
           -s scipy_tnc \
           -o $OUTPUT_DIR/SNAPSHOT_AS_${s} \
           -mi $max_syst_iter \
           -n 64 \
           -i LiteBIRD)

    # Plot
    sbatch --dependency=afterany:$snap_id \
            $SBATCH_ARGS \
            --job-name=FX_plt_${s_short} \
            $SLURM_SCRIPT $OUTPUT_DIR r_analysis plot \
            -r $SOLVER_PATS \
            -ird $OUTPUT_DIR \
            -a \
            --snapshot $OUTPUT_DIR/SNAPSHOT_AS_${s} \
            --no-tex
done


# --- Set 2: Others (L-BFGS + TNC) ---
LB_PATS="kmeans_lbfgs_lszoom_cond kmeans_lbfgs_lsbacktracking_cond"
TNC_PAT="kmeans_scipy_tnc"
OTH_PATS="$LB_PATS $TNC_PAT"

# Validate Others
sbatch --dependency=afterany:$deps \
       $SBATCH_ARGS \
       --job-name=FX_val_OTH \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis validate \
       -r $OTH_PATS \
       -ird $OUTPUT_DIR \
       --noise-ratio 1.0 \
       --no-tex \
       --scales 1e-4 1e-5

# Snapshot Others
snap_oth_id=$(sbatch --dependency=afterany:$deps \
       $SBATCH_ARGS \
       --job-name=FX_snap_OTH \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis snap \
       -r $OTH_PATS \
       -ird $OUTPUT_DIR \
       --no-tex \
       -s scipy_tnc \
       -o $OUTPUT_DIR/SNAPSHOT_OTH \
       -mi $max_syst_iter \
       -n 64 \
       -i LiteBIRD)

# Plot Others
sbatch --dependency=afterany:$snap_oth_id \
       $SBATCH_ARGS \
       --job-name=FX_plt_OTH \
       $SLURM_SCRIPT $OUTPUT_DIR r_analysis plot \
       -r $OTH_PATS \
       -ird $OUTPUT_DIR \
       -a \
       --snapshot $OUTPUT_DIR/SNAPSHOT_OTH \
       --no-tex