#!/bin/bash
SBATCH_ARGS="--account=nih@h100 --nodes=1 --gres=gpu:1 --tasks-per-node=1 -C h100"

# =============================================================================
# KMEANS SETS (Regex)
# =============================================================================

# Set 1: Varying BS
DIR="RESULTS/KMEANS/BD10000_TD500_BSXXX"
REGEX="kmeans_c1d1s1_BD10000_TD500_BS(\d+)"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM1 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM1 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_KM1 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

# Set 2: Varying TD
DIR="RESULTS/KMEANS/BD10000_TDXXX_BS500"
REGEX="kmeans_c1d1s1_BD10000_TD(\d+)_BS500"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM2 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM2 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_KM2 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

# Set 3: Varying BS Small
DIR="RESULTS/KMEANS/BD4000_TD10_BSXX"
REGEX="kmeans_c1d1s1_BD4000_TD10_BS(\d+)"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM3 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM3 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_KM3 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

# Set 4: Varying TD Small
DIR="RESULTS/KMEANS/BD4000_TDXX_BS10"
REGEX="kmeans_c1d1s1_BD4000_TD(\d+)_BS10"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM4 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM4 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_KM4 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

# Set 5: Varying BD
DIR="RESULTS/KMEANS/BDXXX_TD500_BS500"
REGEX="kmeans_c1d1s1_BD(\d+)_TD500_BS500"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_KM5 $SLURM_SCRIPT ANALYSIS r_analysis snap -r "$REGEX" -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_KM5 $SLURM_SCRIPT ANALYSIS r_analysis plot -r "$REGEX" -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_KM5 $SLURM_SCRIPT ANALYSIS r_analysis validate -r "$REGEX" -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10


# =============================================================================
# MINIMIZE (No Regex)
# =============================================================================
FGBUSTER='fgbuster_c1d1s1_BD10000_TD500_BS500 kmeans_c1d1s1_FIXED_scipy_tnc'
KMEANS_ADAM='kmeans_c1d1s1_FIXED_active_set kmeans_c1d1s1_FIXED_adam_condFalse  kmeans_c1d1s1_FIXED_adam_condTrue'

# Noiseless
DIR="RESULTS/MINIMIZE/NOISELESS"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_MIN0 $SLURM_SCRIPT ANALYSIS r_analysis snap -r $FGBUSTER -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_MIN0 $SLURM_SCRIPT ANALYSIS r_analysis plot -r $FGBUSTER -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_MIN0 $SLURM_SCRIPT ANALYSIS r_analysis validate -r $FGBUSTER -ird $DIR --noise-ratio 0.0 --no-tex --scales 1e-4 1e-5 --steps 10

jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_MIN0_KM $SLURM_SCRIPT ANALYSIS r_analysis snap -r $KMEANS_ADAM -ird $DIR --no-tex -o $DIR/SNAPSHOT_KM -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_MIN0_KM $SLURM_SCRIPT ANALYSIS r_analysis plot -r $KMEANS_ADAM -ird $DIR -a --snapshot $DIR/SNAPSHOT_KM --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_MIN0_KM $SLURM_SCRIPT ANALYSIS r_analysis validate -r $KMEANS_ADAM -ird $DIR --noise-ratio 0.0 --no-tex --scales 1e-4 1e-5 --steps 10


# Noisy
DIR="RESULTS/MINIMIZE/NOISY"
jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_MIN100 $SLURM_SCRIPT ANALYSIS r_analysis snap -r $FGBUSTER -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_MIN100 $SLURM_SCRIPT ANALYSIS r_analysis plot -r $FGBUSTER -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_MIN100 $SLURM_SCRIPT ANALYSIS r_analysis validate -r $FGBUSTER -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

jid=$(sbatch $SBATCH_ARGS --job-name=SNAP_MIN100_KM $SLURM_SCRIPT ANALYSIS r_analysis snap -r $KMEANS_ADAM -ird $DIR --no-tex -o $DIR/SNAPSHOT_KM -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid $SBATCH_ARGS --job-name=PLOT_MIN100_KM $SLURM_SCRIPT ANALYSIS r_analysis plot -r $KMEANS_ADAM -ird $DIR -a --snapshot $DIR/SNAPSHOT_KM --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_MIN100_KM $SLURM_SCRIPT ANALYSIS r_analysis validate -r $KMEANS_ADAM -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10



# =============================================================================
# MULTIRES (PTEP) - Chained
# =============================================================================
DIR="RESULTS/MULTIRES"

# Set 1: ptep1
jid1=$(sbatch $SBATCH_ARGS --job-name=SNAP_PTEP1 $SLURM_SCRIPT ANALYSIS r_analysis snap -r ptep1 -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=PLOT_PTEP1 $SLURM_SCRIPT ANALYSIS r_analysis plot -r ptep1 -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_PTEP1 $SLURM_SCRIPT ANALYSIS r_analysis validate -r ptep1 -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

# Set 2: ptep2 (Depends on jid1 to share/update snapshot)
jid2=$(sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=SNAP_PTEP2 $SLURM_SCRIPT ANALYSIS r_analysis snap -r ptep2 -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid2 $SBATCH_ARGS --job-name=PLOT_PTEP2 $SLURM_SCRIPT ANALYSIS r_analysis plot -r ptep2 -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_PTEP2 $SLURM_SCRIPT ANALYSIS r_analysis validate -r ptep2 -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10


# =============================================================================
# TENSOR_TO_SCALER - Chained
# =============================================================================
DIR="RESULTS/TESNOR_TO_SCALER_34"

# Set 1: cr3d1s1
jid1=$(sbatch $SBATCH_ARGS --job-name=SNAP_CR3 $SLURM_SCRIPT ANALYSIS r_analysis snap -r kmeans_cr3d1s1 -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=PLOT_CR3 $SLURM_SCRIPT ANALYSIS r_analysis plot -r kmeans_cr3d1s1 -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_CR3 $SLURM_SCRIPT ANALYSIS r_analysis validate -r kmeans_cr3d1s1 -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10

# Set 2: cr4d1s1 (Depends on jid1)
jid2=$(sbatch --dependency=afterany:$jid1 $SBATCH_ARGS --job-name=SNAP_CR4 $SLURM_SCRIPT ANALYSIS r_analysis snap -r kmeans_cr4d1s1 -ird $DIR --no-tex -o $DIR/SNAPSHOT -mi 2000 -s active_set -n 64 -i LiteBIRD)
sbatch --dependency=afterany:$jid2 $SBATCH_ARGS --job-name=PLOT_CR4 $SLURM_SCRIPT ANALYSIS r_analysis plot -r kmeans_cr4d1s1 -ird $DIR -a --snapshot $DIR/SNAPSHOT --no-tex
sbatch $SBATCH_ARGS --job-name=VAL_CR4 $SLURM_SCRIPT ANALYSIS r_analysis validate -r kmeans_cr4d1s1 -ird $DIR --noise-ratio 1.0 --no-tex --scales 1e-4 1e-5 --steps 10
