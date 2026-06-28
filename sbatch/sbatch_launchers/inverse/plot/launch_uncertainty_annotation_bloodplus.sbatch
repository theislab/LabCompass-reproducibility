#!/bin/bash
#SBATCH --job-name=inverse_plot
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions_lollo/%x_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions_lollo/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"

EXPERIMENT=$1

# EXPERIMENT_DIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/bloodplus_inverse_fm_official/
# EXPERIMENT_DIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/bloodplus_inverse_fm_old_measurements_official/
# EXPERIMENT_DIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/bloodplus_inverse_fm_loop2
EXPERIMENT_DIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/bloodplus_inverse_fm_loop3

WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC

cd "$WDIR" || exit 1

source ~/.bashrc
# conda activate sc_exp_design_12
micromamba activate sc_exp_design

echo "Running experiment: $EXPERIMENT"
echo "Experiment dir: $EXPERIMENT_DIR"

python scripts/inverse/plots/uncertainty_estimation.py \
    --target_cell_types "CD14Mono__CD16Mono__ProB__late_MgkPro__preProB" \
    --n_noise_samples 5000 \
    --n_populations 10 \
    --base_config_path ../../../inverse/loss_guidance/config/ \
    --base_config_name run_inverse \
    --result_dir "$EXPERIMENT_DIR" \
    --experiment_type "$EXPERIMENT" \
    --true_concentration_path  /lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/loop3/unique_concentrations.h5ad \
    --paths bloodplus_loop3_old_clf \
    --cell_type_column cell_type_leiden \
    --annotation bloodplus_loop3 \
    --constraints default_all_cols_loop3 \


# --target_cell_types "EryPro__HSCs__late_MgkPro__preProB__ProB" \
# --target_cell_types "CD14Mono__CD16Mono__CyclingPro1__CyclingPro2__EosBasMast1__EosBasMast2__EosBasMast3__EryPro__HSCs__ProB__late_MgkPro__GMP-Neutro__preProB__pre_cDC1" \