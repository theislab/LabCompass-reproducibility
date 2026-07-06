#!/bin/bash
#SBATCH --job-name=inverse_plot
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions1/%x_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions1/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"

EXPERIMENT=$1

EXPERIMENT_DIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/loop4_replicate

# WDIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC
WDIR=/home/icb/alessandro.palma/environment/collab-goettgens-SFC

cd "$WDIR" || exit 1

source ~/.bashrc
conda activate sc_exp_design_12

echo "Running experiment: $EXPERIMENT"
echo "Experiment dir: $EXPERIMENT_DIR"

# python scripts/inverse/plots/uncertainty_estimation.py \
#     --target_cell_types "EryPro__late_MgkPro" \
#     --n_noise_samples 5000 \
#     --n_populations 10 \
#     --base_config_path ../../../inverse/loss_guidance/config/ \
#     --base_config_name run_inverse \
#     --result_dir "$EXPERIMENT_DIR" \
#     --experiment_type "$EXPERIMENT" \
#     --true_concentration_path  /lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/loop3/unique_concentrations.h5ad \
#     --paths loop2p5_replicate \
#     --cell_type_column cell_type_reannot_final \
#     --annotation bloodplus \
#     --constraints default_all_cols \

python scripts/inverse/plots/uncertainty_estimation.py \
    --target_cell_types "EryPro__late_MgkPro" \
    --n_noise_samples 5000 \
    --n_populations 10 \
    --base_config_path ../../../inverse/loss_guidance/config/ \
    --base_config_name run_inverse \
    --result_dir "$EXPERIMENT_DIR" \
    --experiment_type "$EXPERIMENT" \
    --true_concentration_path  /lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/data/gdrive/loop3/unique_concentrations.h5ad \
    --paths loop4_replicate \
    --cell_type_column cell_type \
    --annotation bloodplus_loop3 \
    --constraints default_all_cols_loop3 \
