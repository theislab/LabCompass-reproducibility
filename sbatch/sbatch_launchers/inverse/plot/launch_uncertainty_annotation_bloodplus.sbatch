#!/bin/bash
#SBATCH --job-name=inverse_plot
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint="a100_80gb|h100_80gb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

EXPERIMENT=$1

EXPERIMENT_DIR=$REPO_ROOT/project_folder/output/inverse/loss_guidance/loop3_replicate

# WDIR=$REPO_ROOT
WDIR=/home/icb/alessandro.palma/environment/collab-goettgens-SFC

cd "$WDIR" || exit 1

source ~/.bashrc
conda activate sc_exp_design_12

echo "Running experiment: $EXPERIMENT"
echo "Experiment dir: $EXPERIMENT_DIR"

# python scripts/inverse/uncertainty/uncertainty_estimation.py \
#     --target_cell_types "EryPro__late_MgkPro" \
#     --n_noise_samples 5000 \
#     --n_populations 10 \
#     --base_config_path ../../../inverse/loss_guidance/config/ \
#     --base_config_name run_inverse \
#     --result_dir "$EXPERIMENT_DIR" \
#     --experiment_type "$EXPERIMENT" \
#     --true_concentration_path  $REPO_ROOT/project_folder/data/gdrive/loop3/unique_concentrations.h5ad \
#     --paths loop2p5_replicate \
#     --cell_type_column cell_type_reannot_final \
#     --annotation bloodplus \
#     --constraints default_all_cols \

python scripts/inverse/uncertainty/uncertainty_estimation.py \
    --target_cell_types "late_MgkPro__EryPro" \
    --n_noise_samples 5000 \
    --n_populations 10 \
    --base_config_path ../../../inverse/loss_guidance/config/ \
    --base_config_name run_inverse \
    --result_dir "$EXPERIMENT_DIR" \
    --experiment_type "$EXPERIMENT" \
    --true_concentration_path  $REPO_ROOT/project_folder/data/gdrive/loop3/unique_concentrations.h5ad \
    --paths loop3_replicate \
    --cell_type_column cell_type \
    --annotation bloodplus_loop3 \
    --constraints default_all_cols_loop3 \
