#!/bin/bash
#SBATCH --job-name=inverse_plot1
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions1/%x_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions1/%x_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal

EXPERIMENT_DIR=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/raw_data
WDIR=/home/icb/alessandro.palma/environment/collab-goettgens-SFC

cd "$WDIR" || exit 1

source ~/.bashrc
conda activate sc_exp_design_12

EXPERIMENTS=(
    # "penalized_all_axes-custom_populations-constant"
    # "penalized_all_axes-custom_populations-reciprocal"
    # "penalized_all_axes-custom_populations_masked-constant"
    # "penalized_all_axes-custom_populations_masked-reciprocal"
    "penalized_all_axes-pure_populations-constant"
    "penalized_all_axes-pure_populations-reciprocal"
    # "penalized_oxy_days-custom_populations-constant"
    # "penalized_oxy_days-custom_populations-reciprocal"
    # "penalized_oxy_days-custom_populations_masked-constant"
    # "penalized_oxy_days-custom_populations_masked-reciprocal"
    "penalized_oxy_days-pure_populations-constant"
    "penalized_oxy_days-pure_populations-reciprocal"
    # "unconstrained-custom_populations-constant"
    # "unconstrained-custom_populations-reciprocal"
    # "unconstrained-custom_populations_masked-constant"
    # "unconstrained-custom_populations_masked-reciprocal"
    "unconstrained-pure_populations-constant"
    "unconstrained-pure_populations-reciprocal"
)


# CELL_TYPES=(
#     "CD14-Mono"
#     "CyclingProgenitor*"
#     "DCs-CD14"
#     "Early_Pro-B"
#     "EosBasoMastPre"
#     "EryPro"
#     "GMP-Neutro:CD16-Mono"
#     "HSCs"
#     "MgkPro"
#     "Pro-B"
# )

for experiment in "${EXPERIMENTS[@]}"; do
    echo "Processing $experiment"
    python scripts/inverse/plots/uncertainty_estimation.py \
        --target_cell_types "CD14-Mono__CyclingProgenitor*__DCs-CD14__Early_Pro-B__EosBasoMastPre__EryPro__GMP-Neutro/CD16-Mono__HSCs__MgkPro__Pro-B" \
        --n_noise_samples 2000 \
        --n_populations 20 \
        --base_config_path ../../../inverse/loss_guidance/config/ \
        --base_config_name run_inverse \
        --experiment_type "$experiment" \
        --true_concentration_path /lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/miscellaneous/unique_concentrations_adata.h5ad
done
