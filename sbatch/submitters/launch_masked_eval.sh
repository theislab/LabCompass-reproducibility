#!/bin/bash
set -euo pipefail

# ----- Configuration -----
PYTHON_SCRIPT="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/scripts/inverse/plots/run_masked_eval_loop3.py"

# Python script arguments
COLS_TO_MASK='["il7_[ng_ml]", "mcsf_[ng_ml]", "ly_cocktail_[ul/well]"]'
NUM_TIME_STEPS=1000
SOLVER_KWARGS='{"method":"euler", "atol": 5e-5, "rtol": 5e-5}'
NUM_SAMPLES=20000
CELL_TYPE_COL="cell_type"

RES_BASE_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/loop3_replicate"
CELL_TYPES=("CD14Mono" "CD16Mono")

# SLURM settings
TIME="04:00:00"
CPUS_PER_TASK=6
MEM="500G"
PARTITION="gpu_p"
QOS="gpu_normal"
GRES="gpu:1"

# ----- Iterate over folder structure -----
for opt_dir in "$RES_BASE_DIR"/*/; do
    opt_id=$(basename "$opt_dir")
    for ct_dir in "$opt_dir"/*/; do
        ct_id=$(basename "$ct_dir")

        # Skip cell types not in the allowed list
        if [[ ! " ${CELL_TYPES[*]} " =~ " ${ct_id} " ]]; then
            echo "Skipping cell type $ct_id (not in allowed list)"
            continue
        fi

        for run_dir in "$ct_dir"/*/; do
            run_id=$(basename "$run_dir")
            run_dir_clean="${run_dir%/}"

            if [ ! -f "$run_dir_clean/candidates.csv" ]; then
                echo "Skipping $run_dir_clean: missing candidates.csv"
                continue
            fi

            # Submit directly via a heredoc – no temporary file
            sbatch <<- EOF
			#!/bin/bash
			#SBATCH --job-name=${opt_id}_${ct_id}_${run_id}
			#SBATCH --output=${run_dir_clean}/masked_eval_%j.out
			#SBATCH --error=${run_dir_clean}/masked_eval_%j.err
			#SBATCH --time=${TIME}
			#SBATCH --cpus-per-task=${CPUS_PER_TASK}
			#SBATCH --mem=${MEM}
			#SBATCH --partition=${PARTITION}
			#SBATCH --qos=${QOS}
			#SBATCH --gres=${GRES}

			micromamba run -n sc_exp_design python ${PYTHON_SCRIPT} \
			    --run_dir "${run_dir_clean}" \
			    --columns_to_mask "${COLS_TO_MASK}" \
			    --num_time_steps ${NUM_TIME_STEPS} \
			    --solver_kwargs "${SOLVER_KWARGS}" \
			    --num_samples ${NUM_SAMPLES} \
			    --cell_type_column "${CELL_TYPE_COL}"
			EOF

            echo "Submitted job for $run_dir_clean"
        done
    done
done

echo "All jobs have been submitted."
