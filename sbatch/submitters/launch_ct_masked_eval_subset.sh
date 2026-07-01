#!/bin/bash
set -euo pipefail

# ----- Configuration -----
PYTHON_SCRIPT="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/scripts/inverse/plots/run_masked_eval_loop3.py"
COLS_TO_MASK='["il7_[ng_ml]", "mcsf_[ng_ml]", "ly_cocktail_[ul/well]"]'
NUM_TIME_STEPS=1000
SOLVER_KWARGS='{"method":"euler", "atol": 5e-5, "rtol": 5e-5}'
NUM_SAMPLES=10000
CELL_TYPE_COL="cell_type"

echo ${COLS_TO_MASK}
echo ${SOLVER_KWARGS}

# ---- Settings for SLURM ----
TIME="04:00:00"
CPUS_PER_TASK=6
MEM="500G"
PARTITION="gpu_p"
QOS="gpu_normal"
GRES="gpu:1"
CONSTRAINT="h100_80gb|a100_80gb"

# ---- Reading file name argument from CLI ----
if [ $# -ne 1 ]; then
    echo "Usage: $0 <runs_file.txt>"
    echo "  runs_file.txt : plain text file with one run directory per line"
    exit 1
fi


# ---- Absorb file name as variable and read file ---
RUNS_FILE="$1"
[ -f "$RUNS_FILE" ] || { echo "File $RUNS_FILE not found"; exit 1; }


# ---- Loop over runs which to run the evaluation for ----
while IFS= read -r run_dir_clean; do
    # --- Skip empty lines ----
    [ -z "$run_dir_clean" ] && continue

    # --- Skip when file is not found ----
    if [ ! -f "$run_dir_clean/candidates.csv" ]; then
        echo "Skipping $run_dir_clean: missing candidates.csv"
        continue
    fi

    # ---- Extract identifiers for a meaningful job name ----
    opt_id=$(basename "$(dirname "$(dirname "$run_dir_clean")")")
    ct_id=$(basename "$(dirname "$run_dir_clean")")
    run_id=$(basename "$run_dir_clean")

    # ---- Submit SBATCH job ----
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
#SBATCH --constraint=${CONSTRAINT}

micromamba run -n sc_exp_design python ${PYTHON_SCRIPT} \
    --run_dir "${run_dir_clean}" \
    --columns_to_mask "${COLS_TO_MASK}" \
    --num_time_steps ${NUM_TIME_STEPS} \
    --solver_kwargs "${SOLVER_KWARGS}" \
    --num_samples ${NUM_SAMPLES} \
    --cell_type_column "${CELL_TYPE_COL}"
EOF

    echo "Submitted job for $run_dir_clean"
done < "$RUNS_FILE"

echo "All jobs from $RUNS_FILE submitted."
