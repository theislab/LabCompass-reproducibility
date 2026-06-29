#!/bin/bash
set -euo pipefail

# ----- Configuration -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/masked_eval.py"

# Default Python script arguments
COLS_TO_MASK='["il7_[ng_ml]", "mcsf_[ng_ml]", "ly_cocktail_[ul/well]"]'
NUM_TIME_STEPS=1000
SOLVER_KWARGS='{"method":"euler", "atol": 5e-5, "rtol": 5e-5}'
NUM_SAMPLES=20000
CELL_TYPE_COL="cell_type_reannot_final"

# Default SLURM settings (can be overridden per job if you wish)
TIME="02:00:00"
CPUS_PER_TASK=4
MEM="16G"

if [ $# -ne 1 ]; then
    echo "Usage: $0 <res_base_dir>"
    exit 1
fi

RES_BASE_DIR="$1"
[ -d "$RES_BASE_DIR" ] || { echo "Directory $RES_BASE_DIR does not exist"; exit 1; }

# ----- Iterate over folder structure -----
for opt_dir in "$RES_BASE_DIR"/*/; do
    opt_id=$(basename "$opt_dir")
    for ct_dir in "$opt_dir"/*/; do
        ct_id=$(basename "$ct_dir")
        for run_dir in "$ct_dir"/*/; do
            run_id=$(basename "$run_dir")
            run_dir_clean="${run_dir%/}"

            if [ ! -f "$run_dir_clean/candidates.csv" ]; then
                echo "Skipping $run_dir_clean: missing candidates.csv"
                continue
            fi

            # ----- Build a temporary SBATCH file -----
            tmp_sbatch=$(mktemp "${run_dir_clean}/.submit_XXXXXX.sh")
            cat > "$tmp_sbatch" <<- 'EOF_INNER'
#!/bin/bash
#SBATCH --job-name=JOBNAME_PLACEHOLDER
#SBATCH --output=OUTPUT_PLACEHOLDER
#SBATCH --error=ERROR_PLACEHOLDER
#SBATCH --time=TIME_PLACEHOLDER
#SBATCH --cpus-per-task=CPUS_PLACEHOLDER
#SBATCH --mem=MEM_PLACEHOLDER

python PYSCRIPT_PLACEHOLDER \
    --run_dir RUNDIR_PLACEHOLDER \
    --columns_to_mask COLS_PLACEHOLDER \
    --num_time_steps NTIMESTEPS_PLACEHOLDER \
    --solver_kwargs SOLVERKWARGS_PLACEHOLDER \
    --num_samples NUMSAMPLES_PLACEHOLDER \
    --cell_type_column CELLTYPECOL_PLACEHOLDER
EOF_INNER

            # Replace placeholders with actual values
            sed -i \
                -e "s|JOBNAME_PLACEHOLDER|${opt_id}_${ct_id}_${run_id}|" \
                -e "s|OUTPUT_PLACEHOLDER|${run_dir_clean}/masked_eval_%j.out|" \
                -e "s|ERROR_PLACEHOLDER|${run_dir_clean}/masked_eval_%j.err|" \
                -e "s|TIME_PLACEHOLDER|${TIME}|" \
                -e "s|CPUS_PLACEHOLDER|${CPUS_PER_TASK}|" \
                -e "s|MEM_PLACEHOLDER|${MEM}|" \
                -e "s|PYSCRIPT_PLACEHOLDER|${PYTHON_SCRIPT}|" \
                -e "s|RUNDIR_PLACEHOLDER|${run_dir_clean}|" \
                -e "s|COLS_PLACEHOLDER|${COLS_TO_MASK}|" \
                -e "s|NTIMESTEPS_PLACEHOLDER|${NUM_TIME_STEPS}|" \
                -e "s|SOLVERKWARGS_PLACEHOLDER|${SOLVER_KWARGS}|" \
                -e "s|NUMSAMPLES_PLACEHOLDER|${NUM_SAMPLES}|" \
                -e "s|CELLTYPECOL_PLACEHOLDER|${CELL_TYPE_COL}|" \
                "$tmp_sbatch"

            # Make it executable (optional)
            chmod +x "$tmp_sbatch"

            # Submit the temporary script
            sbatch "$tmp_sbatch"
            echo "Submitted job for $run_dir_clean (script: $tmp_sbatch)"
        done
    done
done

echo "All jobs have been submitted."
