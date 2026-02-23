#!/bin/bash

BASE_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/raw_data"
EXP_TYPE="unconstrained-custom_populations_masked-reciprocal"
PYTHON_SCRIPT="./scripts/inverse/sensitivity_analysis/run_sensitivity_analysis.py"
LOG_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions"

mkdir -p $LOG_DIR

# 1. Loop through each Cell Type folder (e.g., CyclingProgenitor:)
for CT_PATH in "$BASE_DIR/$EXP_TYPE"/*/; do
    [ -e "$CT_PATH" ] || continue # Handle empty glob
    
    CT_NAME_DIR=$(basename "$CT_PATH")
    # Convert colon back to slash for the Python argument if needed
    CT_ARG=$(echo "$CT_NAME_DIR" | sed 's/:/\//g')

    # 2. Loop through each Run folder inside the Cell Type
    for RUN_PATH in "$CT_PATH"*/; do
        [ -e "$RUN_PATH" ] || continue
        
        RUN_ID=$(basename "$RUN_PATH")

        # Skip the output folder if it already exists to avoid recursion
        if [[ "$RUN_ID" == "sensitivity_analysis" ]]; then
            continue
        fi

        # Check if results already exist (Optional: avoids re-running)
        if [ -f "${RUN_PATH}sensitivity_analysis/sensitivity_response_data.pkl" ]; then
            echo "Skipping $CT_NAME_DIR/$RUN_ID - already processed."
            continue
        fi

        echo "Submitting: CT=$CT_ARG | Run=$RUN_ID"

        # 3. Submit the job
        sbatch --job-name="sens_${RUN_ID}" \
               --output="$LOG_DIR/%j_${RUN_ID}.out" \
               --error="$LOG_DIR/%j_${RUN_ID}.err" \
               --export=ALL,TARGET_CT="$CT_ARG",RUN_ID="$RUN_ID",EXP_TYPE="$EXP_TYPE",BASE_DIR="$BASE_DIR",PY_SCRIPT="$PYTHON_SCRIPT" \
               ./sbatch/sbatch_launchers/inverse/sensitivity_analysis/submit_sensitivity_analysis.sbatch
    done
done
