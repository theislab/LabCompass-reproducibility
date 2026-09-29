#!/bin/bash

#SBATCH --job-name=verify_slim
#SBATCH --output=verify_slim_%j.out
#SBATCH --error=verify_slim_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

# Compares each slimmed checkpoint against its original. Read-only.
# Usage: sbatch verify_slim_checkpoints.sbatch <loop-name>

if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}' | awk '{print $1}')"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ ! -d "$REPO_ROOT/scripts/data" ]; then
    echo "ERROR: REPO_ROOT resolved to '$REPO_ROOT', which is not the repository." >&2
    exit 1
fi

LOOP="${1:?Usage: sbatch verify_slim_checkpoints.sbatch <loop-name>}"
ENV_NAME="${2:-sc_exp_design_12}"

cd "$REPO_ROOT" || exit 1
echo "Verifying ${LOOP} on $(hostname)"

conda run --no-capture-output -n "$ENV_NAME" python scripts/data/verify_slim_checkpoints.py --loop "$LOOP"
