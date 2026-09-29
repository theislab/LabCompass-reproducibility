#!/bin/bash

#SBATCH --job-name=slim_checkpoints
#SBATCH --output=slim_checkpoints_%j.out
#SBATCH --error=slim_checkpoints_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

# Writes inference-only copies of one loop's checkpoints. Reads the originals
# and writes elsewhere; it never modifies or deletes them.
#
# Usage: sbatch slim_checkpoints.sbatch <loop-name>
#   loop-name: loop0 | loop1 | loop2_bloodplus | loop2p5_replicate
#              | loop3_replicate | loop4_replicate

# SLURM copies the batch script to a node-local spool dir, so BASH_SOURCE does
# not point into the repo. Recover the submitted path from the job record.
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

LOOP="${1:?Usage: sbatch slim_checkpoints.sbatch <loop-name>}"
ENV_NAME="${2:-sc_exp_design_12}"

cd "$REPO_ROOT" || exit 1

LOG_DIR="$REPO_ROOT/project_folder/output/logs/release"
mkdir -p "$LOG_DIR"

echo "------------------------------------------------"
echo "Loop        : ${LOOP}"
echo "Environment : ${ENV_NAME}"
echo "Repo root   : ${REPO_ROOT}"
echo "Node        : $(hostname)"
echo "------------------------------------------------"

conda run --no-capture-output -n "$ENV_NAME" python scripts/data/slim_checkpoints.py \
    --loops "$LOOP" \
    2>&1 | tee "$LOG_DIR/slim_checkpoints-${LOOP}.out"
