#!/bin/bash

#SBATCH --job-name=build_solutions
#SBATCH --output=build_solutions_%j.out
#SBATCH --error=build_solutions_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nice=0
#SBATCH --time=12:00:00

# Consolidates one loop's sweep output into a single gzipped CSV.
# Reads the dumps only; never modifies them.
# Usage: sbatch build_solutions.sbatch <loop-name>

if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}' | awk '{print $1}')"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ ! -d "$REPO_ROOT/scripts" ]; then
    echo "ERROR: REPO_ROOT resolved to '$REPO_ROOT', which is not the repository." >&2
    exit 1
fi

LOOP="${1:?Usage: sbatch build_solutions.sbatch <loop-name>}"
ENV_NAME="${2:-labcompass}"

cd "$REPO_ROOT" || exit 1
echo "Building solutions table for ${LOOP} on $(hostname)"

conda run --no-capture-output -n "$ENV_NAME" python scripts/data/build_solutions_tables.py --loop "$LOOP"
