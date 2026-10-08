#!/bin/bash

#SBATCH --job-name=train_flow_matching_prior
#SBATCH --output=train_flow_maps-%j.log
#SBATCH --error=train_flow_maps-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
#SBATCH --constraint="h100_80gb"

# SLURM copies the batch script to a node-local spool dir, so BASH_SOURCE does
# not point into the repo. Recover the submitted path from the job record.
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

 
# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate labcompass

export HYDRA_FULL_ERROR=1
 
# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$REPO_ROOT"
SCRIPT="$PROJECT_DIR/scripts/prior/train/distill_flow_map_prior.py"
 
# ── Run ──────────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
 
python "$SCRIPT" "$@"
 
echo "End time: $(date)"
 