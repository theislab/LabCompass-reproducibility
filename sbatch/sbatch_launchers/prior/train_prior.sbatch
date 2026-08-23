#!/bin/bash

#SBATCH --job-name=train_flow_matching_prior
#SBATCH --output=train_flow_matching_prior-%j.log
#SBATCH --error=train_flow_matching_prior-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
#SBATCH --constraint="h100_80gb"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
 
# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate sc_exp_design_12
 
# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="$REPO_ROOT"
SCRIPT="$PROJECT_DIR/scripts/prior/train/train_flow_matching_prior.py"
 
# ── Run ──────────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
 
python "$SCRIPT" "$@" annotation="loop3"
 
echo "End time: $(date)"
 