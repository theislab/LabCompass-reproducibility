#!/bin/bash

#SBATCH --job-name=train_flow_matching_prior
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/prior/train_flow_matching_prior-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/prior/train_flow_matching_prior-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
 
# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc

ENV_NAME="jupyter"
echo "Using environment: $ENV_NAME"

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC"
SCRIPT="$PROJECT_DIR/scripts/prior/train/train_flow_matching_prior.py"
 
# ── Run ──────────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
 
micromamba run -n "$ENV_NAME" python "$SCRIPT" "$@"
 
echo "End time: $(date)"
 