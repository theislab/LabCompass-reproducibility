#!/bin/bash

#SBATCH --job-name=distill_flow_map
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/flow_map/distill_fm-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/flow_map/distill_fm-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
 
# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
 
# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC"
SCRIPT="$PROJECT_DIR/scripts/prior/train/distill_flow_map_prior.py"
export HYDRA_FULL_ERROR=1

# ── Run ──────────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
 
micromamba run -n jupyter python "$SCRIPT" "$@"
 
echo "End time: $(date)"
 