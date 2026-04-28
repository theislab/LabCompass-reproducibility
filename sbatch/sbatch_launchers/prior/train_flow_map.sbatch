#!/bin/bash

#SBATCH --job-name=distill_flow_map
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/conditional_model/job_submissions/distill_fm-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/forward/conditional_model/job_submissions/distill_fm-%j.err
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --mem=500G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
#SBATCH --constraint="h100_80gb"
 
# ── Environment ──────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate sc_exp_design_12
 
# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"
SCRIPT="$PROJECT_DIR/scripts/prior/train/distill_flow_map_prior.py"
 
# ── Run ──────────────────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "Start time: $(date)"
 
python "$SCRIPT" "$@"
 
echo "End time: $(date)"
 