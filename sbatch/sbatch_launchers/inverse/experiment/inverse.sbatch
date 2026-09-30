#!/bin/bash

#SBATCH --job-name=run_inverse
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/inverse/loss_guidance/job_submissions/inverse_run_%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/inverse/loss_guidance/job_submissions/inverse_run_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=300G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=h100_80gb|a100_80gb


source ~/.bashrc
ENV_NAME="jupyter"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC
cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

micromamba run -n "$ENV_NAME" python ./scripts/inverse/experiment/run_inverse_experiment_morphogens.py