#!/bin/bash

#SBATCH --job-name=sweep_classifier
#SBATCH --output=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/cell_type_classifier/sweep/sweep_classifier-%j.log
#SBATCH --error=/lustre/groups/ml01/workspace/ilia.navosha/projects/ood/output/logs/forward/cell_type_classifier/sweep/sweep_classifier-%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00
source ~/.bashrc

SAMPLE_REP="X_train_pca"


echo "Running experiments on sample: $SAMPLE_REP"

ENV_NAME="jupyter"
echo "Using environment: $ENV_NAME"

WDIR=/home/icb/ilia.navosha/expDesign/collab-goettgens-SFC

cd "$WDIR" || exit 1
echo "Current Working Directory: $WDIR"

export HYDRA_FULL_ERROR=1

WANDB="sweep_big_RA_4"
echo "Wandb project ${WANDB}"

micromamba run -n "$ENV_NAME" python ./scripts/forward/train/train_target_prediction_model_RA_4.py \
    --multirun \
    +launcher=slurm \
    +hparams_search=default \
    #callbacks.wandb_project_name="$wandb_name" \

