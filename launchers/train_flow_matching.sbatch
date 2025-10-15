#!/bin/bash

#SBATCH --job-name=sweep_flow_ohe_protocols
#SBATCH --output=../logs/train_flow_haematopoiesis.out
#SBATCH --error=../logs/train_flow_haematopoiesis.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=256G
#SBATCH --nice=0
#SBATCH --time=01-00:00:00

## set environment name
ENV_NAME="sc_exp_design";
echo "Using environment  $ENV_NAME";

## set working directory
WDIR=./scripts;
cd $WDIR;
echo "Current Working Directory $WDIR";

## hydra error logging
export HYDRA_FULL_ERROR=1;

## defining base representation obsm keys
CHANNEL_FEATS_OBS_KEYS="X_channel";
PCA_FEATS_OBS_KEYS="X_pca";
SCATTER_FEATS_OBS_KEYS="X_pca";

## defining concatenated representation keys
CONCAT_CHANNEL_FEATS_OBS_KEYS="X_channel_concat";
CONCAT_PCA_FEATS_OBS_KEYS="X_pca_concat";

## defining indentifier for condition modeling settings
OHE_PROTOCOLS_CONDITION_MODELING="ohe_protocols";
OHE_PROTOCOLS_AXES_CONDITION_MODELING="ohe_protocols_axes";
CONCAT_PROTOCOLS_AXES_CONDITION_MODELING="concat_protocols_axes";

##################### 0. HOT ENCODED PROTOCOLS
######### 0.0 channel features
echo "Haematopoiesis Train Flow Matching Model (Perturbation Response Prediction)";
echo "Condition Modeling: One Hot Encoded Protocols";
echo "State Modeling: Channel features";
conda run -n $ENV_NAME python train_flow_matching.py\
    data=ohe_protocols\
    vf=default_ohe_protocols\
    transforms.sample_rep=$CHANNEL_FEATS_OBS_KEYS\
    callbacks.wandb_project_name="haematopoiesis_forward_model_${OHE_PROTOCOLS_CONDITION_MODELING}_${CHANNEL_FEATS_OBS_KEYS}";

######### 0.1 pca features
echo "Haematopoiesis Train Flow Matching Model (Perturbation Response Prediction)";
echo "Condition Modeling: One Hot Encoded Protocols";
echo "State Modeling: Channel features";
conda run -n $ENV_NAME python train_flow_matching.py\
    data=ohe_protocols\
    vf=default_ohe_protocols\
    transforms.sample_rep=$PCA_FEATS_OBS_KEYS\
    callbacks.wandb_project_name="haematopoiesis_forward_model_${OHE_PROTOCOLS_CONDITION_MODELING}_${PCA_FEATS_OBS_KEYS}";

######### 0.2 scatter features
echo "Haematopoiesis Train Flow Matching Model (Perturbation Response Prediction)";
echo "Condition Modeling: One Hot Encoded Protocols";
echo "State Modeling: Channel features";
conda run -n $ENV_NAME python train_flow_matching.py\
    data=ohe_protocols\
    vf=default_ohe_protocols\
    transforms.sample_rep=$SCATTER_FEATS_OBS_KEYS_FEATS_OBS_KEYS\
    callbacks.wandb_project_name="haematopoiesis_forward_model_${OHE_PROTOCOLS_CONDITION_MODELING}_${SCATTER_FEATS_OBS_KEYS}";

######### 0.3 channel concat features
echo "Haematopoiesis Train Flow Matching Model (Perturbation Response Prediction)";
echo "Condition Modeling: One Hot Encoded Protocols";
echo "State Modeling: Channel features";
conda run -n $ENV_NAME python train_flow_matching.py\
    data=ohe_protocols\
    vf=default_ohe_protocols\
    transforms.sample_rep=$CONCAT_CHANNEL_FEATS_OBS_KEYS\
    callbacks.wandb_project_name="haematopoiesis_forward_model_${OHE_PROTOCOLS_CONDITION_MODELING}_${CONCAT_CHANNEL_FEATS_OBS_KEYS}";

######### 0.4 pca concat features
echo "Haematopoiesis Train Flow Matching Model (Perturbation Response Prediction)";
echo "Condition Modeling: One Hot Encoded Protocols";
echo "State Modeling: Channel features";
conda run -n $ENV_NAME python train_flow_matching.py\
    data=ohe_protocols\
    vf=default_ohe_protocols\
    transforms.sample_rep=$CONCAT_PCA_FEATS_OBS_KEYS\
    callbacks.wandb_project_name="haematopoiesis_forward_model_${OHE_PROTOCOLS_CONDITION_MODELING}_${CONCAT_PCA_FEATS_OBS_KEYS}";
