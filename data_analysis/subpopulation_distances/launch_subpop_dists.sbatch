#!/bin/bash

#SBATCH --job-name=compute_subpops_distances
#SBATCH --output=../../../logs/compute_dists_heamatopoiesis.out
#SBATCH --error=../../../logs/compute_dists_heamatopoiesis.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_priority
#SBATCH --mem=50G
#SBATCH --nice=0
#SBATCH --time=0-02:00:00


SAMPLE_REP=${1:-"X_channel_standardized"}
DISTANCE_FN=${2:-"e_distance"}
OBS_COLUMN=${3:-"protocol_id"}

echo "Computing ${DISTANCE_FN} on ${SAMPLE_REP}, grouped by ${OBS_COLUMN}"

ENV_NAME="sc_exp_design"
echo "Using environment: $ENV_NAME"
conda run --no-capture-output -n "$ENV_NAME" python ./compute_subpopulation_distances.py \
    distance.state_repr=${SAMPLE_REP} \
    distance.distance_fn=${DISTANCE_FN} \
    distance.groups_obs_key=${OBS_COLUMN} \
    2>&1 | tee "../../../logs/compute_subpop_dists-${SAMPLE_REP}-${DISTANCE_FN}-${OBS_COLUMN}.log" 
