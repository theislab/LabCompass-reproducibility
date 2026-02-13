#!/bin/bash

#SBATCH --job-name=sweep_loss_guidance
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions/sweep_loss_guidance_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions/sweep_loss_guidance_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

# define defaults
TARGET_CELL_TYPE="HSCs"
SCHEDULER="reciprocal"
OPT="unconstrained"
QUERY_TYPE="pure_populations"
ENV_NAME="sc_exp_design"

# Parse Flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--target-ct)
      TARGET_CELL_TYPE="$2"
      shift 2
      ;;
    -s|--scheduler)
      SCHEDULER="$2"
      shift 2
      ;;
    -o|--optimization-type)
      OPT="$2"
      shift 2
      ;;
    -q|--query-type)
      QUERY_TYPE="$2"
      shift 2
      ;;
    -e|--env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# Map scheduler to configurations
case $SCHEDULER in
    "constant")    hparam_search="sweep_constant" ;;
    "exp-decay")   hparam_search="sweep_exp_decay" ;;
    "reciprocal")  hparam_search="sweep_reciprocal" ;;
    "lin-decay")   hparam_search="sweep_lin_decay" ;;
    *) echo "❌ Unknown scheduler: $SCHEDULER"; exit 1 ;;
esac

# select script
[[ "$OPT" == "unconstrained" ]] && script_name="run_inverse_experiment" || script_name="run_inverse_experiment_penalized"
[[ "$OPT" == "penalized_all_axes" ]] && constraints="default_all_cols" || constraints="default"


# Map scheduler to configurations
case $QUERY_TYPE in
    "pure_populations") query_pure="True" mask_gradients="False";;
    "custom_populations") query_pure="False" mask_gradients="False";;
    "custom_populations_masked") query_pure="False" mask_gradients="True" ;;
    *) echo "❌ Unknown query type: $Q_TYPE"; exit 1 ;;
esac

# define path based on configuration
dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/raw_data/${OPT}-${QUERY_TYPE}-${SCHEDULER}"

# Pring logging messages
echo "------------------------------------------------"
echo "Target Cell Type : ${TARGET_CELL_TYPE}"
echo "Scheduler        : ${SCHEDULER} -> ${hparam_search}"
echo "Optimization     : ${OPT} -> ${script_name}"
echo "Query Mode       : ${QUERY_TYPE} (pure=${query_pure}, mask=${mask_gradients})"
echo "Dump Directory   : ${dump_dir}"
echo "Environment      : ${ENV_NAME}"
echo "------------------------------------------------"

# handle hydra errors
export HYDRA_FULL_ERROR=1

# run python script
conda run -n ${ENV_NAME} --no-capture-output \
    python scripts/inverse/experiment/${script_name}.py \
    --multirun \
    +launcher=slurm \
    +hparam_search="${hparam_search}" \
    constraints="${constraints}" \
    paths.dump_dir=${dump_dir} \
    sampling.target_cell_type='"'"${TARGET_CELL_TYPE}"'"' \
    sampling.query_pure_cell_types="${query_pure}" \
    sampling.mask_gradients="${mask_gradients}"
