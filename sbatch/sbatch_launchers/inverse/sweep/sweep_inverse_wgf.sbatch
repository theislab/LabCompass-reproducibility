#!/bin/bash

#SBATCH --job-name=sweep_wgf
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/wgf/job_submissions/sweep_loss_guidance_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/wgf/job_submissions/sweep_loss_guidance_%j.err
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
PATHS="default"
ENV_MANAGER="micromamba"

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
    -p|--paths)
      PATHS="$2"
      shift 2
      ;;
    -em|--env-manager)
      ENV_MANAGER="$2"
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
    "cnv")   hparam_search="sweep_cnv" ;;
    "cos")   hparam_search="sweep_cos" ;;
    "cvx")   hparam_search="sweep_cvx" ;;
    "exp")   hparam_search="sweep_exp" ;;
    "lin")   hparam_search="sweep_lin" ;;
    *) echo "❌ Unknown scheduler: $SCHEDULER"; exit 1 ;;
esac


# Map scheduler to configurations
case $QUERY_TYPE in
    "pure_populations") query_pure="True" mask_gradients="False";;
    "custom_populations") query_pure="False" mask_gradients="False";;
    "custom_populations_masked") query_pure="False" mask_gradients="True" ;;
    *) echo "❌ Unknown query type: $Q_TYPE"; exit 1 ;;
esac

# define path based on configuration
# dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/raw_data/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/new_runs_mgkpro/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/bloodplus_inverse/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/wgf_guidance/bloodplus_inverse_wgf/${OPT}-${QUERY_TYPE}-${SCHEDULER}"

# prepare command
if [[ "${ENV_MANAGER}" == "conda" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME} --no-capture-output"
elif [[ "${ENV_MANAGER}" == "micromamba" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME}"
else
  echo "❌ Unknown env manager: $ENV_MANAGER"
  exit 1
fi

# Pring logging messages
echo "------------------------------------------------"
echo "Target Cell Type : ${TARGET_CELL_TYPE}"
echo "Scheduler        : ${SCHEDULER} -> ${hparam_search}"
echo "Query Mode       : ${QUERY_TYPE} (pure=${query_pure}, mask=${mask_gradients})"
echo "Dump Directory   : ${dump_dir}"
echo "Environment      : ${ENV_NAME}"
echo "Path             : ${PATHS}"
echo "Env Manager      : ${ENV_MANAGER}"
echo "Command          : ${CMD}"
echo "------------------------------------------------"

# handle hydra errors
export HYDRA_FULL_ERROR=1
    
$CMD python scripts/inverse/experiment/run_inverse_experiment_wgf.py \
    --multirun \
    +launcher=slurm \
    +hparam_search="${hparam_search}" \
    paths=${PATHS} \
    paths.dump_dir=${dump_dir} \
    sampling.target_cell_type='"'"${TARGET_CELL_TYPE}"'"' \
    sampling.query_pure_cell_types="${query_pure}" \
    sampling.mask_gradients="${mask_gradients}"
