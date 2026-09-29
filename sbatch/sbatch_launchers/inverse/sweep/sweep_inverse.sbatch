#!/bin/bash

#SBATCH --job-name=sweep_loss_guidance
#SBATCH --output=sweep_loss_guidance_%j.out
#SBATCH --error=sweep_loss_guidance_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

# SLURM copies the batch script to a node-local spool dir, so BASH_SOURCE does
# not point into the repo. Recover the submitted path from the job record.
if [ -n "${SLURM_JOB_ID:-}" ]; then
    SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2; exit}' | awk '{print $1}')"
else
    SCRIPT_PATH="${BASH_SOURCE[0]}"
fi
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

if [ ! -d "$REPO_ROOT/scripts" ]; then
    echo "ERROR: REPO_ROOT resolved to '$REPO_ROOT', which is not the repository." >&2
    exit 1
fi


# define defaults
TARGET_CELL_TYPE="HSCs"
SCHEDULER="reciprocal"
OPT="unconstrained"
QUERY_TYPE="pure_populations"
ENV_NAME="sc_exp_design"
PATHS="default"
# ENV_MANAGER="micromamba"
ENV_MANAGER="conda"
N_SAMPLES=50
N_FWD_SAMPLES=350
CELL_TYPE_COLUMN=cell_type_leiden # "cell_type"
# name of the output subfolder under project_folder/output/inverse/loss_guidance/,
# now selectable via -d/--dump-name. Previously this was hardcoded as one of:
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/raw_data/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/new_runs_mgkpro/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/bloodplus_inverse/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/bloodplus_inverse_fm_best/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/bloodplus_inverse_fm_old_measurements_official/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/bloodplus_inverse_fm_loop2/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
# dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/bloodplus_inverse_fm_loop3/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
DUMP_NAME="bloodplus_inverse_fm_loop3"

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
    -n|--n-samples)
      N_SAMPLES="$2"
      shift 2
      ;;
    -nf|--n-fwd-samples)
      N_FWD_SAMPLES="$2"
      shift 2
      ;;
    -ctc|--ct-column)
      CELL_TYPE_COLUMN="$2"
      shift 2
      ;;
    -d|--dump-name)
      DUMP_NAME="$2"
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

if [[ "$OPT" == "unconstrained" ]]; then
    script_name="run_inverse_experiment"
else
    script_name="run_inverse_experiment_penalized"
    hparam_search="${hparam_search}_penalized"
fi

if [[ "$OPT" == "penalized_all_axes" ]]; then
    # constraints="default_all_cols"
    constraints="default_all_cols_loop3"
else
    constraints="default"
fi


# Map scheduler to configurations
case $QUERY_TYPE in
    "pure_populations") query_pure="True" mask_gradients="False";;
    "custom_populations") query_pure="False" mask_gradients="False";;
    "custom_populations_masked") query_pure="False" mask_gradients="True" ;;
    *) echo "❌ Unknown query type: $Q_TYPE"; exit 1 ;;
esac

# define path based on configuration (subfolder name selectable via -d/--dump-name)
dump_dir="$REPO_ROOT/project_folder/output/inverse/loss_guidance/${DUMP_NAME}/${OPT}-${QUERY_TYPE}-${SCHEDULER}"

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
echo "Optimization     : ${OPT} -> ${script_name}"
echo "Query Mode       : ${QUERY_TYPE} (pure=${query_pure}, mask=${mask_gradients})"
echo "Dump Name        : ${DUMP_NAME}"
echo "Dump Directory   : ${dump_dir}"
echo "Environment      : ${ENV_NAME}"
echo "Path             : ${PATHS}"
echo "Env Manager      : ${ENV_MANAGER}"
echo "Command          : ${CMD}"
echo "N                : ${N_SAMPLES}"
echo "N Forward        : ${N_FWD_SAMPLES}"
echo "ct column        : ${CELL_TYPE_COLUMN}"
echo "------------------------------------------------"

# handle hydra errors
export HYDRA_FULL_ERROR=1
    

$CMD python scripts/inverse/experiment/${script_name}.py \
    --multirun \
    +launcher=slurm \
    +hparam_search="${hparam_search}" \
    constraints="${constraints}" \
    loss_guidance.num_forward_pass_per_sample="${N_FWD_SAMPLES}" \
    paths=${PATHS} \
    loss.cell_type_column="${CELL_TYPE_COLUMN}" \
    paths.dump_dir=${dump_dir} \
    sampling.target_cell_type='"'"${TARGET_CELL_TYPE}"'"' \
    sampling.query_pure_cell_types="${query_pure}" \
    sampling.mask_gradients="${mask_gradients}" \
    sampling.N="${N_SAMPLES}" \
    annotation=bloodplus_loop3
