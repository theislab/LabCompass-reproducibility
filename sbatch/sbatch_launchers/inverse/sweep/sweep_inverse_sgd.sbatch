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


# ---- 1. Define defaults ----
TARGET_CELL_TYPE="late_MgkPro"
SCHEDULER="reciprocal"
ENV_NAME="sc_exp_design"
PATHS="loop1"
ENV_MANAGER="micromamba"
N_SAMPLES=50 
N_FWD_SAMPLES=350
CELL_TYPE_COLUMN="cell_type" 
SGD_INIT_FROM_DATA=0

# ---- 2. Parse Flags ----
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
    -sid|--sgd-init-from-data)
      SGD_INIT_FROM_DATA="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done

# ---- 3. Map scheduler to configurations ----
case $SCHEDULER in
    "constant")    hparam_search="constant" ;;
    "exp-decay")   hparam_search="exponential" ;;
    "lin-decay")   hparam_search="linear" ;;
    *) echo "❌ Unknown scheduler: $SCHEDULER"; exit 1 ;;
esac

# ---- 
case $SGD_INIT_FROM_DATA in
    0)    sgd_init_from_data="False" ;;
    1)    sgd_init_from_data="True" ;;
    *) echo "❌ Invalid flag for optional data initialization: $SGD_INIT_FROM_DATA"; exit 1 ;;
esac

# ---- 4. Define path based on configuration ----
dump_dir="$REPO_ROOT/project_folder/output/inverse/sgd_gaussian_init/loop1/${SCHEDULER}"

# ---- 5. Prepare command ----
script_name="run_inverse_experiment"
if [[ "${ENV_MANAGER}" == "conda" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME} --no-capture-output"
elif [[ "${ENV_MANAGER}" == "micromamba" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME}"
else
  echo "❌ Unknown env manager: $ENV_MANAGER"
  exit 1
fi

# ---- 6. Additional settings ----
annotation="bloodplus"
constraints="default_all_cols"
forward_model="default"
launcher="slurm"
loss="sgd"
loss_guidance="default"
non_linearity="default"
run="default"
sampling="default_pure_pop"


# ---- 7. Print logging messages ----
echo "------------------------------------------------"
echo "Target Cell Type : ${TARGET_CELL_TYPE}"
echo "Scheduler        : ${SCHEDULER} -> ${hparam_search}"
echo "Dump Directory   : ${dump_dir}"
echo "Environment      : ${ENV_NAME}"
echo "Path             : ${PATHS}"
echo "Env Manager      : ${ENV_MANAGER}"
echo "Command          : ${CMD}"
echo "N                : ${N_SAMPLES}"
echo "N Forward        : ${N_FWD_SAMPLES}"
echo "ct column        : ${CELL_TYPE_COLUMN}"
echo "------------------------------------------------"

# ---- 8. Handle hydra errors ----
export HYDRA_FULL_ERROR=1
  
# ---- 9. Run command ----
script_name="run_sgd_baseline"
$CMD python scripts/inverse/experiment/${script_name}.py \
    --multirun \
    +launcher=${launcher} \
    +sgd_grids="${hparam_search}" \
    +sgd_init_from_data="${init_from_data}" \
    annotation="${annotation}" \
    constraints="${constraints}" \
    forward_model="${forward_model}" \
    loss=sgd \
    loss.cell_type_column="${CELL_TYPE_COLUMN}" \
    loss_guidance.num_forward_pass_per_sample="${N_FWD_SAMPLES}" \
    non_linearity="${non_linearity}" \
    run="${run}" \
    sampling="${sampling}" \
    paths=${PATHS} \
    paths.dump_dir=${dump_dir} \
    sampling.target_cell_type='"'"${TARGET_CELL_TYPE}"'"' \
    sampling.query_pure_cell_types=True \
    sampling.mask_gradients=False \
    sampling.N="${N_SAMPLES}" \
