#!/bin/bash
#SBATCH --job-name=sweep_marker_opt
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/marker_opt/job_submissions/sweep_loss_guidance_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/marker_opt/job_submissions/sweep_loss_guidance_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

# ----------------------------------------------------------------------
# 1. Force correct working directory (where the Python script lives)
# ----------------------------------------------------------------------
PROJECT_ROOT="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"
cd "$PROJECT_ROOT" || { echo "❌ Cannot cd to $PROJECT_ROOT"; exit 1; }

# ----------------------------------------------------------------------
# 2. Ensure output directories exist (Slurm does not create parents)
# ----------------------------------------------------------------------
LOG_DIR="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/marker_opt/job_submissions"
mkdir -p "$LOG_DIR"

# ----------------------------------------------------------------------
# 3. Default values
# ----------------------------------------------------------------------
TARGET_CELL_TYPE="late_MgkPro"
TARGET_MARKER="CD41a"
SCHEDULER="reciprocal"
OPT="unconstrained"
ENV_NAME="sc_exp_design"
PATHS="bloodplus_fm"
ENV_MANAGER="micromamba"
N_OPT_SAMPLES=10

# ----------------------------------------------------------------------
# 4. Parse command line arguments
# ----------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--target-marker)     TARGET_MARKER="$2"; shift 2 ;;
    -tct|--target-ct)       TARGET_CELL_TYPE="$2"; shift 2 ;;
    -s|--scheduler)         SCHEDULER="$2"; shift 2 ;;
    -o|--optimization-type) OPT="$2"; shift 2 ;;
    -e|--env-name)          ENV_NAME="$2"; shift 2 ;;
    -p|--paths)             PATHS="$2"; shift 2 ;;
    -em|--env-manager)      ENV_MANAGER="$2"; shift 2 ;;
    -n|--n-opt-samples)     N_OPT_SAMPLES="$2"; shift 2 ;;
    *) echo "❌ Unknown parameter: $1"; exit 1 ;;
  esac
done

# ----------------------------------------------------------------------
# 5. Map scheduler to Hydra sweep config
# ----------------------------------------------------------------------
case $SCHEDULER in
    "constant")    hparam_search="sweep_constant" ;;
    "exp-decay")   hparam_search="sweep_exp_decay" ;;
    "reciprocal")  hparam_search="sweep_reciprocal" ;;
    "lin-decay")   hparam_search="sweep_lin_decay" ;;
    *) echo "❌ Unknown scheduler: $SCHEDULER"; exit 1 ;;
esac

if [[ "$OPT" != "unconstrained" ]]; then
    hparam_search="${hparam_search}_penalized"
fi

if [[ "$OPT" == "penalized_all_axes" ]]; then
    constraints="default_all_cols"
else
    constraints="default"
fi

# ----------------------------------------------------------------------
# 6. Define dump directory and create it
# ----------------------------------------------------------------------
dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/bloodplus_inverse_marker_opt/${OPT}-${SCHEDULER}"
mkdir -p "$dump_dir"

# ----------------------------------------------------------------------
# 7. Prepare environment command
# ----------------------------------------------------------------------
if [[ "${ENV_MANAGER}" == "conda" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME} --no-capture-output"
elif [[ "${ENV_MANAGER}" == "micromamba" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME}"
else
  echo "❌ Unknown env manager: $ENV_MANAGER"
  exit 1
fi

# ----------------------------------------------------------------------
# 8. Log what we're doing (this will appear in the .out file)
# ----------------------------------------------------------------------
echo "------------------------------------------------"
echo "Project root       : $PROJECT_ROOT"
echo "Target cell type   : $TARGET_CELL_TYPE"
echo "Target marker      : $TARGET_MARKER"
echo "Scheduler          : $SCHEDULER -> $hparam_search"
echo "Optimization       : $OPT"
echo "Constraints        : $constraints"
echo "Num Samples        : $N_OPT_SAMPLES"
echo "Dump directory     : $dump_dir"
echo "Path config        : $PATHS"
echo "Environment        : $ENV_NAME ($ENV_MANAGER)"
echo "------------------------------------------------"

# ----------------------------------------------------------------------
# 9. Run Hydra
# ----------------------------------------------------------------------
export HYDRA_FULL_ERROR=1

$CMD python scripts/inverse/experiment/run_inverse_experiment_marker_opt.py \
    --multirun \
    +launcher=slurm \
    +hparam_search="${hparam_search}" \
    +marker_opt=sweep_marker_opt \
    annotation=bloodplus \
    loss.target_marker_names="[${TARGET_MARKER}]" \
    constraints="${constraints}" \
    paths="${PATHS}" \
    paths.dump_dir="${dump_dir}" \
    sampling.N=${N_OPT_SAMPLES}
