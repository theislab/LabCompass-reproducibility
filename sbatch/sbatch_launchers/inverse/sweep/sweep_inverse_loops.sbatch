#!/bin/bash

#SBATCH --job-name=sweep_loss_guidance
#SBATCH --output=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions_latest/sweep_loss_guidance_%j.out
#SBATCH --error=/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/logs/inverse/loss_guidance/job_submissions_latest/sweep_loss_guidance_%j.err
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --mem=20G
#SBATCH --nice=0
#SBATCH --time=1-00:00:00

# ---- 1. Define cli overridable defaults ----
TARGET_CELL_TYPE="HSCs"
SCHEDULER="reciprocal"
OPT="unconstrained"
QUERY_TYPE="pure_populations"
ENV_NAME="sc_exp_design"
ENV_MANAGER="micromamba"
N_SAMPLES=50 
N_FWD_SAMPLES=350
LOOP="2.5"
OLD_CLF="false"


# ---- 2. Define constants set automatically by the script ----
loss="default"
paths="default"
annotation="bloodplus"
base_dump_dir="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/"
query_pure="True"
mask_gradients="False"


# ---- 3. Override defaults ----
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
    -em|--env-manager)
      ENV_MANAGER="$2"
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
    -l|--loop)
      LOOP="$2"
      shift 2
      ;;
    -ocl|--old-clf)
      OLD_CLF="true"
      shift
      ;;
    *)
      echo "Unknown parameter passed: $1"
      exit 1
      ;;
  esac
done


# ---- 4. Handle scheduler ----
case $SCHEDULER in
    "constant")    hparam_search="sweep_constant" ;;
    "exp-decay")   hparam_search="sweep_exp_decay" ;;
    "reciprocal")  hparam_search="sweep_reciprocal" ;;
    "lin-decay")   hparam_search="sweep_lin_decay" ;;
    *) echo "❌ Unknown scheduler: $SCHEDULER"; exit 1 ;;
esac


# ---- 5. Handle constraints ----
if [[ "$OPT" == "unconstrained" ]]; then
    script_name="run_inverse_experiment"
else
    script_name="run_inverse_experiment_penalized"
    hparam_search="${hparam_search}_penalized"
fi
if [[ "$OPT" == "penalized_all_axes" ]]; then
    constraints="default_all_cols"
else
    constraints="default"
fi


# ---- 6. Handle query type ----
case $QUERY_TYPE in
    "pure_populations") query_pure="True" mask_gradients="False";;
    "custom_populations") query_pure="False" mask_gradients="False";;
    "custom_populations_masked") query_pure="False" mask_gradients="True" ;;
    *) echo "❌ Unknown query type: $QUERY_TYPE"; exit 1 ;;
esac


# ---- 7. Handle loops ----
case $LOOP in
  1)
    paths="loop1"
    dump_dir="${base_dump_dir}/loop1/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
    ;;
  2)
    paths="loop2_replicate"
    loss="loop2_replicate"
    dump_dir="${base_dump_dir}/loop2_replicate/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
    if [[ "${OLD_CLF}" == "true" ]]; then
        paths="${paths}_old_clf"
        dump_dir="${base_dump_dir}/loop2_replicate_old_clf/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
        loss=f"${loss}_old_clf"
    fi
    ;;
  2.5)
    paths="loop2p5_replicate"
    loss="loop2p5_replicate"
    dump_dir="${base_dump_dir}/loop2p5_replicate/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
    if [[ "${OLD_CLF}" == "true" ]]; then
        paths="${paths}_old_clf"
        dump_dir="${base_dump_dir}/loop2p5_replicate_old_clf/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
        loss=f"${loss}_old_clf"
    fi
    ;;
  3)
    paths="loop3_replicate"
    loss="loop3_replicate"
    annotation="bloodplus_loop3"
    dump_dir="${base_dump_dir}/loop3_replicate/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
    if [[ "$OPT" == "penalized_all_axes" ]]; then
        constraints="default_all_cols_loop3"
    fi
    if [[ "${OLD_CLF}" == "true" ]]; then
        paths="${paths}_old_clf"
        dump_dir="${base_dump_dir}/loop3_replicate_old_clf/${OPT}-${QUERY_TYPE}-${SCHEDULER}"
        loss=f"${loss}_old_clf"
    fi
    ;;
  *)
    echo "Unknown LOOP value: $LOOP"
    exit 1
    ;;
esac


# ---- 8. Prepare command ----
if [[ "${ENV_MANAGER}" == "conda" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME} --no-capture-output"
elif [[ "${ENV_MANAGER}" == "micromamba" ]]; then
  CMD="${ENV_MANAGER} run -n ${ENV_NAME}"
else
  echo "❌ Unknown env manager: $ENV_MANAGER"
  exit 1
fi


# ---- 9. Handle hydra errors ----
export HYDRA_FULL_ERROR=1
    
# ---- 10. Run command ----
$CMD python scripts/inverse/experiment/${script_name}.py \
    --multirun \
    +launcher=slurm \
    +hparam_search="${hparam_search}" \
    constraints="${constraints}" \
    loss_guidance.num_forward_pass_per_sample="${N_FWD_SAMPLES}" \
    paths="${paths}" \
    loss="${loss}" \
    paths.dump_dir=${dump_dir} \
    sampling.target_cell_type='"'"${TARGET_CELL_TYPE}"'"' \
    sampling.query_pure_cell_types="${query_pure}" \
    sampling.mask_gradients="${mask_gradients}" \
    sampling.N="${N_SAMPLES}" \
    annotation="${annotation}"
