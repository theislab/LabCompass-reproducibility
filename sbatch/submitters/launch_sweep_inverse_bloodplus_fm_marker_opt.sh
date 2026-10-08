#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set -eu

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ENV_NAME="labcompass"
PATHS="bloodplus"          # base name; will append _fm if USE_FM=1
USE_FM=1
ENV_MANAGER="micromamba"

# Path to the inner sbatch script (adjust if needed)
SBATCH_SCRIPT="$REPO_ROOT/sbatch/sbatch_launchers/inverse/sweep/sweep_inverse_marker_opt.sbatch"

# Experimental grid
TARGET_CELL_TYPE="late_MgkPro"
# TARGET_MARKERS=("CD41a" "CD56" "EPCR")
TARGET_MARKERS=("CD56" "EPCR")
SCHEDULERS=("constant" "reciprocal")
OPTIMIZATION_TYPE=("unconstrained" "penalized_oxy_days" "penalized_all_axes")

# ----------------------------------------------------------------------
# Optionally append _fm if requested
# ----------------------------------------------------------------------
if [[ $USE_FM -eq 1 ]]; then
    PATHS="${PATHS}_fm"
fi

# ----------------------------------------------------------------------
# Submit all combinations
# ----------------------------------------------------------------------
for target_marker in "${TARGET_MARKERS[@]}"; do
    for scheduler in "${SCHEDULERS[@]}"; do
        for optimization_type in "${OPTIMIZATION_TYPE[@]}"; do
            echo "# -------------------------------- #"
            echo "Submitting: $target_marker | $scheduler | $optimization_type | paths=$PATHS"
            sbatch "$SBATCH_SCRIPT" \
                --target-marker "$target_marker" \
                --scheduler "$scheduler" \
                --optimization-type "$optimization_type"
        done
    done
done

echo "All jobs submitted."
