#!/bin/bash

# Submits one slimming job per loop. Each job reads that loop's checkpoints and
# writes inference-only copies; originals are never modified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ENV_NAME="sc_exp_design_12"

while [[ $# -gt 0 ]]; do
  case $1 in
    -e|--env-name) ENV_NAME="$2"; shift 2 ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
done

LOOPS=(
    "loop0"
    "loop1"
    "loop2"
    "loop2p5"
    "loop3"
    "loop4"
)

for loop in "${LOOPS[@]}"; do
    echo "Submitting ${loop}"
    sbatch "${REPO_ROOT}/sbatch/sbatch_launchers/release/slim_checkpoints.sbatch" "${loop}" "${ENV_NAME}"
done
