#!/bin/bash

EXPERIMENTS=(
    "penalized_all_axes-pure_populations-constant"
    "penalized_all_axes-pure_populations-reciprocal"
    "penalized_oxy_days-pure_populations-constant"
    "penalized_oxy_days-pure_populations-reciprocal"
    "unconstrained-pure_populations-constant"
    "unconstrained-pure_populations-reciprocal"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "Submitting $exp"
    sbatch ../sbatch_launchers/inverse/plot/launch_uncertainty_annotation.sbatch "$exp"
done