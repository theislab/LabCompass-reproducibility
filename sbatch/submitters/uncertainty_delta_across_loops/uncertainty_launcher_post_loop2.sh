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
    sbatch ../../sbatch_launchers/inverse/plot/uncertainty_delta_across_loops/launch_uncertainty_annotation_bloodplus_post_loop2.sbatch "$exp"
done