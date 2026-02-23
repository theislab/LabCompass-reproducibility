from datetime import datetime
import logging
import os
import sys
import uuid

import cloudpickle
from hydra import initialize, compose
from omegaconf import DictConfig
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import torch

from sc_exp_design.utils import set_reproducibility


BASE_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"

logger = logging.getLogger(__name__)


def main(config: DictConfig):
    ################################################################
    ############# PREPARE SCRIPT
    ################################################################
    # lazily import modules 
    sys.path.insert(0, os.path.join(BASE_DIR, "shared_utils"))
    from experiment_utils import (
        get_forward_model,
        create_dir,
        get_sample_grid,
        get_sensitivity_results,
    )
    from z_norm_modules import get_rescaling
    from plot_utils import plot_sensitivity_analysis_results

    # Create run id 
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_id}"
    logger.info(f"Starting inverse run {run_id}...")

    # Set reproducibility
    logger.info(f"Reproducibility set to {config.random_seed}")
    set_reproducibility(config.random_seed)

    # initialize base config
    logger.info("Starting script for running sensitivity analysis of inverse results.")
    logger.info(f"Initializing base configuration.")
    with initialize(config_path=config.base_config_path, version_base=None):
        base_cfg = compose(config_name=config.base_config_name)

    ct_string = config.target_cell_type
    ct_safe_string = ct_string.replace("/", ":") # cell type dir

    ################################################################
    ############# LOAD MODELS AND PREPARE DATA
    ################################################################
    # load forward model
    forward_model, (
        perturbation_response_prediction_model,
        target_prediction_model,
    ) = get_forward_model(base_cfg, logger=logger)
    forward_model.target_prediction_model.resc_model["model"].eval()
    forward_model.forward_model.velocity_field.eval()

    # data for cellular response prediction model
    train_adata_phi = perturbation_response_prediction_model.train_data.adata
    val_adata_phi = perturbation_response_prediction_model.validation_data[0].adata
    logger.info(f"{train_adata_phi=}\n{val_adata_phi=}")

    # concatenate + compute pcs, neighbors and umap
    adata_phi = sc.concat((train_adata_phi, val_adata_phi), uns_merge="same")

    # data for classifier model
    train_adata_g = target_prediction_model.train_data.adata
    val_adata_g = target_prediction_model.validation_data.adata
    logger.info(f"{train_adata_g=}\n{val_adata_g=}")

    # concatenate data
    adata_g = sc.concat((train_adata_g, val_adata_g), uns_merge="same")

    # rescale features back from g to phi
    izn_g = get_rescaling(train_adata_g, inverse=True)
    zn_phi = get_rescaling(train_adata_phi)
    X_true = torch.from_numpy(adata_g.obsm[perturbation_response_prediction_model.data_manager.sample_rep]).float().cuda()
    X_true = izn_g(X_true)
    X_true = zn_phi(X_true).detach().cpu().numpy()
    adata_g.obsm["rescaled_features"] = X_true

    # Prepare label encoder
    ct_le = LabelEncoder()
    ct_values = adata_g.obs["cell_type"].values
    ct_le.fit(ct_values)

    # get data bounds
    logger.info(f"Computing bounds on condition data...")
    conc_unique = np.unique(adata_phi.obsm["condition_concat"], axis=0)
    conc_ubound = np.max(adata_phi.obsm["condition_concat"], axis=0)
    conc_lbound = np.min(adata_phi.obsm["condition_concat"], axis=0)
    logger.info(f"{conc_unique.shape=}, {conc_ubound.shape=}, {conc_lbound.shape=}")

    ################################################################
    ############# LOAD RESULTS AND PREPARE FOLDERS
    ################################################################
    # construct current cell type directory
    logger.info(f"Starting sensitivity analysis for CT {ct_string} and run {config.experiment_type}")
    target_ct_dir = os.path.join(config.base_dir, config.experiment_type)
    target_ct_dir = os.path.join(target_ct_dir, ct_safe_string)
    run_dir = os.path.join(target_ct_dir, config.run)
    sensitivity_res_dir = os.path.join(run_dir, "sensitivity_analysis")
    dump_dir = os.path.join(sensitivity_res_dir, run_id)
    plot_dir = os.path.join(dump_dir, "plots")
    logger.info(f"Reading cell type results results from directory {target_ct_dir}")
    logger.info(f"Sensivity analysis data will be saved at {dump_dir}")
    logger.info(f"Sensivity analysis plots will be saved at {plot_dir}")
    create_dir(sensitivity_res_dir, logger=logger)
    create_dir(dump_dir, logger=logger)
    create_dir(plot_dir, logger=logger)

    # read run files
    inverse_results = np.load(
        os.path.join(run_dir, "inverse_results.npz")
    )

    # prepare grid
    logger.info("Preparing grid.")
    samples = np.maximum(0, inverse_results["trajectory"][:, -1, :])
    sample_grid = get_sample_grid(samples, grid_size=config.grid_size, lbound=conc_lbound, ubound=conc_ubound)
    noise = torch.from_numpy(inverse_results["noise"]).float().to(perturbation_response_prediction_model.device_id)
    logger.info(f"Grid ready! {samples.shape=}, {sample_grid.shape=}, {noise.shape=}.")

    # retrieve sensitivity results
    logger.info(f"Starting evaluation of forward model on grid.")
    sensitivity_results_dict = get_sensitivity_results(
        forward_model,
        sample_grid,
        noise,
        base_cfg.forward_model.num_time_steps,
        base_cfg.forward_model.solver_kwargs,
        ct_le,
    )
    logger.info(f"Sensitivity Analysis Ran. Saving the results.")
    with open(os.path.join(dump_dir, "sensitivity_response_data.pkl"), "wb") as fb:
        cloudpickle.dump(sensitivity_results_dict, fb)

    # plotting sensitivity analysis results
    logger.info("Generating plot...")
    fig = plot_sensitivity_analysis_results(
        sensitivity_results_dict,
        base_cfg.annotation.protocol_columns,
        ct_string,
        ct_le,
    )
    plot_path = os.path.join(plot_dir, "sensitivity_analysis.svg")
    logger.info(f"Plot generated, it will be saved at {plot_path}")
    fig.savefig(
        plot_path,
        dpi=300
    )
    logger.info("Figured saved to disk.")
    logger.info("Run finished with exit code 0, goodbye!")


def parse_args():
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_cell_type", required=False, default="CyclingProgenitor*")
    parser.add_argument("--run", required=False, default="2026-02-18_03-46-45_ce20f2e0")
    parser.add_argument("--experiment_type", required=False, default="unconstrained-pure_populations-reciprocal")
    parser.add_argument("--base_config_path", required=False, default="../../../inverse/loss_guidance/config")
    parser.add_argument("--base_config_name", required=False, default="run_inverse")
    parser.add_argument("--base_dir", required=False, default="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/inverse/loss_guidance/raw_data")
    parser.add_argument("--grid_size", required=False, default=10)
    parser.add_argument("--random_seed", required=False, default=42)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
