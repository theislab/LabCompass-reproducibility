import logging
import os
import sys
import yaml

from hydra import initialize, compose
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm


BASE_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"

logger = logging.getLogger(__name__)


def main(config: DictConfig):
    ################################################################
    ############# PREPARE SCRIPT
    ################################################################
    # lazily import modules 
    sys.path.insert(0, os.path.join(BASE_DIR, "shared_utils"))
    from experiment_utils import get_forward_model, create_dir
    from plot_utils import (
        plot_adata,
        run_level_plots,
        covariate_level_plots,
    )
    from z_norm_modules import get_rescaling

    # initialize base config
    logger.info("Starting script for generating plots of inverse results")
    logger.info(f"Initializing base configuration.")
    with initialize(config_path=config.paths.base_config_path, version_base=None):
        base_cfg = compose(config_name=config.paths.base_config_name)
    annotation_dict = base_cfg.annotation

    ################################################################
    ############# LOAD MODELS AND PREPARE DATA
    ################################################################
    # load forward model
    forward_model, (
        perturbation_response_prediction_model,
        target_prediction_model,
    ) = get_forward_model(base_cfg, logger=logger)

    # data for cellular response prediction model
    train_adata_phi = perturbation_response_prediction_model.train_data.adata
    val_adata_phi = perturbation_response_prediction_model.validation_data[0].adata
    logger.info(f"{train_adata_phi=}\n{val_adata_phi=}")

    # subsampling data
    train_adata_phi = sc.pp.sample(
        train_adata_phi,
        n=config.subsample.n_train_cells,
        copy=True
    )
    val_adata_phi = sc.pp.sample(
        val_adata_phi,
        n=config.subsample.n_val_cells,
        copy=True
    )
    logger.info(f"{train_adata_phi=}, {val_adata_phi=}")

    # concatenate + compute pcs, neighbors and umap
    adata_phi = sc.concat((train_adata_phi, val_adata_phi), uns_merge="same")
    sc.pp.pca(adata_phi)
    sc.pp.neighbors(adata_phi)
    sc.tl.umap(adata_phi)

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
    classes = ct_le.classes_.tolist()

    ################################################################
    ############# LOAD RESULTS AND PREPARE FOLDERS
    ################################################################
    # construct current cell type directory
    logger.info(f"Generating plots for target cell type {config.target_ct}.")
    target_ct_dir = os.path.join(config.paths.target_ct_dir, config.target_ct)
    logger.info(f"Reading cell type results results from directory {target_ct_dir}")

    # read run files
    run_dir = os.path.join(target_ct_dir, config.run)
    fwd_results = np.load(
        os.path.join(run_dir, "fwd_results.npz")
    )
    inverse_results = np.load(
        os.path.join(run_dir, "inverse_results.npz")
    )

    # define plots dir
    plots_dir = os.path.join(run_dir, "plots")
    create_dir(plots_dir)

    # define run level plots dir
    run_level_plots_dir = os.path.join(plots_dir, "run")
    create_dir(run_level_plots_dir)

    # define samples level plots dir
    samples_level_plots_dir = os.path.join(plots_dir, "samples")
    create_dir(samples_level_plots_dir)

    ################################################################
    ############# RUN LEVEL VISUALIZATIONS
    ################################################################
    run_level_plots(
        perturbation_response_prediction_model,
        cond_adata,
        config.target_ct,
        run_level_plots_dir,
        fwd_results,
        inverse_results,
        annotation_dict,
        classes,
        data_color_val,
        gen_color_val,
        n_noise_samples=config.n_noise_samples,
    )

    ################################################################
    ############# COVARIATE LEVEL VISUALIZATIONS
    ################################################################

    ...

    ################################################################
    ############# SAMPLE LEVEL VISUALIZATIONS
    ################################################################

    # iterate over each sample
    for idx in loss.shape[0]:
        # create sample plot dir
        sample_dir = os.path.join(samples_level_plots_dir, str(idx))
        create_dir(sample_dir)

        # get adata
        adata_pred_min = get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, idx)
        plot_adata(adata_pred_min, config.target_ct, classes, sample_dir, "")


def parse_args():
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ct", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--n_noise_samples", required=False, default=10_000)
    return parser.parse_args()


def run():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    run()
