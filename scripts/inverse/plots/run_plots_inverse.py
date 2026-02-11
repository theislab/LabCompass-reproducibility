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

from sc_exp_design.constants import ParamsFields

ANNOTATION_CONFIG_PATH = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/generative_modeling/conditional_flow_matching/config/annotation/default.yaml"
BASE_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"
# RESULTS_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/results"
RESULTS_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/results_new"
CONFIG_DIR = "../config"
CONFIG_NAME = "run_inverse"
CELL_STATE_REP = "X_channel_standardized+X_scatter_standardized"
CONDITION_REP = "condition_concat"
n_train_cells = 70_000
n_val_cells = 30_000

logger = logging.getLogger(__name__)


def main(config: DictConfig):
    # lazily import modules 
    sys.path.insert(0, os.path.join(BASE_DIR, "shared_utils"))
    from experiment_utils import get_forward_model, create_dir
    from plot_utils import (
        get_adata_from_idx,
        plot_heatmap,
        plot_loss_history,
        plot_adata,
    )
    from z_norm_modules import get_rescaling

    # initialize base config
    logger.info("Starting script for generating plots of inverse results")
    logger.info(f"Initializing base configuration.")
    with initialize(config_path=CONFIG_DIR, version_base=None):
        base_cfg = compose(config_name=CONFIG_NAME)

    # open annotation dict
    with open(ANNOTATION_CONFIG_PATH, "r") as fb:
        annotation_dict = yaml.safe_load(fb)

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
        n=n_train_cells,
        copy=True
    )
    val_adata_phi = sc.pp.sample(
        val_adata_phi,
        n=n_val_cells,
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
    X_true = torch.from_numpy(adata_g.obsm[CELL_STATE_REP]).float().cuda()
    X_true = izn_g(X_true)
    X_true = zn_phi(X_true).detach().cpu().numpy()
    adata_g.obsm["rescaled_features"] = X_true

    # Prepare label encoder
    ct_le = LabelEncoder()
    ct_values = adata_g.obs["cell_type"].values
    ct_le.fit(ct_values)
    classes = ct_le.classes_.tolist()

    # construct current cell type directory
    logger.info(f"Generating plots for target cell type {config.target_ct}.")
    target_ct_dir = os.path.join(RESULTS_DIR, config.target_ct)
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

    # parse inverse results archive
    loss_history = inverse_results["loss_history"]
    loss = loss_history[:, -1]

    # loss history
    loss_history_fig = plot_loss_history(config.target_ct, loss_history)
    loss_history_fig.savefig(
        os.path.join(plots_dir, "loss_history.png"),
        dpi=300,
    )
    plt.close(loss_history_fig)

    # heatmap samples
    plot_heatmap(
        plots_dir,
        classes,
        config.target_ct,
        fwd_results,
        inverse_results,
        annotation_dict,
        samples_vmin=0.0,
        samples_vmax=10.0,
        loss_vmin=0.0,
        loss_vmax=10.0,
        plot_pheno=False
    )

    # heatmap pheno
    plot_heatmap(
        plots_dir,
        classes,
        config.target_ct,
        fwd_results,
        inverse_results,
        annotation_dict,
        samples_vmin=0.0,
        samples_vmax=10.0,
        loss_vmin=0.0,
        loss_vmax=10.0,
        plot_pheno=True
    )

    # min sample
    adata_pred_min = get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, loss.argmin())
    plot_adata(adata_pred_min, config.target_ct, classes, plots_dir, "min_loss")
    # first quartile
    adata_pred_25q = get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, np.argsort(loss)[len(loss)//4])
    plot_adata(adata_pred_25q, config.target_ct, classes, plots_dir, "first_quartile")
    # median
    adata_pred_median = get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, np.argsort(loss)[len(loss)//2])
    plot_adata(adata_pred_median, config.target_ct, classes, plots_dir, "median")
    # third quartile
    adata_pred_75q = get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, np.argsort(loss)[(len(loss)//4)*3])
    plot_adata(adata_pred_75q, config.target_ct, classes, plots_dir, "third_quartile")
    # max sample
    adata_pred_max = get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, loss.argmax())
    plot_adata(adata_pred_max, config.target_ct, classes, plots_dir, "max_loss")


def parse_args():
    import argparse    
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ct", required=True)
    parser.add_argument("--run", required=True)
    return parser.parse_args()


def run():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    run()
