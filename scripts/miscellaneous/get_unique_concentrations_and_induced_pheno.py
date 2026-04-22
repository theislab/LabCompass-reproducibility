from collections import defaultdict
import logging
import sys
import yaml

import hydra
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)

ANNOTATION_CONFIG_PATH = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/forward/conditional_model/flow_matching/config/annotation/default.yaml"
DATA_CONFIG_PATH = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/forward/conditional_model/flow_matching/config/data/protocol_concat.yaml"
OUT_ADATA_PATH = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/output/miscellaneous/unique_concentrations_adata.h5ad"
BATCH_SIZE = 500_000


def predict_on_conc(
    X,
    concentrations,
    unique_concs,
    izn_phi,
    zn_g,
    target_prediction_model,
    classes,
    rescale=True,
):
    # list of mean probabilities
    mean_probs_df = defaultdict(list)

    # iterate over unique values to compute mean probs
    pbar = tqdm(range(unique_concs.shape[0]))
    for conc in range(unique_concs.shape[0]):
        # rescaling state data
        conc_idxs = np.all(concentrations == unique_concs[conc], axis=1)
        X_conc = X[conc_idxs]

        # updating progress bar
        pbar.set_description(f"{X_conc.shape=}")
        pbar.update()

        # compute predicted mean probabilities
        all_probs = []
        with torch.no_grad():
            for i in range(0, X_conc.shape[0], BATCH_SIZE):
                batch = torch.from_numpy(X_conc[i:i + BATCH_SIZE]).cuda()
                if rescale:
                    batch = izn_phi(batch)
                    batch = zn_g(batch)
                logits = target_prediction_model.target_prediction_model(batch)["cell_type"]
                probs = torch.nn.functional.softmax(logits, dim=1)
                all_probs.append(probs)

        g_probs = torch.cat(all_probs, dim=0)
        g_mean_probs = g_probs.mean(0).cpu().numpy()

        for idx, ct in enumerate(classes):
            mean_probs_df[ct.replace("/", "_").replace("*", "").replace(" ", "_")].append(g_mean_probs[idx])
    return pd.DataFrame(mean_probs_df)


def get_observed_proportions(adata, concentrations, unique_concs, classes):
    """Return DataFrame of observed cell type proportions for each unique concentration."""
    obs_props = defaultdict(list)
    cell_types = adata.obs["cell_type"].values
    
    for conc in range(unique_concs.shape[0]):
        conc_idxs = np.all(concentrations == unique_concs[conc], axis=1)
        conc_cell_types = cell_types[conc_idxs]
        # Count occurrences of each class
        counts = {ct: np.sum(conc_cell_types == ct) for ct in classes}
        total = len(conc_cell_types)
        proportions = {ct: counts[ct] / total if total > 0 else 0 for ct in classes}
        for ct in classes:
            col_name = ct.replace("/", "_").replace("*", "").replace(" ", "_") + "_observed"
            obs_props[col_name].append(proportions[ct])
    return pd.DataFrame(obs_props)


@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/loss_guidance/config",
    config_name="run_inverse"
)
def main(config):
    # lazily import modules
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils")
    from experiment_utils import get_forward_model
    from z_norm_modules import get_rescaling

    # open data config dict
    with open(DATA_CONFIG_PATH, "r") as fb:
        data_cfg_dict = yaml.safe_load(fb)
    annotation_dict = config.annotation

    # Get forward model
    logger.info(f"Preparing forward model...")
    forward_model, (
        phi_model,
        target_prediction_model
    ) = get_forward_model(config, logger=logger)
    target_prediction_model.target_prediction_model.eval()
    logger.info(f"Forward model ready!\n{forward_model}")

    # data for cellular response prediction model
    train_adata_phi = phi_model.train_data.adata
    val_adata_phi = phi_model.validation_data[0].adata
    adata_phi = sc.concat((train_adata_phi, val_adata_phi), uns_merge="same")
    logger.info(f"{train_adata_phi=}\n{val_adata_phi=}")

    # data for classifier model
    train_adata_g = target_prediction_model.train_data.adata
    val_adata_g = target_prediction_model.validation_data.adata
    adata_g = sc.concat((train_adata_g, val_adata_g), uns_merge="same")
    logger.info(f"{train_adata_g=}\n{val_adata_g=}")

    # Prepare label encoder
    ct_le = LabelEncoder()
    ct_values = train_adata_g.obs["cell_type"].values
    ct_le.fit(ct_values)
    classes = ct_le.classes_.tolist()

    # rescale features back from g to phi
    zn_g = get_rescaling(train_adata_g)
    izn_phi = get_rescaling(train_adata_phi, inverse=True)

    # get unique perturbation values
    logger.info("Getting unique concentrations")
    concentrations = adata_phi.obsm[annotation_dict["protocol_obsm_key"]]
    concentrations_g = adata_g.obsm[annotation_dict["protocol_obsm_key"]]
    unique_concs, idx = np.unique(concentrations, axis=0, return_index=True)
    protocol_df = adata_phi.obs[annotation_dict.protocol_columns].iloc[idx]
    logger.info(f"Found unique concentrations of shape {unique_concs.shape}, {protocol_df.shape}")

    # extract data
    X = adata_phi.obsm[data_cfg_dict["sample_rep"]] if \
        data_cfg_dict["sample_rep"] is not None else adata_phi.X
    logger.info(f"{X.shape=}")

    # extract data
    X_g = adata_g.obsm[data_cfg_dict["sample_rep"]] if \
        data_cfg_dict["sample_rep"] is not None else adata_g.X
    logger.info(f"{X_g.shape=}")   # fixed log

    # predict on phi data
    mean_probs_df_full = predict_on_conc(
        X,
        concentrations,
        unique_concs,
        izn_phi,
        zn_g,
        target_prediction_model,
        classes,
        rescale=True,
    )
    logger.info(f"{mean_probs_df_full.head()=}")

    # predict on g data (using concentrations_g)
    mean_probs_df_subset = predict_on_conc(
        X_g,
        concentrations_g,
        unique_concs,
        None,
        None,
        target_prediction_model,
        classes,
        rescale=False,
    )
    mean_probs_df_subset.columns = [f"{e}_subset" for e in mean_probs_df_subset.columns]
    logger.info(f"{mean_probs_df_subset.head()=}")

    # Compute observed proportions
    observed_props_df = get_observed_proportions(adata_g, concentrations_g, unique_concs, classes)
    logger.info(f"Observed proportions shape: {observed_props_df.shape}")

    # Compute cell counts per condition in full and subset
    n_cells_full = []
    n_cells_subset = []
    for conc in range(unique_concs.shape[0]):
        conc_idxs_full = np.all(concentrations == unique_concs[conc], axis=1)
        conc_idxs_subset = np.all(concentrations_g == unique_concs[conc], axis=1)
        n_cells_full.append(np.sum(conc_idxs_full))
        n_cells_subset.append(np.sum(conc_idxs_subset))
    
    counts_df = pd.DataFrame({
        "n_cells_full": n_cells_full,
        "n_cells_subset": n_cells_subset
    })
    logger.info(f"Counts DF shape: {counts_df.shape}")

    # Combine all DataFrames
    obs_df = pd.concat([mean_probs_df_full, mean_probs_df_subset, observed_props_df, counts_df], axis=1)
    logger.info(f"Final DF of shape: {obs_df.shape}")
    logger.info(f"Final columns: {obs_df.columns.tolist()}")

    # constructing unique concentrations adata
    unique_concs_adata = sc.AnnData(
        X=protocol_df.values,
        obs=obs_df,
        obsm={"log_conc": unique_concs},
        var=pd.DataFrame(index=annotation_dict["protocol_columns"])
    )
    logger.info(f"{unique_concs_adata=}, writing to {OUT_ADATA_PATH}")
    unique_concs_adata.write_h5ad(OUT_ADATA_PATH)


if __name__ == "__main__":

    main()
