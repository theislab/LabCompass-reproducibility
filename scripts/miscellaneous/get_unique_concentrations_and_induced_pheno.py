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

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
import os
os.environ["REPO_ROOT"] = str(REPO_ROOT)


logger = logging.getLogger(__name__)

OUT_ADATA_PATH = str(REPO_ROOT / "project_folder" / "output/miscellaneous_new/unique_concentrations_adata_bloodplus.h5ad")
BATCH_SIZE = 500_000


@hydra.main(
    config_path=str(REPO_ROOT / "inverse/loss_guidance/config"),
    config_name="run_inverse",
    version_base=None
)
def main(config):
    # lazily import modules
    sys.path.insert(0, str(REPO_ROOT / "shared_utils"))
    from experiment_utils import (
        get_forward_model,
        compute_condition_means,
        clean_name,
        get_observed_proportions_aligned,
    )
    from z_norm_modules import get_rescaling
    from data_utils import get_protocol_tranformations
    from inverse_utils import map_df

    # open data config dict
    annotation_dict = config.annotation

    # Get forward model
    logger.info("Preparing forward model...")
    forward_model, (phi_model, target_prediction_model) = get_forward_model(config, logger=logger)
    target_prediction_model.target_prediction_model.eval()
    logger.info("Forward model ready!")

    # data for cellular response prediction model (full)
    train_adata_phi = phi_model.train_data.adata
    val_adata_phi = next(iter(phi_model.validation_data.values())).adata
    adata_phi = sc.concat((train_adata_phi, val_adata_phi), uns_merge="same")
    logger.info(f"Full data shape: {adata_phi.shape}")

    # data for classifier model (subset)
    train_adata_g = target_prediction_model.train_data.adata
    val_adata_g = next(iter(phi_model.validation_data.values())).adata
    adata_g = sc.concat((train_adata_g, val_adata_g), uns_merge="same")
    logger.info(f"Subset data shape: {adata_g.shape}")

    # Prepare label encoder for cell types
    ct_le = LabelEncoder()
    ct_values = train_adata_g.obs["cell_type"].values
    ct_le.fit(ct_values)
    classes = ct_le.classes_.tolist()
    logger.info(f"Cell types: {classes}")

    # Rescaling modules
    zn_g = get_rescaling(train_adata_g)
    izn_phi = get_rescaling(train_adata_phi, inverse=True)
    def rescale_full(batch):
        batch = izn_phi(batch)
        batch = zn_g(batch)
        return batch

    # -------- 1. Build condition IDs from raw protocol values (full data) --------
    protocol_cols = annotation_dict["protocol_columns"]
    for col in protocol_cols:
        adata_phi.obs[col] = adata_phi.obs[col].astype(float)
    raw_full = adata_phi.obs[protocol_cols].values  # shape (n_cells_full, n_protocols)
    unique_raw, cond_id_full = np.unique(raw_full, axis=0, return_inverse=True)
    n_conditions = unique_raw.shape[0]
    logger.info(f"Found {n_conditions} unique raw conditions")

    # Create mapping from raw tuple -> condition ID
    raw_to_cond = {tuple(row): i for i, row in enumerate(unique_raw)}

    # -------- 2. Get transformed concentrations for each condition (for output) --------
    # These are already stored in adata_phi.obsm[protocol_obsm_key] for each cell.
    # We'll take the first cell of each condition as representative.
    concentrations_full = adata_phi.obsm[annotation_dict["protocol_obsm_key"]]  # (n_cells_full, n_protocols)
    unique_concs_transformed = np.zeros((n_conditions, concentrations_full.shape[1]))
    for cond_id in range(n_conditions):
        first_idx = np.where(cond_id_full == cond_id)[0][0]
        unique_concs_transformed[cond_id] = concentrations_full[first_idx]
    logger.info(f"Unique transformed concentrations shape: {unique_concs_transformed.shape}")

    # -------- 3. Map subset data to the same condition IDs --------
    for col in protocol_cols:
        adata_g.obs[col] = adata_g.obs[col].astype(float)
    raw_subset = adata_g.obs[protocol_cols].values
    cond_id_subset = np.full(raw_subset.shape[0], -1, dtype=int)
    for i, row in enumerate(raw_subset):
        key = tuple(row)
        if key in raw_to_cond:
            cond_id_subset[i] = raw_to_cond[key]
    # Remove cells that don't match any full condition (should be none if data is consistent)
    valid_subset = cond_id_subset != -1
    if not np.all(valid_subset):
        logger.warning(f"Dropping {np.sum(~valid_subset)} subset cells with unknown raw combos")
        adata_g = adata_g[valid_subset].copy()
        cond_id_subset = cond_id_subset[valid_subset]
        raw_subset = raw_subset[valid_subset]
    logger.info(f"Subset after alignment: {adata_g.shape}")

    # -------- 4. Extract feature matrices --------
    sample_rep = config.data.sample_rep
    X_full = adata_phi.obsm[sample_rep] if sample_rep is not None else adata_phi.X
    X_subset = adata_g.obsm[sample_rep] if sample_rep is not None else adata_g.X
    logger.info(f"X_full shape: {X_full.shape}, X_subset shape: {X_subset.shape}")

    # -------- 5. Compute mean predicted probabilities for full and subset --------
    unique_cond_list = list(range(n_conditions))
    
    logger.info("Computing full data predictions (with rescaling)...")
    mean_probs_full = compute_condition_means(
        X_full, cond_id_full, unique_cond_list,
        rescale_func=rescale_full,
        target_prediction_model=target_prediction_model,
        classes=classes
    )
    
    logger.info("Computing subset data predictions (no rescaling)...")
    mean_probs_subset = compute_condition_means(
        X_subset, cond_id_subset, unique_cond_list,
        rescale_func=None,
        target_prediction_model=target_prediction_model,
        classes=classes
    )
    # Rename columns to indicate they come from subset
    mean_probs_subset.columns = [f"{col}_subset" for col in mean_probs_subset.columns]
    
    # -------- 6. Observed proportions from subset --------
    logger.info("Computing observed proportions from subset...")
    obs_props = get_observed_proportions_aligned(adata_g, cond_id_subset, unique_cond_list, classes)
    
    # -------- 7. Cell counts per condition --------
    n_cells_full = [np.sum(cond_id_full == i) for i in unique_cond_list]
    n_cells_subset = [np.sum(cond_id_subset == i) for i in unique_cond_list]
    counts_df = pd.DataFrame({"n_cells_full": n_cells_full, "n_cells_subset": n_cells_subset})
    
    # -------- 8. Combine all results --------
    obs_df = pd.concat([mean_probs_full, mean_probs_subset, obs_props, counts_df], axis=1)
    logger.info(f"Final aligned DataFrame shape: {obs_df.shape}")
    
    # -------- 9. Build output AnnData (one row per unique condition) --------
    # Store raw values as X, transformed as obsm, and results in obs
    unique_concs_adata = sc.AnnData(
        X=unique_raw,   # raw concentration values
        obs=obs_df,
        obsm={"log_conc": unique_concs_transformed},
        var=pd.DataFrame(index=protocol_cols)
    )
    logger.info(f"Writing output to {OUT_ADATA_PATH}")
    unique_concs_adata.write_h5ad(OUT_ADATA_PATH)


if __name__ == "__main__":
    main()
