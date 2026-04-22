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
    unique_concs = np.unique(concentrations, axis=0)
    protocol_df = adata_phi.obs[annotation_dict.protocol_columns].drop_duplicates()
    logger.info(f"Found unique concentrations of shape {unique_concs.shape}, {protocol_df.shape}")

    # extract data
    X = adata_phi.obsm[data_cfg_dict["sample_rep"]] if \
        data_cfg_dict["sample_rep"] is not None else adata_phi.X
    logger.info(f"{X.shape=}")

    # list of mean probabilities
    mean_probs_df = defaultdict(list)

    # iterate over unique values to compute mean probs
    pbar = tqdm(range(unique_concs.shape[0]))
    for conc in pbar:
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
                batch = izn_phi(batch)
                batch = zn_g(batch)
                logits = target_prediction_model.target_prediction_model(batch)["cell_type"]
                probs = torch.nn.functional.softmax(logits, dim=1)
                all_probs.append(probs)

        g_probs = torch.cat(all_probs, dim=0)
        g_mean_probs = g_probs.mean(0).cpu().numpy()

        for idx, ct in enumerate(classes):
            mean_probs_df[ct.replace("/", "_").replace("*", "").replace(" ", "_")].append(g_mean_probs[idx])
    mean_probs_df = pd.DataFrame(mean_probs_df)
    logger.info(f"{mean_probs_df.head()=}")

    # constructing unique concentrations adata
    unique_concs_adata = sc.AnnData(
        X=protocol_df.values,
        obs=mean_probs_df,
        obsm={"log_conc": unique_concs},
        var=pd.DataFrame(index=annotation_dict["protocol_columns"])
    )
    logger.info(f"{unique_concs_adata=}, writing to {OUT_ADATA_PATH}")
    unique_concs_adata.write_h5ad(OUT_ADATA_PATH)


if __name__ == "__main__":

    main()
