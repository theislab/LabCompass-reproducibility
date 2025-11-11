import logging
import os
import sys
import yaml

import hydra
from omegaconf import DictConfig
import scanpy as sc
from tqdm import tqdm

sys.path.insert(0, "../../scripts")
from data_utils import annotate_perturbations, get_protocol_tranformations, apply_shared_transformations
from train_utils import resolve_omegaconf_to_dictionary
from distances import compute_e_distance


logger = logging.getLogger(__name__)

distances_dict = {
    "e_distance": compute_e_distance
}


def get_adata(config: DictConfig):
    # Data 0. read data
    logger.info("Reading data...")
    adata = sc.read_h5ad(config.paths.h5ad_path)
    logger.info(f"Data read! {adata}")

    # Data 1. annotate perturbation data
    logger.info("Annotating perturbation data...")
    column2tranform = get_protocol_tranformations(
        config.annotation.protocol_columns,
        log1p_exp_cols=config.annotation.log1p_exp_cols,
        log21p_exp_cols=config.annotation.log21p_exp_cols,
    )
    adata = annotate_perturbations(
        adata,
        config.annotation.protocol_columns,
        protocol_obs_key_added=config.annotation.protocol_obs_key_added,
        protocol_sep=config.annotation.protocol_sep,
        one_hot_uns_key_added=config.annotation.one_hot_uns_key_added,
        column2tranform=column2tranform,
        protocol_obsm_key=config.annotation.protocol_obsm_key,
    )
    logger.info(f"Perturbation data annotated! {adata}")

    # Data 3. apply transformations
    logger.info("Computing tranformation params on train data and applying to both train and ood data...")
    adata = apply_shared_transformations(
        adata,
        None,
        config.transforms.scatter_columns,
        compute_channel_pcs=config.transforms.compute_channel_pcs,
    )
    logger.info("Shared tranformations applied!")
    return adata, None


def compute_distance_fn(
    adata,
    state_repr,
    groups,
    distance_fn,
    sep = "|",
    **kwargs
):
    # define dictionary to store results
    results_dict = {}

    # define progress bar
    n_groups = len(groups)
    pbar = tqdm(range())

    # outer loop: iterating over each group
    for idx0, (group0_id, group0_idxs) in enumerate(groups.items()):
        # retrieving states
        X0 = adata[group0_idxs].obsm[state_repr]

        # inner loop: iterating over each group
        for idx1, (group1_id, group1_idxs) in enumerate(groups.items()):
            # skipping already computed distances
            if idx1 <= idx0:
                continue
            # retrieving states
            X1 = adata[group1_idxs].obsm[state_repr]

            # updating progress bar
            pbar.set_description(f"Computing Distances {group0_id}:{X0.shape[0]}:({idx0}/{n_groups}) <-> {group1_id}:{X1.shape[0]}:({idx1}/{n_groups})")

            # defining key for storing results
            key = f"{group0_id}{sep}{group1_id}"

            # computing distances
            results_dict[key] = distance_fn(X0, X1, **kwargs).item()
    return results_dict


@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/data_analysis/subpopulation_distances/config",
    config_name="compute_subpop_dists"
)
def main(
    config: DictConfig,
) -> int:
    
    # 0. Get data
    logger.info(f"Starting script. Reading data...")
    adata = get_adata(config)

    # 1. Group by target column
    logger.info(f"Grouping adata by {config.distance.groups_obs_key}...")
    groups = adata.obs.groupby(by=config.distance.groups_obs_key).groups
    logger.info(f"Found {len(groups)} groups.")

    # 2. Compute distance function
    results_dict = compute_distance_fn(
        adata,
        config.distance.state_repr,
        groups,
        distances_dict.get(config.distance.distance_fn, compute_e_distance),
        sep = "|",
        **resolve_omegaconf_to_dictionary(config.distance.distance_kwargs)
    )

    # 3. storing results
    results_path = os.path.join(
        config.paths.results_dir, f"{config.distance.state_repr}_{config.distance.groups_obs_key}_{config.distance.distance_fn}.yaml"
    )
    logger.info(f"Storing results at {results_path}")
    with open(results_path, "w") as fb:
        yaml.dump(results_dict, fb, default_flow_style=False)
    logger.info("Results dumped, exiting successfully!")
