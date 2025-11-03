import logging
import os
import yaml

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import scanpy as sc
import seaborn as sns
from tqdm import tqdm
import wandb

from sc_exp_design.constants import PredictionFields, DataFields
from sc_exp_design.models import FlowMatching
from sc_exp_design.utils import set_reproducibility
from sc_exp_design.training.callbacks import MetricsCallBack

from train_flow_matching import get_adata_splits, state_transforms
from validation_utils import validate_on_ood_data

logger = logging.getLogger(__name__)



def validate_run(
    config,
    run
):

    # 2. retrieving adata and ensuring reproducibility
    _, ood_adatas_dict = get_adata_splits(run.config)

    # 3. loading model
    logger.info("Loading model...")
    model_path = os.path.join(run.config.paths.dump_dir, f"{run.name}_FlowMatching.pkl")
    flow_matching = FlowMatching.load(model_path)
    logger.info("Model loaded")

    # 4. preparing validation data
    logger.info("Preparing OOD data...")
    ood_data_dict = {
        k: flow_matching.data_manager.get_data(v) for k, v in ood_adatas_dict.items()
    }
    logger.info("OOD data ready!")

    # 5. preparing metrics
    metrics_callback = MetricsCallBack(
        config.validation.metrics,
        state_transforms=state_transforms.get(run.config.training.state_transforms, None),
    )

    # 6. runnning validation
    results_dict = {}
    for ood_idx, ood_data in ood_data_dict.items():
        results_dict[ood_idx] = validate_on_ood_data(
            config.sampling.N,
            flow_matching,
            ood_data,
            metrics_callback,
        )
    return results_dict


@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/generative_modeling/conditional_flow_matching/config/",
    config_name="validate_cfm",
)
def main(config: DictConfig):

    # 0. load wandb run and set reproducibility
    logger.info(f"Loading wandb project {config.run.team}/{config.run.project}.")
    api = wandb.Api()
    runs = api.runs(f"{config.run.team}/{config.run.project}")
    logger.info(f"Project loaded. Setting random seed to {config.reproducibility.seed}.")
    set_reproducibility(config.reproducibility.seed)

    # 1. retrieving runs for current column and create dump directory
    logger.info(f"Loading OOD runs for {config.ood.obs_column}")
    runs = [
        run for run in runs if run.config["ood"]["obs_column"] == config.ood.obs_column
        and run.config["ood"]["unique_value_ids"] == config.ood.unique_value_ids
    ]
    logger.info(f"Found {len(runs)} for the currrent column")
    column_dir = os.path.join(config.paths.results_dir, config.ood.obs_column)
    ood_val_dir = os.path.join(column_dir, config.ood.unique_value_ids)
    create_dir(column_dir, logger)

    # 2. iterating ove the runs
    for run_id, run in enumerate(runs):
        # 2.0 creating directory for current run
        run_dir = os.path.join(ood_val_dir, f"{run.name}")
        logger.info(f"Validating run {run_id} with name {run.name} for value {config.ood.unique_value_ids} ({run_id + 1}/{len(runs)})")
        create_dir(run_dir, logger)

        # 2.1 running validation on ood data
        logger.info("Running validation.")
        run_results_dict = validate_run(
            config,
            run
        )

        # 2.2 Iterate over the ood conditions
        logger.info(f"Validation done, processing results for each OOD condition...")
        pbar = tqdm(range(len(run_results_dict)))
        for ood_cond, ood_results in run_results_dict.items():
            # creating directory for current condition
            pbar.set_description(f"Validating on ood condition {ood_cond} for column {run.config.ood.obs_column}...")
            pbar.update()
            cond_dir = os.path.join(run_dir, f"{ood_cond}")
            create_dir(cond_dir, logger)

            # retrieving results
            cond_metrics = ood_results["metrics"]
            cond_samples = ood_results["samples"]

            # storing metrics and samples
            metrics_path = os.path.join(cond_dir, f"{run.name}_metrics.yaml")
            with open(metrics_path, "w") as fb:
                yaml.dump(cond_metrics, fb, default_flow_style=False)

            # storing samples
            samples_path = os.path.join(cond_dir, f"{run.name}_samples.npz")
            np.savez(samples_path, **cond_samples)

            # plot results
            adata_gen = plot_results_and_return_adata(
                cond_dir,
                run.name,
                cond_samples,
                is_pca="pca" in run.config["data"]["sample_rep"]
            )
            adata_path = os.path.join(cond_dir, f"{run.name}_adata.h5ad")
            adata_gen.write_h5ad(adata_path)

    logger.info("All OOD runs validated successfully.")

