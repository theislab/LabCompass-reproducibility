import logging
import json
import os
import yaml
import sys

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

UTILS_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils"


def main(config: DictConfig):
    # ---- Import additional modules ----
    sys.path.insert(0, UTILS_DIR)
    from experiment_utils import get_forward_model,  generate_with_condition

    # ---- Prepare file names for current run ----
    run_dir = config.run_dir
    candidates_path = os.path.join(run_dir, "candidates.csv")
    config_path = os.path.join(run_dir, "config.yaml")
    masked_eval_dir = os.path.join(run_dir, "masked_eval")
    logger.info(f"Processing run {run_dir}, the masked evaluation results data will be dumped at {masked_eval_dir}")

    # ---- Create directory ----
    if not os.path.isdir(masked_eval_dir):
        logger.info("Creating directory for masked evaluation results...")
        os.makedirs(masked_eval_dir)
        logger.info("Results directory successfully created")

    # ---- Initialize inverse configurations ----
    logger.info(f"Initializing base configuration...")
    inv_config = OmegaConf.load(config_path)
    protocol_cols = inv_config.annotation.protocol_columns
    logger.info(f"Configuration initialized -> protocol columns: {protocol_cols}")

    # ---- Load forward model ----
    logger.info(f"Preparing forward model...")
    forward_model, (
        _,
        target_prediction_model
    ) = get_forward_model(config, logger=logger)
    logger.info(f"Forward model ready!\n{forward_model}")

    # ---- Prepare label encoder ----
    logger.info("Preparing label encoder...")  
    ct_le = LabelEncoder()
    ct_values = target_prediction_model.train_data.adata.obs["cell_type"].values
    ct_le.fit(ct_values)
    classes = ct_le.classes_.tolist()
    logger.info(f"Label encoder ready, {len(classes)} classes found.")

    # ---- Read optimization data ----
    logger.info(f"Reading optimization data...")
    opt_df = pd.read_csv(candidates_path)
    logger.info(f"Optimization data read {opt_df.shape}")

    # ---- Retrieve molecule concentrations ----
    logger.info(f"Retrieving molecule concentrations...")
    X_cond = opt_df[protocol_cols].values
    logger.info(f"Concentration retrieved and {X_cond.shape[1]} molecules found.")

    # ---- Log1p transformation on optimized concentrations ----
    logger.info(f"Applying log1p transformation to concentration values...")
    X_cond_log1p = np.log1p(X_cond)
    logger.info(f"Concentration values transformed with log1p.")

    # ---- Prepare conterfactual concentration data (i.e.: mask new molecules) ----
    cols_to_mask = json.loads(config.columns_to_mask)
    logger.info(f"Running counterfactual experiments by masking the {cols_to_mask} columns...")

    X_cond_log1p_masked = X_cond_log1p.copy()
    for col in cols_to_mask:
        col_idx = protocol_cols.index(col)
        X_cond_log1p_masked[:, col_idx] = 0.0
    logger.info(f"Masked data ready -> {X_cond_log1p_masked.shape}")

    # ---- Query the forward model with the masked conditions ----
    num_time_steps = config.num_time_steps
    solver_kwargs = json.loads(config.solver_kwargs)
    num_samples = config.num_samples
    cell_type_column = config.cell_type_column
    logger.info(f"Querying the forward model: {num_time_steps=}, {solver_kwargs=}, {num_samples=}, {cell_type_column=}")

    pred_masked = generate_with_condition(
        X_cond_log1p_masked,
        forward_model,
        num_time_steps,
        solver_kwargs,
        ct_le,
        num_samples=num_samples,
        logger=logger,
        cell_type_column=cell_type_column
    )

    X = pred_masked["X"]
    X_channel = pred_masked["X_channel"]
    X_scatter = pred_masked["X_scatter"]
    ct_probs = pred_masked["ct_probs"]
    logger.info(f"Forward model queried: {X.shape=}, {X_channel.shape=}, {X_scatter.shape=}, {ct_probs.shape=}")

    # ---- Aggregate the cell type proportions ----
    logger.info(f"Aggregating cell type probabilities over the generated subpopulation...")
    mean_ct_probs = ct_probs.mean(0)
    logger.info(f"Mean cell type probabilities of shape {mean_ct_probs.shape=}")

    # ---- Store generated cell type proportions for the masked predictions ----
    logger.info(f"Storing the predicted cell type proportions for the masked prediction data...")
    ct_probs_df = pd.DataFrame(
        mean_ct_probs,
        index=opt_df.index,
        columns=[f"{c}_prop" for c in classes]
    )
    logger.info(f"Cell type proportion DataFrame created: {ct_probs_df.shape}")

    # ---- Save generated prediction data to disk ----    
    fwd_results_path = os.path.join(masked_eval_dir, "masked_fwd_results.npz")
    logger.info(f"Saving raw masked forward results to {fwd_results_path}...")
    np.savez(fwd_results_path, **pred_masked)
    logger.info("Raw masked forward results saved to disk.")

    # ---- Save generated cell type proportions DataFrame to disk ----
    probs_csv_path = os.path.join(masked_eval_dir, "masked_ct_props.csv")
    logger.info(f"Saving generated cell type proportions to {probs_csv_path}...")
    ct_probs_df.to_csv(probs_csv_path)
    logger.info("Generated cell type proportions saved to disk.")
    logger.info("Run executed successfully, exiting with code 0. Goodbye!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="The directory of the run to process.")
    parser.add_argument("--columns_to_mask", type=str, default='["il7_[ng_ml]", "mcsf_[ng_ml]", "ly_cocktail_[ul/well]"]', help="The names of the protocol columns to mask for the counterfactual experiment.")
    parser.add_argument("--num_time_steps", type=int, default=1_000, help="The number of time steps used to query the forward model.")
    parser.add_argument("--solver_kwargs", type=str, default='{"method":"euler", "atol": 5e-5, "rtol": 5e-5}', help="The solver key-word arguments used to query the forward model.")
    parser.add_argument("--num_samples", type=int, default=20_000, help="The number of samples to generate with the forward model.")
    parser.add_argument("--cell_type_column", type=str, default="cell_type_reannot_final", help="The identifier of the cell type column used by the forward model.")
    args = parser.parse_args()

    main(args)