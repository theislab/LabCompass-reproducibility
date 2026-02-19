from datetime import datetime
import logging
import os
import sys
import uuid
import traceback

import cloudpickle
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder
import torch

from sc_exp_design.data.container import DataMixin
from sc_exp_design.utils import set_reproducibility


logger = logging.getLogger(__name__)


@hydra.main(
    "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/forward/conditional_model/flow_matching/config",
    config_name="load_models",
)
def main(config):
    # import libraries
    sys.path.insert(0, f"/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/shared_utils")
    from data_utils import (
        get_condition_data_from_file,
    )
    from experiment_utils import (
        create_dir,
        get_forward_model,
        generate_with_condition,
    )
    from z_norm_modules import get_rescaling

    # Create run id 
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_id}"
    logger.info(f"Starting inverse run {run_id}...")

    # Set reproducibility
    logger.info(f"Reproducibility set to {config.reproducibility.seed}")
    set_reproducibility(config.reproducibility.seed)

    # Get forward model
    logger.info(f"Preparing forward model...")
    forward_model, (
        phi_model,
        target_prediction_model
    ) = get_forward_model(config, logger=logger)
    forward_model.target_prediction_model.resc_model["model"].eval()
    forward_model.forward_model.velocity_field.eval()
    logger.info(f"Forward model ready!\n{forward_model}")

    # Prepare label encoder
    logger.info(f"Fitting label encoder on cell type...")
    ct_le = LabelEncoder()
    ct_values = target_prediction_model.train_data.adata.obs["cell_type"].values
    ct_le.fit(ct_values)

    # prepare data
    logger.info(f"Preparing data...")
    train_adata_g = target_prediction_model.train_data.adata
    val_adata_g = target_prediction_model.validation_data.adata
    train_adata_phi = phi_model.train_data.adata

    # get rescalings
    zn_phi = get_rescaling(
        train_adata_phi, inverse=False
    )
    izn_g = get_rescaling(
        train_adata_g, inverse=True
    )

    # prepare obs
    X_train_g_model = train_adata_g.obsm[config.data.sample_rep]
    X_val_g_model = val_adata_g.obsm[config.data.sample_rep]

    # applying transformations to data
    X_train_g = izn_g(torch.from_numpy(X_train_g_model).cuda())
    X_train_g = zn_phi(X_train_g).detach().cpu().numpy()
    X_val_g = izn_g(torch.from_numpy(X_val_g_model).cuda())
    X_val_g = zn_phi(X_val_g).detach().cpu().numpy()

    # preparing protocol data
    protocol_data, condition_adata = get_condition_data_from_file(
        phi_model.data_manager,
        config.paths.condition_metadata_path,
        config.annotation.protocol_columns,
        config.annotation.log1p_exp_cols,
        config.annotation.log21p_exp_cols,
        config.annotation.protocol_obs_key_added,
        config.annotation.protocol_sep,
        config.annotation.one_hot_uns_key_added,
        config.annotation.protocol_obsm_key,
    )
    protocol_dict = DataMixin(protocol_data)
    samples = next(iter(protocol_dict.values()))

    # Define paths directories
    run_dir = os.path.join(config.paths.dump_dir, run_id) # cell type dir
    config_path = os.path.join(run_dir, "config.yaml")
    fwd_results_path = os.path.join(run_dir, "fwd_results.npz")
    logger.info(
        f"Creating dump directories for: \n"
        f"\t Dump directory for run will be created at {run_dir}.\n"
        f"\t Configuration will be dumped at {config_path}.\n"
        f"\t Raw forward data will be dumped at {fwd_results_path}.\n"
    )

    # Create directories
    create_dir(config.paths.dump_dir, logger=logger) # base dir
    create_dir(run_dir, logger=logger) # run dir
    logger.info("All the directories have been successfully created!")

    # generate with conditions
    fwd_query_res_dict = generate_with_condition(
        samples,
        forward_model,
        config.sampling.num_time_steps,
        config.sampling.solver_kwargs,
        ct_le,
        num_samples=config.sampling.N,
        logger=logger,
        n_scatter_feats=6,
    )

    # Save corresponding configuration
    logger.info("Saving configurations...")
    config_container = OmegaConf.to_container(config, resolve=True)
    OmegaConf.save(config=config_container, f=config_path)
    logger.info(f"Configuration saved!")

    # Save forward model results
    logger.info("Saving raw forward query data and protocol metadata...")
    np.savez(fwd_results_path, **fwd_query_res_dict)
    logger.info(
        f"Forward results saved! \n"
        "Raw query data and associated configurations saved. \n"
        "Post-processing the results."
    )
    logger.info("Exit code 0, goodbye!")
    return 0.0


if __name__ == "__main__":

    # running the experiment
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)