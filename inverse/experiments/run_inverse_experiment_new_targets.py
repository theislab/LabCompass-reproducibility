from datetime import datetime
import logging
import os
import sys
import traceback
from typing import Any
import uuid

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
import torch

from sc_exp_design.constants import DataFields, ParamsFields, PredictionFields
from sc_exp_design.models import FlowMatching, TargetPredictionModel
from sc_exp_design.utils import set_reproducibility


logger = logging.getLogger(__name__)


non_linearities_dict = {
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU
}


TARGET_PROPS_DICTIONARY = {
    "HSCs": [0.07, 0.07, 0.00, 0.00, 0.07, 0.01, 0.07, 0.07, 0.00, 0.07, 0.07, 0.00, 0.21, 0.07, 0.07, 0.07, 0.00, 0.01, 0.07],
    "MgkPro": [0.05, 0.05, 0.00, 0.00, 0.05, 0.01, 0.01, 0.05, 0.00, 0.05, 0.05, 0.00, 0.05, 0.05, 0.05, 0.05, 0.42, 0.01, 0.05],
    "EryPro": [0.05, 0.05, 0.00, 0.00, 0.05, 0.01, 0.01, 0.05, 0.42, 0.05, 0.05, 0.00, 0.05, 0.05, 0.05, 0.05, 0.0, 0.01, 0.05],
}
MASK_DICT = {
    "HSCs": [False, False, True, True, False, False, False, False, True, False, False, True, True, False, False, False, True, False, False],
    "MgkPro": [False, False, True, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False],
    "EryPro": [False, False, True, True, False, False, False, False, True, False, False, True, False, False, False, False, True, False, False],
}

@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/config",
    config_name="run_inverse"
)
def main(config: DictConfig) -> float:

    # Import modules
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/model_utils")
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/scripts")
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/inverse_utils")
    sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/experiments")
    from lambda_schedulers import schedulers_dict
    from loss_guidance import LossGuidedFlow
    from train_utils import resolve_omegaconf_to_dictionary
    from experiment_utils import (
        create_dir,
        get_forward_model,
        query_forward_model,
    )

    # Create run id 
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_id}"
    logger.info(f"Starting inverse run {run_id}...")

    # Set reproducibility
    logger.info(f"Reproducibility set to {config.run.random_seed}")
    set_reproducibility(config.run.random_seed)

    # Get forward model
    logger.info(f"Preparing forward model...")
    forward_model, (
        _,
        target_prediction_model
    ) = get_forward_model(config, logger=logger)
    logger.info(f"Forward model ready!\n{forward_model}")

    # Prepare label encoder
    ct_le = LabelEncoder()
    ct_values = target_prediction_model.train_data.adata.obs["cell_type"].values
    ct_le.fit(ct_values)

    # Get prior flow
    logger.info(f"Loading Prior Flow Model from {config.paths.prior_flow_path}...")
    prior_flow = FlowMatching.load(config.paths.prior_flow_path)
    logger.info(f"Prior Flow ready!\n{prior_flow}")

    # Define target for current cell type
    logger.info(f"Preparing target value for cell type {config.sampling.target_cell_type}...")
    prop = torch.tensor(
        TARGET_PROPS_DICTIONARY[config.sampling.target_cell_type]
    ).float().to(forward_model.forward_model.device)
    mask = MASK_DICT[config.sampling.target_cell_type]
    target = {
        "cell_type": prop.unsqueeze(0)
    }
    logger.info(f"Target ready!\n{target}")

    # Define loss function
    logger.info(f"Preparing loss function (Cross-Entropy)...")
    if config.sampling.mask_gradients:
        logger.info(f"Using masked objective")
        loss_fns = {
            "cell_type": lambda pred, target: -torch.sum(target[..., mask]*torch.nn.functional.log_softmax(pred[..., mask], dim=-1), dim=-1)
        }
    else:
        loss_fns = {
            "cell_type": lambda pred, target: -torch.sum(target*torch.nn.functional.log_softmax(pred, dim=-1), dim=-1)
        }
    logger.info(f"Loss function ready!\n{loss_fns}")

    # Initialize guided flow
    logger.info(
        "Initializing Guided flow with configurations:\n"
        f"\t fix_noise={config.loss_guidance.fix_noise}\n"
        f"\t num_forward_pass_per_sample={config.loss_guidance.num_forward_pass_per_sample}\n"
        f"\t regularization={config.loss_guidance.regularization}\n"
        f"\t reg_strength={config.loss_guidance.reg_strength}\n"
    )
    guided_flow =  LossGuidedFlow(
        prior_flow,
        forward_model,
        loss_fns,
        fix_noise=config.loss_guidance.fix_noise,
        num_forward_pass_per_sample=config.loss_guidance.num_forward_pass_per_sample,
        regularization=config.loss_guidance.regularization,
        reg_strength=config.loss_guidance.reg_strength,
        n_time_steps_forward_model=config.loss_guidance.n_time_steps_forward_model,
        solver_kwargs_forward_model=resolve_omegaconf_to_dictionary(config.loss_guidance.solver_kwargs_forward_model)
    )
    torch.cuda.empty_cache()
    logger.info(f"Guided Flow Ready!\n{guided_flow}")

    # Initialize non linearity
    logger.info(f"Initializing non linearity {config.non_linearity.non_linearity_id}...")
    non_linearity_class = non_linearities_dict.get(config.non_linearity.non_linearity_id, None)
    if non_linearity_class is None:
        msg = f"Non linearity {config.non_linearity.non_linearity_id} not valid"
        raise ValueError(msg)
    non_linearity = non_linearity_class(**resolve_omegaconf_to_dictionary(config.non_linearity.non_linearity_kwargs))
    logger.info(f"Non linearity initialized!\n{non_linearity}")

    # Prepare scheduler
    logger.info(f"Initializing guidance strength scheduler {config.scheduler.scheduler_id} with {config.scheduler.scheduler_kwargs}")
    scheduler_cls = schedulers_dict.get(
        config.scheduler.scheduler_id,
        None,
    )
    if scheduler_cls is None:
        msg = f"Scheduler {config.scheduler.scheduler_id} not valid"
        raise ValueError(msg)
    scheduler = scheduler_cls(**resolve_omegaconf_to_dictionary(config.scheduler.scheduler_kwargs))
    logger.info(f"Scheduler Ready!\n{scheduler}")

    # Sample from guided flow
    logger.info(
        "Sampling from guided flow with configurations:\n"
        f"N={config.sampling.N}\n"
        f"num_time_steps={config.sampling.num_time_steps}\n"
        f"solver_kwargs={config.sampling.solver_kwargs}\n"
        f"sde_sampling={config.sampling.sde_sampling}\n"
    )
    torch.cuda.empty_cache()
    trajectory, loss_history, lambda_history, noise = guided_flow.sample_posterior(
        config.sampling.N,
        target,
        num_time_steps=config.sampling.num_time_steps,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.sampling.solver_kwargs),
        lambda_scheduler=scheduler,
        non_linearity=non_linearity,
        sde_sampling=config.sampling.sde_sampling
    )
    torch.cuda.empty_cache()
    logger.info(f"Inverse model queried! {trajectory.shape=}, {loss_history.shape=}, {lambda_history.shape=}, {noise.shape=}")

    # Query forward model
    fwd_query_res_dict = query_forward_model(
        trajectory,
        noise,
        forward_model,
        config.forward_model.num_time_steps,
        resolve_omegaconf_to_dictionary(config.forward_model.solver_kwargs),
        ct_le,
        logger=logger
    )
    torch.cuda.empty_cache()

    # Prepare directories
    dump_dir = config.paths.dump_dir
    if config.sampling.mask_gradients:
        dump_dir = dump_dir + "-masked"        
    create_dir(dump_dir, logger=logger) # base dir
    ct = config.sampling.target_cell_type.replace("/", "_") # cell type dir
    ct_dir = os.path.join(dump_dir, ct) # cell type dir
    create_dir(ct_dir, logger=logger) # cell type dir
    run_dir = os.path.join(ct_dir, run_id) # run dir
    create_dir(run_dir, logger=logger) # run dir

    # Save corresponding configuration
    config_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(config=OmegaConf.to_container(config, resolve=True), f=config_path)
    logger.info(f"Configuration saved to {config_path}")

    # Save inverse results
    inverse_results_path = os.path.join(run_dir, "inverse_results.npz")
    logger.info(f"Saving inverse model results at {inverse_results_path}...")
    inverse_results_data = {
        "trajectory": trajectory,
        "loss_history": loss_history,
        "lambda_history": lambda_history,
        "noise": noise.detach().cpu().numpy()
    }
    np.savez(inverse_results_path, **inverse_results_data)
    logger.info(f"Inverse model results saved!")

    # Save forward model results
    fwd_results_path = os.path.join(run_dir, "fwd_results.npz")
    logger.info(f"Saving forward model results at {fwd_results_path}...")
    np.savez(fwd_results_path, **fwd_query_res_dict)
    logger.info(f"Forward results saved!")
    return 0.0

if __name__ == "__main__":

    # running the experiment
    try:
        main()
    except Exception as e:
        logger.info(f"An error occurred: {e}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
