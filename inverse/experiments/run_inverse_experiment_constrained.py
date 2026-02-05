from datetime import datetime
import logging
import os
import sys
import traceback
import uuid

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from sc_flow.constants import DataFields, PredictionFields
from sc_exp_design.models import FlowMatching
from sc_exp_design.utils import set_reproducibility
from sc_exp_design.inverse import ImpliciDualGuidedFlow

logger = logging.getLogger(__name__)


ROOT_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"
NON_LINEARITIES_REGISTRY = {
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU
}


def constrain_fn_factory(config, device):
    def _linear_bound(x1, bound, mask, upper=True):
        x1 = x1[..., mask]
        bound = bound[..., mask]
        return x1 - bound if upper else bound - x1
    ubound = torch.tensor(config.constraints.ubound).float().to(device)
    lbound = torch.tensor(config.constraints.lbound).float().to(device)
    mask = config.constraints.constraint_mask
    return [
        lambda x1: _linear_bound(x1, ubound, mask, upper=True),
        lambda x1: _linear_bound(x1, lbound, mask, upper=False),
    ]


def loss_fn_factory(
    loss_fn,
    config,
    noise,
    optimal_condition,
    non_linearity,
    forward_model,
):
    

    # sampling fixed noise for the forward model
    if config.loss_guidance.fix_noise:
        noise = forward_model.forward_model.noise_distribution(
            (N, config.loss_guidance.num_forward_pass_per_sample, config.loss_guidance.forward_model.forward_model.velocity_field.config.flow_dim)
        ).squeeze(dim=0).to(config.loss_guidance.forward_model.forward_model.device)
    else:
        noise = None

    def compute_target_loss(target_pred_dict, optimal_condition):
        """"""
        if config.fix_noise:
            return torch.sum(
                torch.stack(
                    [(loss_fn(target_pred_dict[covariate], optimal_condition[covariate].to(target_pred_dict[covariate].device)).mean(1)) 
                    for covariate, loss_fn in loss_fn.items()],
                    dim = 0)
                , dim=0)
        else:
            return torch.sum(
                torch.stack(
                    [loss_fn(target_pred_dict[covariate], optimal_condition[covariate].to(target_pred_dict[covariate].device)) 
                    for covariate, loss_fn in loss_fn.items()],
                    dim = 0)
                , dim=0)

    # function for computing loss from x_t
    def _compute_loss(x1):
        if non_linearity is not None:
            x1 = non_linearity(x1)
        batch_dict = {}
        x1_fwd = x1
        if noise is not None:
            batch_dict[DataFields.SOURCE_STATE] = noise
            if config.loss_guidance.fix_noise:
                x1_fwd = x1.unsqueeze(1).repeat(1, config.loss_guidance.num_forward_pass_per_sample, 1)
        
        condition_repr = next(iter(forward_model.forward_model.train_data.data.perturbation_covariates))
        batch_dict[DataFields.PERTURBATION_DATA] = {
            condition_repr: x1_fwd
        }
        forward_out = forward_model.predict(
            batch_dict,
            no_grad=False,
            fix_noise=config.loss_guidance.fix_noise,
            num_time_steps=config.loss_guidance.n_time_steps_forward_model,
            solver_kwargs=config.loss_guidance.solver_kwargs_forward_model
        )
        pred = forward_out[PredictionFields.TARGET_PREDICTION_DATA]
        phen_loss = compute_target_loss(pred, optimal_condition)
        return phen_loss
    return _compute_loss, noise


@hydra.main(
    config_path=os.path.join(ROOT_DIR, "inverse/config"),
    config_name="run_inverse_constrained"
)
def main(config: DictConfig) -> float:

    # Import modules
    sys.path.insert(0, os.path.join(ROOT_DIR, "model_utils"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "scripts"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "inverse/inverse_utils"))
    sys.path.insert(0, os.path.join(ROOT_DIR, "inverse/experiments"))
    from lambda_schedulers import schedulers_dict
    from loss_guidance import LossGuidedFlow
    from train_utils import resolve_omegaconf_to_dictionary
    from experiment_utils import (
        create_dir,
        get_forward_model,
        query_forward_model,
        flatten_conf,
        get_transformed_data,
        get_target_dict,
        get_loss_fn,
    )
    from data_utils import get_protocol_tranformations
    from plot_utils import plot_loss_history, plot_heatmap

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
    classes = ct_le.classes_.tolist()

    # Get prior flow
    logger.info(f"Loading Prior Flow Model from {config.paths.prior_flow_path}...")
    prior_flow = FlowMatching.load(config.paths.prior_flow_path)
    logger.info(f"Prior Flow ready!\n{prior_flow.velocity_field}")

    # Define loss function
    logger.info(f"Preparing loss function (Cross-Entropy)...")
    loss_fns = get_loss_fn(config)
    logger.info(f"Loss function ready!\n{loss_fns}")

    # Define target for current cell type
    logger.info(f"Preparing target value for cell type {config.sampling.target_cell_type}...")
    target = get_target_dict(
        config,
        classes,
        forward_model.forward_model.device
    )
    logger.info(f"Target ready!\n{target}")

    # Initialize non linearity
    logger.info(f"Initializing non linearity {config.non_linearity.non_linearity_id}...")
    non_linearity_class = NON_LINEARITIES_REGISTRY.get(config.non_linearity.non_linearity_id, None)
    if non_linearity_class is None:
        msg = f"Non linearity {config.non_linearity.non_linearity_id} not valid"
        raise ValueError(msg)
    non_linearity = non_linearity_class(**resolve_omegaconf_to_dictionary(config.non_linearity.non_linearity_kwargs))
    logger.info(f"Non linearity initialized!\n{non_linearity}")

    # compile loss function
    logger.info(f"Compiling final loss function for current cell type...")
    compute_loss, noise = loss_fn_factory(
        loss_fns,
        config,
        target,
        non_linearity,
        forward_model,
    )
    logger.info(f"Loss function compiled!")

    # compile constraints
    logger.info(f"Compiling constraints for current experiment...")
    compute_constraints = constrain_fn_factory(config, forward_model.forward_model.device)
    logger.info(f"Constraints compiled!")

    # inizialize dual flow
    logger.info(f"Initializing implicit guided flow...")
    guided_flow = ImpliciDualGuidedFlow(prior_flow)
    torch.cuda.empty_cache()
    logger.info(f"Implicit dual guided flow initialized {guided_flow}")

    # Prepare lambda_scheduler
    logger.info(f"Initializing guidance strength lambda_scheduler {config.lambda_scheduler.scheduler_id} with {config.lambda_scheduler.scheduler_kwargs}")
    scheduler_cls = schedulers_dict.get(
        config.lambda_scheduler.scheduler_id,
        None,
    )
    if scheduler_cls is None:
        msg = f"lambda_scheduler {config.lambda_scheduler.scheduler_id} not valid"
        raise ValueError(msg)
    lambda_scheduler = scheduler_cls(**resolve_omegaconf_to_dictionary(config.lambda_scheduler.scheduler_kwargs))
    logger.info(f"lambda_scheduler Ready!\n{lambda_scheduler}")

    # prepare c scheduler
    c_scheduler = ...

    # sampling from guided flow
    torch.cuda.empty_cache()
    trajectory, loss_history, lambda_history, noise = guided_flow.sample_posterior(
        config.sampling.N,
        compute_loss,
        compute_constraints,
        lambda_scheduler=lambda_scheduler,
        c_scheduler=c_scheduler,
        num_time_steps=config.sampling.num_time_steps,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.sampling.solver_kwargs),
        eps=config.constraints.eps,
        use_lstsq=config.constraints.use_lstsq,
        g_tol=config.constraints.g_tol,
    )
    torch.cuda.empty_cache()
    logger.info(f"Inverse model queried! {trajectory.shape=}, {loss_history.shape=}, {lambda_history.shape=}, {noise.shape=}")

    # Query forward model
    solver_kwargs = resolve_omegaconf_to_dictionary(config.forward_model.solver_kwargs)
    logger.info(f"Querying forward model with samples from the inverse model...")
    logger.info(f"\tforward_model.num_time_steps={config.forward_model.num_time_steps}")
    logger.info(f"\tforward_model.solver_kwargs={solver_kwargs}")
    fwd_query_res_dict = query_forward_model(
        trajectory,
        noise,
        forward_model,
        config.forward_model.num_time_steps,
        solver_kwargs,
        ct_le,
        logger=logger
    )
    torch.cuda.empty_cache()

    # Define paths directories
    ct_string = config.sampling.target_cell_type
    ct_safe_string = ct_string.replace("/", ":") # cell type dir
    ct_dir = os.path.join(config.paths.dump_dir, ct_safe_string) # cell type dir
    run_dir = os.path.join(ct_dir, run_id) # run dir
    plots_dir = os.path.join(run_dir, "plots") # plots dir
    config_path = os.path.join(run_dir, "config.yaml")
    inverse_results_path = os.path.join(run_dir, "inverse_results.npz")
    fwd_results_path = os.path.join(run_dir, "fwd_results.npz")
    candidates_path = os.path.join(run_dir, "candidates.csv")
    loss_history_plot_path = os.path.join(plots_dir, "loss_history.svg")
    logger.info(
        f"Creating dump directories for: \n"
        f"\t Note: Cell type indentifier changed from \"{ct_string}\" to {ct_safe_string}.\n"
        f"\t Dump directory for cell type will be created at {ct_dir}.\n"
        f"\t Dump directory for run will be created at {run_dir}.\n"
        f"\t Dump directory for run plots will be created at {plots_dir}.\n"
        f"\t Configuration will be dumped at {config_path}.\n"
        f"\t Raw optimization data will be dumped at {inverse_results_path}.\n"
        f"\t Raw forward data will be dumped at {fwd_results_path}.\n"
        f"\t Post Processed run data will be dumped at {candidates_path}.\n"
    )

    # Create directories
    create_dir(config.paths.dump_dir, logger=logger) # base dir
    create_dir(ct_dir, logger=logger) # cell type dir
    create_dir(run_dir, logger=logger) # run dir
    create_dir(plots_dir, logger=logger) # plots dir
    logger.info("All the directories have been successfully created!")

    # Save corresponding configuration
    logger.info("Saving configurations...")
    config_container = OmegaConf.to_container(config, resolve=True)
    fconfig_dict = flatten_conf(config_container)
    OmegaConf.save(config=config_container, f=config_path)
    logger.info(f"Configuration saved!")

    # Save inverse results
    logger.info("Saving raw inverse run data...")
    inverse_res_dict = {
        "trajectory": trajectory,
        "loss_history": loss_history,
        "lambda_history": lambda_history,
        "noise": noise.detach().cpu().numpy()
    }
    np.savez(inverse_results_path, **inverse_res_dict)
    logger.info(f"Inverse model results saved!")

    # Save forward model results
    logger.info("Saving raw forward query data...")
    np.savez(fwd_results_path, **fwd_query_res_dict)
    logger.info(
        f"Forward results saved! \n"
        "All raw optimization data and associated configurations saved. \n"
        "Post-processing the results."
    )

    # Retrieve samples and induced phenotype
    logger.info("Retrieving final results...")
    per_cell_ct_props = fwd_query_res_dict["ct_probs"]
    samples = trajectory[:, -1, :]
    logger.info(
        f"* Found phenotype data of shape {per_cell_ct_props.shape}, " 
        "aggregating over dimension 1.\n"
        f"* Found samples of shape {samples.shape}. "
        "Setting negative values to 0.\n"
        f"* Found loss history of shape {loss_history.shape=}, "
        "retrievig terminal value at index -1 over dimension 1."
    )
    ct_props = fwd_query_res_dict["ct_probs"].mean(1)
    samples = np.maximum(samples, 0)
    terminal_loss = loss_history[:, -1]
    logger.info(
        "* Post-Processed data of shape:\n"
        f"\t -> {ct_props.shape=}\n"
        f"\t -> {samples.shape=}\n"
        f"\t -> {terminal_loss.shape=}"
    )

    # Get inverse transformations to rescale the samples
    logger.info(
        "Retrieving transformation for medium covariates...\n"
        fr"\t -> protocol_columns={config.annotation.protocol_columns}\n"
        fr"\t -> $\log(1 + x)$ exp_cols={config.annotation.log1p_exp_cols}\n"
        fr"\t -> $\log_2(1 + x)$ exp_cols={config.annotation.log21p_exp_cols}\n"
        fr"\t -> inverse={True}"
    )
    column2tranform = get_protocol_tranformations(
        config.annotation.protocol_columns,
        log1p_exp_cols=config.annotation.log1p_exp_cols,
        log21p_exp_cols=config.annotation.log21p_exp_cols,
        inverse=True,
    )

    # Prepare data dictionary
    logger.info(
        f"Tranformations for medium data dictionaries ready, applying them!\n\t{column2tranform}"
    )
    data_dict_transformed, data_dict_original = get_transformed_data(
        samples,
        config.annotation.protocol_columns,
        column2tranform,
    )
    logger.info("Medium data dictionaries ready!")

    # Create data frame
    logger.info(f"Creating pd.DataFrame to store the post-processed results.")
    samples_data_dict = {
        **data_dict_transformed,
        **{f"{k}:rescaled":v for k, v in data_dict_original.items()},
        "loss": terminal_loss,
        **{
            f"{ct}_prop": ct_props[:, idx] for idx, ct in enumerate(classes)
        },
    }
    samples_df = pd.DataFrame(samples_data_dict)

    # append configurations to dataframe
    for key, val in fconfig_dict.items():
        col_name = f"cfg:{key}"
        samples_df[col_name] = [val]*len(samples_df)

    # append paths
    samples_df["configuration path"] = config_path
    samples_df["inverse_results_path"] = inverse_results_path
    samples_df["fwd_results_path"] = fwd_results_path
    
    # handle index
    samples_df.index = samples_df.index.map(lambda e: f"{ct_string}:{run_id}:{e}")
    samples_df.index.name = "sample_id"
    logger.info(
        f"Post-processed data frame ready:\n"
        f"shape={samples_df.shape}\n"
        f"columns={samples_df.columns}\n"
    )

    # dump csv
    samples_df.to_csv(candidates_path)
    logger.info("Post-processed data framed dumped!")

    # plot loss history
    logger.info("Plotting loss history...")
    loss_history_fig = plot_loss_history(ct_string, loss_history)
    loss_history_fig.savefig(
        loss_history_plot_path,
        dpi=300,
    )
    plt.close(loss_history_fig)
    logger.info(f"Plot written to disk!")

    # heatmap samples
    logger.info("Plotting heatmap of sampled solution...")
    plot_heatmap(
        plots_dir,
        classes,
        ct_string,
        fwd_query_res_dict,
        inverse_res_dict,
        config_container["annotation"],
        samples_vmin=0.0,
        samples_vmax=10.0,
        loss_vmin=0.0,
        loss_vmax=10.0,
        plot_pheno=False
    )
    logger.info("Plot written to disk!")

    # heatmap phenotype
    logger.info("Plotting heatmap of induced phenotype...")
    plot_heatmap(
        plots_dir,
        classes,
        ct_string,
        fwd_query_res_dict,
        inverse_res_dict,
        config_container["annotation"],
        samples_vmin=0.0,
        samples_vmax=10.0,
        loss_vmin=0.0,
        loss_vmax=10.0,
        plot_pheno=True
    )
    logger.info("Plot written to disk!")
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
