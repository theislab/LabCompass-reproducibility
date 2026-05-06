from datetime import datetime
from functools import partial
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

from sc_exp_design.models import FlowMatching, FlowMatchingWithScore, FlowMap
from sc_exp_design.utils import set_reproducibility
from sc_exp_design.inverse import LossGuidedFlow

# 1. Configure the logging behavior
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensures it goes to console
    ]
)
logger = logging.getLogger(__name__)


ROOT_DIR = "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC"
NON_LINEARITIES_REGISTRY = {
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU
}
AGG_FN_REGISTRY = {
    "mean": np.mean,
    "median": np.median,
    "min": np.min,
    "max": np.max,
    "pop": lambda x: x,
}

@hydra.main(
    config_path="/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/loss_guidance/config",
    config_name="run_inverse_constrained"
)
def main(config: DictConfig) -> float:

    # Import modules
    sys.path.insert(0, os.path.join(ROOT_DIR, "shared_utils"))
    from lambda_schedulers import schedulers_dict
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
    from inverse_utils import loss_fn_factory, constraint_fn_factory, linear_scheduler_with_warmup

    # Create run id 
    run_id = uuid.uuid4().hex[:8]
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{ts}_{run_id}"
    logger.info(f"Starting inverse run {run_id}...")

    # Set reproducibility
    logger.info(f"Reproducibility set to {config.run.random_seed}")
    set_reproducibility(config.run.random_seed)

    # load forward models
    forward_model, (
        phi_model,
        target_prediction_model
    ) = get_forward_model(config, logger=logger)
    forward_model.target_prediction_model.resc_model["model"].eval()
    forward_model.forward_model.velocity_field.eval()

    # initialize prior models
    prior_cfm = sc_exp_design.models.FlowMatchingWithScore.load(
        config.paths.prior_flow_path
    )
    prior_fm = sc_exp_design.models.FlowMap.load(
        config.paths.prior_flow_map_path
    )
    prior_cfm.velocity_field.eval()
    prior_fm.flow_map.eval()

    # data for classifier model (subset)
    train_adata_g = target_prediction_model.train_data.adata
    val_adata_g = target_prediction_model.validation_data.adata
    adata_g = sc.concat((train_adata_g, val_adata_g), uns_merge="same")
    logger.info(f"Subset data shape: {adata_g.shape}")

    # Prepare label encoder
    ct_le = LabelEncoder()
    ct_values = target_prediction_model.train_data.adata.obs["cell_type"].values
    ct_le.fit(ct_values)
    classes = ct_le.classes_.tolist()

    # get target data and define optimal features
    agg_fn = AGG_FN_REGISTRY[config.loss.agg_id]
    target_adata, query_adata = get_target_adata_marker_opt(
        train_adata_phi,
        adata_g,
        agg_fn,
        config.loss.target_marker_names,
        config.loss.target_morph_feat_names,
        config.annotation.scatter_columns,
        config.loss.filter_dict,
        config.loss.target_quantile,
        agg_fn_kwargs=config.loss.agg_fn_kwargs,
    )
    target_feats_mask = query_adata.uns["target_feats_mask"]
    xstar = torch.from_numpy(target_adata.obsm[cell_state_rep]).\
        to(torch.float32).to(prior_cfm.device)

    # prepare loss function and noise
    loss_fn_comp, noise = loss_fn_factory_marker_opt(
        config, loss_fn, xstar, non_linearity, phi_model,
        target_feats_mask, config.loss.agg_type
    )

    # prepare constraints and scheduler
    compute_constraints = constraint_fn_factory(config, forward_model.forward_model.device, get_protocol_tranformations)
    c_scheduler = partial(
        linear_scheduler_with_warmup,
        t_warmup=config.constraints.c_scheduler_kwargs.t_warmup,
        vmin=config.constraints.c_scheduler_kwargs.vmin,
        vmax=config.constraints.c_scheduler_kwargs.vmax,
    )

    # inizialize dual flow
    logger.info(f"Initializing implicit guided flow...")
    guided_flow = LossGuidedFlow(prior_flow, prior_flow_map=prior_fm)
    torch.cuda.empty_cache()
    logger.info(f"Implicit dual guided flow initialized {guided_flow}")

    # Prepare lambda_scheduler
    logger.info(f"Initializing guidance strength lambda_scheduler {config.scheduler.scheduler_id} with {config.scheduler.scheduler_kwargs}")
    scheduler_cls = schedulers_dict.get(
        config.scheduler.scheduler_id,
        None,
    )
    if scheduler_cls is None:
        msg = f"lambda_scheduler {config.scheduler.scheduler_id} not valid"
        raise ValueError(msg)
    lambda_scheduler_class = scheduler_cls(**resolve_omegaconf_to_dictionary(config.scheduler.scheduler_kwargs))
    lambda_scheduler = lambda t: lambda_scheduler_class.compute_lambda_t(t)
    logger.info(f"lambda_scheduler Ready!\n{lambda_scheduler}")

    # sampling from guided flow
    logger.info(
        "Sampling from guided flow with configurations:\n"
        f"N={config.sampling.N}\n"
        f"num_time_steps={config.sampling.num_time_steps}\n"
        f"solver_kwargs={config.sampling.solver_kwargs}\n"
        f"sde_sampling={config.sampling.sde_sampling}\n"
    )
    torch.cuda.empty_cache()
    trajectory, loss_history, lambda_history = guided_flow.sample_posterior(
        config.sampling.N,
        compute_loss,
        reg_fn_lists=compute_constraints,
        lambda_scheduler=lambda_scheduler,
        c_scheduler=c_scheduler,
        num_time_steps=config.sampling.num_time_steps,
        solver_kwargs=resolve_omegaconf_to_dictionary(config.sampling.solver_kwargs),
    )
    # moving results to numpy
    trajectory = np.permute_dims(trajectory, (1, 0, 2))
    loss_history = loss_history.T
    lambda_history = lambda_history.T
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
        "trajectory": trajectory if isinstance(trajectory, np.ndarray) else trajectory.detach().cpu().numpy(),
        "loss_history": loss_history if isinstance(loss_history, np.ndarray) else loss_history.detach().cpu().numpy(),
        "lambda_history": lambda_history if isinstance(lambda_history, np.ndarray) else lambda_history.detach().cpu().numpy(),
        "noise": noise if isinstance(noise, np.ndarray) else noise.detach().cpu().numpy()
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
    samples_df = pd.DataFrame({
        k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in samples_data_dict.items()
    })

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
    loss_history_fig = plot_loss_history(ct_string, loss_history.detach().cpu().numpy())
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
        {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in fwd_query_res_dict.items()},
        {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in inverse_res_dict.items()},
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
        {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in fwd_query_res_dict.items()},
        {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in inverse_res_dict.items()},
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
