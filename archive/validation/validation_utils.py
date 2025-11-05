import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
import torch
from tqdm import tqdm

from sc_exp_design.constants import DataFields, PredictionFields
from sc_exp_design.data.container import DataMixin

logger = logging.getLogger(__name__)


def predict_with_groups(
    N,
    flow_matching,
    ood_data,
    num_time_steps,
    solver_kwargs,
):
    results_dict = {}
    # iterating over unique combinations
    for comb in tqdm(ood_data.seen_combinations):
        # retrieving data 
        comb_idxs = np.all((ood_data.adata.obs.loc[:, flow_matching.data_manager.perturbations] == comb).values, axis=-1)
        if not np.any(comb_idxs):
            logger.warning(f"No cells found for combination {comb}, skipping.")
            continue
        comb_data = ood_data[comb_idxs]

        # generating samples
        comb_perts = DataMixin(comb_data.perturbation_data)
        comb_perts = comb_perts.apply(lambda x: np.unique(x, axis=0))
        comb_perts = comb_perts.apply(lambda x: torch.from_numpy(x).float().to(flow_matching.device).repeat(N, 1))
        preds = flow_matching.predict(
            {
                DataFields.PERTURBATION_DATA: comb_perts,
            },
            num_time_steps=num_time_steps,
            solver_kwargs=solver_kwargs,
        ).detach().cpu().numpy()

        # storing results
        results_dict[comb] = {
            PredictionFields.PREDICTION_DATA: preds,
            DataFields.TARGET_STATE: comb_data.state_data
        }
    return results_dict


def predict_paired_settings(
    N,
    flow_matching,
    ood_data,
    num_time_steps,
    solver_kwargs,
):

    # retrieving unique perturbations
    perturbation_data = ood_data.perturbation_data
    perturbation_data = DataMixin(ood_data.perturbation_data)
    unique_perts = perturbation_data.apply(lambda x: np.unique(x, axis=0))
    logger.info("unique_perts ", next(iter(unique_perts.values())).shape[0])

    # iterating over unique perturbations
    results_dict = {}
    for idx in tqdm(range(next(iter(unique_perts.values())).shape[0])):
        # retrieving mask for current combination
        comb_perts = unique_perts.apply(lambda x: x[idx, :])
        comb_idxs = np.all(next(iter(perturbation_data.values())) == next(iter(comb_perts.values())), axis=-1)

        # slicing the array and predicting
        comb_data = ood_data[comb_idxs]
        comb_perts = comb_perts.apply(lambda x: torch.from_numpy(x[None]).float().to(flow_matching.device).repeat(N, 1))
        preds = flow_matching.predict(
            {
                DataFields.PERTURBATION_DATA: comb_perts,
            },
            num_time_steps=num_time_steps,
            solver_kwargs=solver_kwargs,
        ).detach().cpu().numpy()

        # storing the results
        results_dict[idx] = {
            PredictionFields.PREDICTION_DATA: preds,
            DataFields.TARGET_STATE: comb_data.state_data
        }
    return results_dict


def predict_on_ood_data(
    N,
    flow_matching,
    ood_data,
    num_time_steps,
    solver_kwargs,
):

    assert ood_data.perturbation_data is not None

    if ood_data.seen_combinations is not None:
        logger.info("Validating with seen combinations.")
        return predict_with_groups(
            N,
            flow_matching,
            ood_data,
            num_time_steps,
            solver_kwargs,
        )

    # paired setting TODO
    else:
        logger.info("Validating in the paired setting.")
        return predict_paired_settings(
            N,
            flow_matching,
            ood_data,
            num_time_steps,
            solver_kwargs,
        )


def validate_on_ood_data(
    N,
    flow_matching,
    ood_data,
    metrics_callback,
    num_time_steps=100,
    solver_kwargs=None,
):
    solver_kwargs = {} if solver_kwargs is None else solver_kwargs
    
    metrics_input = predict_on_ood_data(
        N,
        flow_matching,
        ood_data,
        num_time_steps,
        solver_kwargs,
    )
    return {
        "metrics": metrics_callback.run_on_valid_step(
            metrics_input
        ), 
        "samples": metrics_input
    }


def create_dir(path, logger):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Dump dir created at {path}.")
    else:
        logger.warning(f"Dump dir {path} alrerady exists.")


def get_generated_adata(
    x_true, x_gen,
    n_channel_feats=21, is_pca=False, **kwargs
):
    obs = {"dataset_type": ["Real" for _ in range(x_true.shape[0])] + ["Generated" for _ in range(x_gen.shape[0])], **kwargs}
    if x_true.shape[-1]>n_channel_feats:
        x_true_scatter = x_true[:, n_channel_feats:]
        x_true = x_true[:, -n_channel_feats:]

        x_gen_scatter = x_gen[:, n_channel_feats:]
        x_gen = x_gen[:, n_channel_feats:]
        obsm = {"X_scatter": np.concatenate((x_true_scatter, x_gen_scatter), axis=0)}
    if is_pca:
        obsm["X_pca"] = np.concatenate((x_true, x_gen), axis=0)
    adata_generated = sc.AnnData(X=np.concatenate([x_true, x_gen], axis=0), obs=obs, obsm=obsm)

    if not is_pca:
        sc.pp.pca(adata_generated)
    sc.pp.neighbors(adata_generated)
    sc.tl.umap(adata_generated)
    return adata_generated


def plot_marginals(X_real, X_gen=None, col_names=None, title=""):
   
    # sanity checks
    n_channels = X_real.shape[1]
    if X_gen is not None:
        assert X_gen.shape[1] == n_channels
    if col_names is not None:
        assert len(col_names) == n_channels

    # create subplots (1 row, n_channels columns)
    fig, ax = plt.subplots(1, n_channels, figsize=(45, 5))
    fig.suptitle(f"{title}", fontsize=16)

    # loop over channels
    for i in range(n_channels):
        ax[i].grid(True)
        sns.kdeplot(X_real[:, i], ax=ax[i], color="green", label="real")
        if X_gen is not None:
            sns.kdeplot(X_gen[:, i], ax=ax[i], color="red", label="gen")
        ax[i].legend()
        if col_names is not None:
            ax[i].set_title(col_names[i])

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    # fig.show()
    return fig


def plot_results_and_return_adata(
    cond_dir,
    run_name,
    cond_samples,
    channel_columns=None,
    scatter_columns=None,
    is_pca=False
):

    # parsing results dictionary and creating concatenated anndata.
    x_gen = cond_samples[PredictionFields.PREDICTION_DATA]
    x_true = cond_samples[DataFields.TARGET_STATE]
    n_feats = 20 if is_pca else 21
    adata_generated = get_generated_adata(
        x_true, x_gen,
        n_channel_feats=n_feats, is_pca=is_pca,
    )

    # plot scatter marginals
    if "X_scatter" in adata_generated.obsm.keys():
        X_real = adata_generated[adata_generated.obs["dataset_type"] == "Real"].obsm["X_scatter"]
        X_gen = adata_generated[adata_generated.obs["dataset_type"] == "Generated"].obsm["X_scatter"]
        fig = plot_marginals(
            X_real,
            X_gen=X_gen,
            col_names=scatter_columns,
        )
        fig.savefig(
            os.path.join(cond_dir, f"{run_name}_scatter_marginals.png")
        )

    # plot channel marginals
    X_real = adata_generated[adata_generated.obs["dataset_type"] == "Real"].X
    X_gen = adata_generated[adata_generated.obs["dataset_type"] == "Generated"].X
    fig = plot_marginals(
        X_real,
        X_gen=X_gen,
        col_names=channel_columns,
    )
    fig.savefig(
        os.path.join(cond_dir, f"{run_name}_channels_marginals.png")
    )
    return adata_generated

