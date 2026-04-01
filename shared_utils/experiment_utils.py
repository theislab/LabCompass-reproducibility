from collections import defaultdict
import logging
import math
import os
import sys
from typing import Any

import numpy as np
from omegaconf import DictConfig
import pandas as pd
import scanpy as sc
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import torch
from tqdm import tqdm

from sc_exp_design.constants import DataFields, ParamsFields, PredictionFields
from sc_exp_design.metrics import compute_e_distance
from sc_exp_design.models import FlowMatching, TargetPredictionModel

sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/shared_utils")
sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/shared_utils")
from forward_model import ForwardModel
from z_norm_modules import ZNorm, IZNorm, RescaledTargetPredictionModel


def create_dir(
    path: str,
    logger: logging.Logger | None = None
) -> str:
    """Create a directory if it does not exist and log the action."""
    os.makedirs(path, exist_ok=True)
    if logger is not None:
        logger.info(f"Directory ready: {path}")
    return path


def get_forward_model(
    config: DictConfig,
    logger: logging.Logger | None = None
) -> "ForwardModel":
    # 0. Forward Model
    # step 0.0 Load perturbation response prediction model
    if logger is not None:
        logger.info(f"Loading Perturbation Response Prediction Model From {config.paths.perturbation_prediction_path}...")
    perturbation_response_prediction_model = FlowMatching.load(
        config.paths.perturbation_prediction_path
    )
    if logger is not None:
        logger.info(f"Model loaded!\n{perturbation_response_prediction_model.velocity_field}")

    # step 0.1 Load cell type classification model
    if logger is not None:
        logger.info(f"Loading Cell Type Classifier From {config.paths.ct_classifier_path}...")
    target_prediction_model = TargetPredictionModel.load(
        config.paths.ct_classifier_path
    )
    if logger is not None:
        logger.info(f"Model loaded!\n{target_prediction_model.target_prediction_model}")


    # 0.2 Inverse Z-normalization from perturbation response prediction model
    if logger is not None:
        logger.info("Retrieving (inverse) Z-standardization parameters from cellular response prediction model...")
    X_channel_params_inv = perturbation_response_prediction_model.train_data.adata.uns["X_channel_params"]
    X_scatter_params_inv = perturbation_response_prediction_model.train_data.adata.uns["X_scatter_params"]
    params_inv = {
        ParamsFields.MEAN: torch.from_numpy(
            np.concatenate(
                (
                    X_channel_params_inv[ParamsFields.MEAN],
                    X_scatter_params_inv[ParamsFields.MEAN]
                ), axis=0
            )
        ),
        ParamsFields.COVARIANCE: torch.from_numpy(
            np.concatenate(
                (
                    X_channel_params_inv["std"],
                    X_scatter_params_inv["std"]
                ), axis=0
            )
        ),
    }
    izn = IZNorm(params_inv)
    if logger is not None:
        logger.info(f"Cellular response prediction inverse Z-standardization parameters ready {izn}!")

    # 0.3 Z-normalization from cell type classification model
    if logger is not None:
        logger.info("Retrieving Z-standardization parameters from cell type classifier.")
    X_channel_params_fwd = target_prediction_model.train_data.adata.uns["X_channel_params"]
    X_scatter_params_fwd = target_prediction_model.train_data.adata.uns["X_scatter_params"]
    params_fwd = {
        ParamsFields.MEAN: torch.from_numpy(
            np.concatenate(
                (
                    X_channel_params_fwd[ParamsFields.MEAN],
                    X_scatter_params_fwd[ParamsFields.MEAN]
                ), axis=0
            )
        ),
        ParamsFields.COVARIANCE: torch.from_numpy(
            np.concatenate(
                (
                    X_channel_params_fwd["std"],
                    X_scatter_params_fwd["std"]
                ), axis=0
            )
        ),
    }
    zn = ZNorm(params_fwd)
    if logger is not None:
        logger.info(f"Cell Type Classifier Z-standardization parameters ready {zn}!")

    # 0.4 Prepare rescaled model
    if logger is not None:
        logger.info("Initializing rescaled target prediction model...")
    g_model = RescaledTargetPredictionModel(
        target_prediction_model.target_prediction_model,
        inv_params=izn,
        fwd_params=zn
    )
    if logger is not None:
        logger.info(f"Rescaled model ready!\n{g_model}")

    # 0.5 Initialize forward Model
    return ForwardModel(
       forward_model=perturbation_response_prediction_model,
        target_prediction_model=g_model,
    ), (
        perturbation_response_prediction_model,
        target_prediction_model
    )


def generate_with_condition(
    samples,
    forward_model,
    num_time_steps,
    solver_kwargs,
    le_ct,
    num_samples=None,
    noise=None,
    logger=None,
    n_scatter_feats=6,
):
    # prepare batch data
    perturbation_reps = next(iter(forward_model.forward_model.train_data.data.perturbation_covariates))
    ccondition_data = torch.from_numpy(samples).float().to(forward_model.forward_model.device)
    if noise is not None:
        ccondition_data = ccondition_data.unsqueeze(1).repeat(1, noise.shape[1], 1)
    num_samples = num_samples if noise is None else None
    ccondition_dict = {
        perturbation_reps: ccondition_data,
    }
    batch_dict = {
        DataFields.PERTURBATION_DATA: ccondition_dict,
    }
    if noise is not None:
        batch_dict[DataFields.SOURCE_STATE] = noise

    # query forward model
    cforward_out = forward_model.predict(
        batch_dict,
        return_trajectory=False,
        num_samples=num_samples,
        no_grad=True,
        num_time_steps=num_time_steps,
        solver_kwargs=solver_kwargs,
        fix_noise=noise is not None,
    )
    X_gen = cforward_out[PredictionFields.PREDICTION_DATA].detach().cpu().numpy()
    gen_ct_logits = cforward_out[PredictionFields.TARGET_PREDICTION_DATA]["cell_type"].detach().cpu().numpy()

    # split channel and scatter features
    X_channel_gen = X_gen[..., :-n_scatter_feats]
    X_scatter_gen = X_gen[..., -n_scatter_feats:]

    # compute cell type probabilities and labels
    gen_ct_probs = softmax(gen_ct_logits, axis=-1)
    gen_ct_id_label = gen_ct_probs.argmax(-1)
    # gen_ct_label = le_ct.inverse_transform(gen_ct_id_label.reshape(-1)).\
    #     reshape(gen_ct_id_label.shape[0], gen_ct_id_label.shape[1])
    if logger is not None:
        logger.info(f"{X_gen.shape=}, {X_channel_gen.shape=}, {X_scatter_gen.shape=}, {gen_ct_logits.shape=}")# {gen_ct_id_label.shape=}, {gen_ct_label.shape=}")
    return {
        "X": X_gen,
        "X_channel": X_channel_gen,
        "X_scatter": X_scatter_gen,
        "ct_logits": gen_ct_logits,
        "ct_probs": gen_ct_probs,
    }


def query_forward_model(
    traj: np.ndarray,
    noise: np.ndarray,
    forward_model: "ForwardModel",
    num_time_steps: int,
    solver_kwargs: dict[str, Any],
    le_ct: LabelEncoder,
    n_scatter_feats: int = 6,
    logger: logging.Logger | None = None,
    dim_to_take: int = 1,
):
    # prepare condition data
    samples = np.maximum(np.take(traj, -1, axis=dim_to_take), 0) # this is hard-coded now, maybe change?
    return generate_with_condition(
        samples,
        forward_model,
        num_time_steps,
        solver_kwargs,
        le_ct,
        noise=noise,
        logger=logger,
        n_scatter_feats=n_scatter_feats,
    )


def flatten_conf(conf_dict):
    return_dict = {}
    for key, value in conf_dict.items():
        if isinstance(value, dict):
            update_dict = {f"{key}:{k}": v for k, v in flatten_conf(value).items()}
        else:
            update_dict = {key: value}
        return_dict.update(update_dict)
    return return_dict


def get_transformed_data(
    samples,
    protocol_columns,
    column2tranform,
):
    data_dict_original = {
        mol: samples[:, idx] for idx, mol in enumerate(protocol_columns) 
    }
    data_dict_transformed = {}
    for mol, val in data_dict_original.items():
        trnsf = column2tranform[mol]
        if trnsf is not None:
            val = trnsf(val)
        data_dict_transformed[mol] = val
    return data_dict_transformed, data_dict_original


def get_target_dict(
    config,
    classes,
    device
):
    if config.sampling.query_pure_cell_types:
        nclasses = len(classes)
        idx = classes.index(config.sampling.target_cell_type)
        target = torch.zeros((nclasses,)).float().to(device)
        target[idx] = 1.0
        target = {
            "cell_type": target.unsqueeze(0)
        }
    else:
        prop = torch.tensor(
            config.sampling.target_probs
        ).float().to(device)
        target = {
            "cell_type": prop.unsqueeze(0)
        }
    return target


def get_loss_fn(config):
    if config.sampling.query_pure_cell_types and config.sampling.mask_gradients:
        mask = config.sampling.mask
        return  {
            "cell_type": lambda pred, target: -torch.sum(target[..., mask]*torch.nn.functional.log_softmax(pred[..., mask], dim=-1), dim=-1)
        }
    return {
        "cell_type": lambda pred, target: -torch.sum(target*torch.nn.functional.log_softmax(pred, dim=-1), dim=-1)
    }


def generated_density_knn(
    X_true,
    X_generated,
    k=20,
    bandwidth=None
):

    nbrs = NearestNeighbors(n_neighbors=k).fit(X_true)
    distances, indices = nbrs.kneighbors(X_generated)
    
    # Auto-bandwidth: median distance of neighbors
    if bandwidth is None:
        bandwidth = np.median(distances)
    density = np.zeros(X_true.shape[0])

    # Gaussian kernel weights
    weights = np.exp(-(distances**2) / (2 * bandwidth**2))

    # Add weighted contributions
    for gen_idx in range(len(X_generated)):
        for neighbor_pos, real_idx in enumerate(indices[gen_idx]):
            density[real_idx] += weights[gen_idx, neighbor_pos]
    return density


def get_adata_from_idx(X_true, adata_g, ct_le, fwd_results, min_loss_idx, compute_stuff=True, n_scatter_feats=6):
    X_gen = fwd_results["X"][min_loss_idx]
    ct_label_gen = ct_le.inverse_transform(
        fwd_results["ct_probs"][min_loss_idx].argmax(1)
    )

    X = np.concat((X_gen, X_true), axis=0)
    X_channel = X[:, :-n_scatter_feats]
    X_scatter = X[:, -n_scatter_feats:]
    G = np.concatenate((ct_label_gen, adata_g.obs["cell_type"].values), axis=0)
    ct_adata_gen = sc.AnnData(
        X=X_channel,
        obsm={"X_scatter": X_scatter},
        obs={
            "cell_type": G,
            "data_type": ["gen"]*X_gen.shape[0] + \
                ["real"]*len(adata_g)
        },
        var=pd.DataFrame(index=adata_g.var_names)
    )
    if compute_stuff:
        sc.pp.pca(ct_adata_gen)
        sc.pp.neighbors(ct_adata_gen)
        sc.tl.umap(ct_adata_gen)
    return ct_adata_gen


def get_distance_df(
    adata,
    obsm_key,
    classes,
    distance_fn=compute_e_distance,
):
    adata_pred = adata[adata.obs["data_type"] == "gen"]
    adata_true = adata[adata.obs["data_type"] != "gen"]

    edist_dict = defaultdict(list)
    for ct in tqdm(classes):
        adata_ct = adata_true[adata_true.obs.cell_type == ct]
        x_true = adata_ct.obsm[obsm_key] if obsm_key is not None else adata_ct.X
        edist = distance_fn(adata_pred.obsm[obsm_key]if obsm_key is not None else adata_pred.X, x_true)
        edist_dict["ct"].append(ct)
        edist_dict["edist"].append(edist)
    return pd.DataFrame(edist_dict).sort_values("edist", axis=0, ascending=False)


def get_distance_matrix(
    X,
    Y,
    metric="jensen-shannon"
):
    """
    Compute KL divergence from each row in X to each row in Y.
    X: shape (N, D)
    Y: shape (K, D)
    Returns: shape (N, K)
    """
    eps = 1e-12
    X = np.clip(X, eps, 1)[:, None, :]
    Y = np.clip(Y, eps, 1)[None, :, :]
    if metric == "jensen-shannon":
        return np.sqrt(0.5*np.sum(X*np.log(2*X / (Y + X)), axis=-1) + 0.5*np.sum(Y*np.log(2*Y / (Y + X)), axis=-1))
    elif metric == "jeffrey":
        return 0.5*np.sum(X* np.log(X / Y), axis=-1) + 0.5*np.sum(Y* np.log(Y/ X), axis=-1)
    elif metric == "kl-div":
        return np.sum(X * np.log(X / Y), axis=-1) # WARNING: not a metric
    else:
        raise ValueError


def get_dimensionality_reduced_condition_space(
    original_data,
    perturbation_prediction_model,
    n_noise_samples
):
    # extract latent representation from model
    perturbation_reps = next(
        iter(
            perturbation_prediction_model.forward_model.train_data.data.perturbation_covariates
        )
    )
    latent_rep = perturbation_prediction_model.forward_model.velocity_field.get_condition_embedding(
        {
            perturbation_reps: torch.from_numpy(
                original_data).float().to(perturbation_prediction_model.forward_model.device
            ),
        }
    ).detach().cpu().numpy()

    # sample noise for projection
    noise_orig = np.random.randn(n_noise_samples, original_data.shape[1], 2)
    noise_latent = np.random.randn(n_noise_samples, latent_rep.shape[1], 2)

    # project latent representation and data
    orig_rep_rand_proj = np.einsum("nd,mdk->nmk", original_data, noise_orig) / math.sqrt(2)
    latent_rep_rand_proj = np.einsum("nd,mdk->nmk", latent_rep, noise_latent) / math.sqrt(2)

    # compute pcs
    orig_rep_pcs = PCA(2).fit_transform(original_data)
    latent_rep_pcs = PCA(2).fit_transform(latent_rep)
    return (
        orig_rep_rand_proj, latent_rep_rand_proj, orig_rep_pcs, latent_rep_pcs
    )


def get_sample_grid_per_dim(samples, dim, grid_size=5, lbound=None, ubound=None):
    if lbound is not None and ubound is not None:
        pert_arr = np.linspace(lbound, ubound, num=grid_size*2)
    else:
        pert_arr = np.array(
            [samples[dim] - (step/grid_size)*samples[dim] for step in reversed(range(1, grid_size + 1))] + \
            [samples[dim] + (step/grid_size)*samples[dim] for step in range(1, grid_size + 1)]
        )
    perturbed = []
    for val in pert_arr:
        perturbed_sample = samples.copy()
        perturbed_sample[..., dim] = val
        perturbed.append(perturbed_sample)
    return np.stack(perturbed, axis=0)


def get_sample_grid(samples, grid_size=5, lbound=None, ubound=None):
    perturbed = []
    for dim in range(samples.shape[-1]):
        dim_lbound = lbound[dim] if lbound is not None else None
        dim_ubound = ubound[dim] if ubound is not None else None
        grid = get_sample_grid_per_dim(samples, dim, grid_size=grid_size, lbound=dim_lbound, ubound=dim_ubound)
        perturbed.append(grid)
    return np.stack(perturbed, axis=0)


def get_sensitivity_results(
    forward_model,
    sample_grid,
    noise,
    n_time_steps,
    solver_kwargs,
    le_ct,
    protocol_cols=None,
):
    pbar = tqdm(range(sample_grid.shape[0]*sample_grid.shape[0]))
    ct_res = {}
    for perturbed_ax in range(sample_grid.shape[0]):
        ax_grid = sample_grid[perturbed_ax]
        if protocol_cols is not None:
            assert ax_grid.shape[0] == len(protocol_cols)
            ax_name = protocol_cols[perturbed_ax]
        else:
            ax_name = perturbed_ax
        ax_res = {}
        for grid_val in range(ax_grid.shape[0]):
            pbar.set_description(f"axes:{ax_name}-grid:{grid_val}")
            pbar.update()
            val = ax_grid[grid_val]
            fwd_results = generate_with_condition(
                val,
                forward_model,
                n_time_steps,
                solver_kwargs,
                le_ct,
                noise=noise,
            )
            fwd_results["mean_probs"] = fwd_results["ct_probs"].mean(1)
            ax_res[grid_val] = fwd_results
        ct_res[ax_name] = ax_res
    return ct_res


def shuffle_and_reshape(data, n_pops=5):
    idxs = np.arange(data.shape[0])
    idxs = np.random.shuffle(idxs)
    return data[idxs].reshape(n_pops, -1, data.shape[-1])


def get_shuffled_populations(data, n_pops=5, n_iters=500):
    shuffled_data_list = []
    for _ in range(n_iters):
        shuffled_data = shuffle_and_reshape(data, n_pops=n_pops)
        shuffled_data_list.append(shuffled_data)
    return np.stack(shuffled_data_list, axis=0)


def get_g_star(ct, classes):
    # get target probability vector for current cell type
    ct_idx = classes.index(ct)
    g_star = np.zeros(len(classes))
    g_star[ct_idx] = 1.0
    return g_star[None]


def compute_exploitation_score(surr_candidates_vals):
    max_surr_loss_candidates = surr_candidates_vals.max()
    min_surr_loss_candidates = surr_candidates_vals.min()
    return (surr_candidates_vals - min_surr_loss_candidates) / (max_surr_loss_candidates - min_surr_loss_candidates)


def compute_exploration_score(Dct):
    candidates_nn_dist = Dct.min(1)
    max_nn_dist = candidates_nn_dist.max()
    min_nn_dist = candidates_nn_dist.min()
    return (max_nn_dist - candidates_nn_dist)/(max_nn_dist - min_nn_dist)
 

def compute_surr_exponential_weight(surr_candidates_vals, gamma=1.0):
    exploitation_score = compute_exploitation_score(surr_candidates_vals)
    return gamma*np.exp(-exploitation_score)


def compute_weighted_uncertainty_score(
    surr_candidates_vals, gen_samples, data_samples, gamma=1.0
):
    exp_weight = compute_surr_exponential_weight(surr_candidates_vals, gamma=gamma) # N
    D_data = cdist(gen_samples, data_samples) # N, M
    D_candidates = cdist(
        gen_samples, gen_samples
    ) # N, N
    U_data = D_data.min(1) # N
    U_concat_list = []
    for idx in range(D_candidates.shape[1]):
        D_concat = np.concatenate((D_data, D_candidates[:, idx][:, None]), axis=1) # N, M + 1
        U_concat = D_concat.min(1) # N 
        U_concat_list.append(U_concat)
    U_concat_arr = np.stack(U_concat_list, axis=0) # N, N
    return ((U_data[None] - U_concat_arr)*exp_weight[None]).mean(1)


def compute_ucb_like_acq_fn(mean_loss, std_loss, kappa=1.0):
    return -mean_loss + kappa * std_loss


def compute_phi(X_rest, X_max, gamma=1.0):
    dists_sq = np.sum((X_rest - X_max)**2, axis=1)
    return 1.0 - np.exp(-gamma * dists_sq)


def update_step_with_tracking(acq_fn, X, indices, mean_loss, gamma=1.0):
    # 1. Identify the best point in the current (penalized) acquisition landscape
    max_idx = acq_fn.argmax()
    
    # 2. Extract values for the selected point
    X_max = X[max_idx][None]
    orig_idx = indices[max_idx]
    loss_val = mean_loss[max_idx]
    acq_val = acq_fn[max_idx]

    # 3. Remove the selected point from the pool
    mask = np.ones(X.shape[0], dtype=bool)
    mask[max_idx] = False

    X_rest = X[mask]
    indices_rest = indices[mask]
    acq_fn_rest = acq_fn[mask]
    mean_loss_rest = mean_loss[mask]

    # 4. Apply distance penalty to the remaining points
    phi_val = compute_phi(X_rest, X_max, gamma=gamma)
    penalized_acq = acq_fn_rest * phi_val
    
    return X_max, X_rest, indices_rest, penalized_acq, mean_loss_rest, loss_val, acq_val, orig_idx


def compute_sequential_local_penalization(
    X_candidates,
    mean_candidates_loss,
    std_candidates_loss,
    gamma=1.0,
    kappa=1.0,
):
    # 1. Initial Acquisition calculation
    raw_acq_fn = compute_ucb_like_acq_fn(mean_candidates_loss, std_candidates_loss, kappa=kappa)
    
    # Shift to positive to ensure the local penalizer (0, 1] works as intended
    current_acq = raw_acq_fn - raw_acq_fn.min() 
    
    # 2. Setup tracking
    num_particles = X_candidates.shape[0]
    current_X = X_candidates.copy()
    current_loss = mean_candidates_loss.copy()
    current_indices = np.arange(num_particles) # Track original indices

    results = {
        "indices": [],
        "losses": [],
        "acq_values": []
    }

    # 3. Iteratively pick and penalize until all particles are ranked
    for _ in range(num_particles):
        (
            X_max, 
            current_X, 
            current_indices,
            current_acq, 
            current_loss, 
            loss_val, 
            acq_val,
            orig_idx
        ) = update_step_with_tracking(current_acq, current_X, current_indices, current_loss, gamma=gamma)
        
        results["indices"].append(orig_idx)
        results["losses"].append(loss_val)
        results["acq_values"].append(acq_val)

    # Convert to arrays for easier downstream handling
    return {k: np.array(v) for k, v in results.items()}
def manual_filtering(df_dict, 
                     bounds, 
                     columns_to_keep, 
                     celltype_fraction, 
                     margin_frac_bounds, 
                     protocol_cols):
    
    # Filtered data frame 
    df_dict_filtered = {}
    df_dict_annotated = {}
    
    # Iterate over cell types 
    for cell_type in df_dict:
        # Collect cell type dictionary and sort by proportion 
        cell_type_revert_safe_string = cell_type.replace(":", "/")
        df_ct = df_dict[cell_type].copy()
        df_ct = df_ct.sort_values(by=f"{cell_type_revert_safe_string}_prop", ascending=False)
        
        # Subset data frame 
        columns_to_keep_ct = columns_to_keep + [f"{cell_type_revert_safe_string}_prop"]
        df_ct = df_ct.loc[:, columns_to_keep_ct]
        df_ct["filtering_step"] = "pass" 

        # Filter by fraction
        idx_lower = df_ct[f"{cell_type_revert_safe_string}_prop"] < celltype_fraction[cell_type_revert_safe_string] 
        df_ct.loc[idx_lower & (df_ct["filtering_step"] == "pass"), 
                  "filtering_step"] = "proportion_filter"

        # Oxygen and days filtering 
        idx_pass_oxygen_days = np.logical_and(df_ct["o2_[%]"] > 5, 
                                             df_ct["o2_[%]"] < 25)
        idx_pass_oxygen_days = np.logical_and(idx_pass_oxygen_days, 
                                             df_ct["days_of_culture"] > 12)
        idx_pass_oxygen_days = np.logical_and(idx_pass_oxygen_days, 
                                             df_ct["days_of_culture"] < 20)
        df_ct.loc[~idx_pass_oxygen_days & (df_ct["filtering_step"] == "pass"), 
                  "filtering_step"] = "oxygen_days"

        # Bounds 
        for param in protocol_cols:
            idx_bounds = np.logical_and(
                df_ct[param] >= (bounds[param][0] - bounds[param][0] * margin_frac_bounds), 
                df_ct[param] <= (bounds[param][1] + bounds[param][1] * margin_frac_bounds)
            )
            df_ct.loc[~idx_bounds & (df_ct["filtering_step"] == "pass"), 
                      "filtering_step"] = f"bounds_{param}"

        # Filtered dataset
        df_dict_filtered[cell_type] = df_ct[df_ct.filtering_step == "pass"]
        df_dict_annotated[cell_type] = df_ct[df_ct.filtering_step != "proportion_filter"]
        
    return df_dict_filtered, df_dict_annotated 
