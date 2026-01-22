import logging
import os
import sys
from typing import Any

import numpy as np
from omegaconf import DictConfig
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder
import torch

from sc_exp_design.constants import DataFields, ParamsFields, PredictionFields
from sc_exp_design.models import FlowMatching, TargetPredictionModel

sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/model_utils")
sys.path.insert(0, "/lustre/groups/ml01/workspace/lorenzo.consoli/projects/SFC_cambridge/collab-goettgens-SFC/inverse/inverse_utils")
from forward_model import ForwardModel
from z_norm_modules import ZNorm, IZNorm, RescaledTargetPredictionModel


def create_dir(path: str, logger: logging.Logger | None = None) -> str:
    """Create a directory if it does not exist and log the action."""
    os.makedirs(path, exist_ok=True)
    if logger is not None:
        logger.info(f"Directory ready: {path}")
    return path


def get_forward_model(config: DictConfig, logger: logging.Logger | None = None) -> "ForwardModel":
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


def query_forward_model(
    traj: np.ndarray,
    noise: np.ndarray,
    forward_model: "ForwardModel",
    num_time_steps: int,
    solver_kwargs: dict[str, Any],
    le_ct: LabelEncoder,
    n_scatter_feats: int = 6,
    logger: logging.Logger | None = None
):
    # prepare condition data
    samples = np.maximum(traj[:, -1, :], 0) # this is hard-coded now, maybe change?
    perturbation_reps = next(iter(forward_model.forward_model.train_data.data.perturbation_covariates))
    ccondition_data = torch.from_numpy(samples).float().to(forward_model.forward_model.device)
    ccondition_dict = {
        perturbation_reps: ccondition_data.unsqueeze(1).repeat(1, noise.shape[1], 1),
    }

    # construct batch dictionary
    batch_dict = {
        DataFields.PERTURBATION_DATA: ccondition_dict,
        DataFields.SOURCE_STATE: noise
    }

    # query forward model
    cforward_out = forward_model.predict(
        batch_dict,
        return_trajectory=False,
        no_grad=True,
        num_time_steps=num_time_steps,
        solver_kwargs=solver_kwargs,
        fix_noise=True,
    )
    X_gen = cforward_out[PredictionFields.PREDICTION_DATA].detach().cpu().numpy()
    gen_ct_logits = cforward_out[PredictionFields.TARGET_PREDICTION_DATA]["cell_type"].detach().cpu().numpy()

    # split channel and scatter features
    X_channel_gen = X_gen[..., :-n_scatter_feats]
    X_scatter_gen = X_gen[..., -n_scatter_feats:]

    # compute cell type probabilities and labels
    gen_ct_probs = softmax(gen_ct_logits, axis=-1)
    gen_ct_id_label = gen_ct_probs.argmax(-1)
    gen_ct_label = le_ct.inverse_transform(gen_ct_id_label.reshape(-1)).\
        reshape(gen_ct_id_label.shape[0], gen_ct_id_label.shape[1])
    if logger is not None:
        logger.info(f"{X_gen.shape=}, {X_channel_gen.shape=}, {X_scatter_gen.shape=}, {gen_ct_logits.shape=} {gen_ct_id_label.shape=}, {gen_ct_label.shape=}")
    return {
        "X": X_gen,
        "X_channel": X_channel_gen,
        "X_scatter": X_scatter_gen,
        "ct_logits": gen_ct_logits,
        "ct_probs": gen_ct_probs,
    }

