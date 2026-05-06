from functools import partial

import numpy as np
import pandas as pd
import torch

from sc_exp_design.constants import DataFields, PredictionFields


# define upper bound
def map_df(df, transforms_dict):
    transformed_data_df = {}
    for mol in df.columns:
        trnsf = transforms_dict[mol]
        val = df.loc[:, mol].values
        if trnsf is not None:
            val = trnsf(val)
        transformed_data_df[mol] = val
    return pd.DataFrame(transformed_data_df, index=df.index)


def constraint_fn_factory(config, device, protocol_trasnf_factory):
    def get_constraints_and_mask(
        config,
    ):
        # parse annotation values
        protocol_cols = config.annotation.protocol_columns
        log1p_exp_cols = config.annotation.log1p_exp_cols
        log21p_exp_cols = config.annotation.log21p_exp_cols

        # parse constraints values
        constraints_in_original_space = config.constraints.constraints_in_original_space
        lbound_dict = config.constraints.lbound_dict
        ubound_dict = config.constraints.ubound_dict

        # compute gradient mask
        mask = [col in lbound_dict and col in ubound_dict for col in protocol_cols]

        # construct bound df
        lbound_df = pd.DataFrame({key: np.array([val]) for key, val in lbound_dict.items()})
        ubound_df = pd.DataFrame({key: np.array([val]) for key, val in ubound_dict.items()})

        # optionaly get protocol transformations
        if constraints_in_original_space:
            ptransf = protocol_trasnf_factory(
                protocol_cols,
                log1p_exp_cols=log1p_exp_cols,
                log21p_exp_cols=log21p_exp_cols,
                inverse=False
            )
            lbound_df = map_df(lbound_df, ptransf)
            ubound_df = map_df(ubound_df, ptransf)

        # initialize tensors
        lbound = torch.zeros(len(protocol_cols)).float().to(device)
        ubound = torch.zeros(len(protocol_cols)).float().to(device)

        # write bounds values
        for k in lbound_df.columns:
            idx = protocol_cols.index(k)
            lbound[idx] = lbound_df[k].item()
        for k in ubound_df.columns:
            idx = protocol_cols.index(k)
            ubound[idx] = ubound_df[k].item()
        return lbound, ubound, mask

    def _quad_bound(x1, bound, mask, upper=True):
        x1 = x1[..., mask]
        bound = bound[..., mask]
        deviation = x1 - bound if upper else bound - x1
        if config.constraints.use_exponential_penalty:
            val = torch.expm1(deviation)
        else:
            val = torch.nn.functional.relu(deviation)**2
        return torch.sum(val, dim=-1)
    lbound, ubound, mask = get_constraints_and_mask(config)
    return [
        lambda x1: _quad_bound(x1, ubound, mask, upper=True),
        lambda x1: _quad_bound(x1, lbound, mask, upper=False),
    ]


def loss_fn_factory(
    loss_fn,
    config,
    optimal_condition,
    non_linearity,
    forward_model,
):
    

    # sampling fixed noise for the forward model
    if config.loss_guidance.fix_noise:
        noise = forward_model.forward_model.noise_distribution(
            (
                config.sampling.N,
                config.loss_guidance.num_forward_pass_per_sample,
                forward_model.forward_model.velocity_field.config.flow_dim
            )
        ).squeeze(dim=0).to(forward_model.forward_model.device)
    else:
        noise = None

    def compute_target_loss(target_pred_dict, optimal_condition):
        """"""
        if config.loss_guidance.fix_noise:
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


def linear_scheduler_with_warmup(
    t,
    t_warmup=0.0,
    vmin=1.0,
    vmax=1.0,
):
    slope = (vmax - vmin)/(1 - t_warmup)
    t_shifted = torch.nn.functional.relu(t - t_warmup)
    c = t_shifted*slope + vmin
    return c[..., 0]


# 2. Define the recursive patch function
def replace_silu_with_safe_silu(model: torch.nn.Module) -> None:
    # 1. Define the AD-safe SiLU
    class SafeSiLU(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)
    for name, module in model.named_children():
        if isinstance(module, torch.nn.SiLU):
            setattr(model, name, SafeSiLU())
        else:
            # Recursively apply to sub-modules
            replace_silu_with_safe_silu(module)


# ---- MARKER BASED LOSS FUNCTIONS ----
def l2_loss(pred, target, feature_mask, sigma=0.25):
    """
    pred: (N, M, D)
    target: (1, D)
    """
    err = pred - target
    err = err[..., feature_mask]
    return torch.sum(err**2, dim=-1)/(2*(sigma**2))


def l1_loss(pred, target, feature_mask):
    """
    pred: (N, M, D)
    target: (1, D)
    """
    err = pred - target
    err = err[..., feature_mask]
    return torch.sum(torch.abs(err), dim=-1)  # (N,)


def cauchy_loss(pred, target, feature_mask, gamma=1.0):
    """
    pred: (N, M, D)
    target: (1, D)
    """
    err = pred - target
    err = err[..., feature_mask]
    err = torch.sum((err/gamma)**2, dim=-1)
    return 0.5*(gamma**2)*torch.log(1 + err)


def hinge_loss(pred, target, feature_mask):
    """
    pred: (N, M, D)
    target: (1, D)
    """
    err = target - pred
    err = err[..., feature_mask]
    err = torch.nn.functional.relu(err)
    return torch.sum(err**2, dim=-1)


#--- Factory for Loss Functions ---
def loss_fn_factory_marker_opt(
    config,
    loss_fn,
    target, # (1, D)
    non_linearity,
    cellular_response_model,
    target_feats_mask,
    agg_type, # "cell", "pop"
    loss_kwargs=None,
):
    """
    Returns (loss_fn, noise)
    """
    # Fixed noise if required
    if config.loss_guidance.fix_noise:
        noise = cellular_response_model.noise_distribution(
            (config.sampling.N,
             config.loss_guidance.num_forward_pass_per_sample,
             cellular_response_model.velocity_field.config.flow_dim)
        ).squeeze(dim=0).to(cellular_response_model.device)
    else:
        noise = None

    # prepare loss fn
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    loss_fn = partial(loss_fn, **loss_kwargs)

    def _compute_loss(x1):
        if non_linearity is not None:
            x1 = non_linearity(x1)

        batch_dict = {}
        x1_fwd = x1
        if noise is not None:
            batch_dict[DataFields.SOURCE_STATE] = noise
            if config.loss_guidance.fix_noise:
                x1_fwd = x1.unsqueeze(1).repeat(1, noise.shape[1], 1)

        condition_repr = next(iter(cellular_response_model.train_data.data.perturbation_covariates))
        batch_dict[DataFields.PERTURBATION_DATA] = {
            condition_repr: x1_fwd
        }

        pred = cellular_response_model.predict(
            batch_dict,
            no_grad=False,
            fix_noise=config.loss_guidance.fix_noise,
            num_time_steps=config.loss_guidance.n_time_steps_forward_model,
            solver_kwargs=config.loss_guidance.solver_kwargs_forward_model
        )   # shape (N, M, D)

        # Dispatch
        if agg_type == "pop":
            pred = pred.mean(-2)
        loss = loss_fn(pred, target, target_feats_mask)
        if agg_type == "cell":
            loss = loss.mean(-1)
        return loss

    return _compute_loss, noise
