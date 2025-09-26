from collections.abc import Callable
from itertools import combinations
import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import energy_distance
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch

from sc_exp_design.constants import DataFields, LossFields, ParamsFields
from sc_exp_design.data import DataManager, SequentialDataLoader
from sc_exp_design.networks import ConditionEncoder, MLPGaussianNoiseModel
from sc_exp_design.training.base import BaseTrainer
from sc_exp_design.training.callbacks import BaseCallBack


def get_concatenated_transformed_obs_columns(
    adata: sc.AnnData,
    column2tranform: dict[str, Callable | None],
    obsm_col="cond_concat",
):
    for column, transorm_fn in column2tranform.items():
        col_values = adata.obs[column].values[:, None]
        if isinstance(adata.obs[column].values.dtype, pd.CategoricalDtype):
            col_values = adata.obs[column].astype(float).values[:, None]
        if transorm_fn is not None:
            col_values = transorm_fn(col_values)
        adata.obsm[column] = col_values
    adata.obsm[obsm_col] = np.concatenate(
        [adata.obsm[col] for col in column2tranform.keys()], axis=-1
    )
    return adata


def log_prob_normal(x, params, eps=1e-15):
    # parsing parameters dict
    mean = params[ParamsFields.MEAN]
    log_std = params[ParamsFields.COVARIANCE]
    log_Z = torch.sum(log_std, dim=-1) + 0.5*x.shape[-1]*math.log(2*np.pi)
    cov=torch.exp(2*log_std)
    potential = - 0.5 * torch.sum((x - mean)**2/(cov + eps), dim=-1)
    assert torch.all(torch.isfinite(log_Z))#, logDcov
    assert torch.all(torch.isfinite(potential))
    return potential - log_Z


# def log_prob_normal(
#     X, # BxD
#     params,
# ):
#     mean = params[ParamsFields.MEAN]  
#     log_std = params[ParamsFields.COVARIANCE]

#     assert len(X.shape) == 2
#     assert X.shape == mean.shape
#     assert X.shape == log_std.shape
#     _, D = X.shape

#     log_Z = torch.sum(log_std, dim=-1) + 0.5*D*math.log(2*np.pi)
#     cov = torch.exp(2*log_std)
#     diff = X - mean
#     log_potential = -0.5*torch.einsum("bd, bd -> b", diff, diff/cov)
#     return torch.mean(log_potential - log_Z)


def Dkl_standard_normal(params, eps=1e-15):
    mean = params[ParamsFields.MEAN]
    log_std = params[ParamsFields.COVARIANCE]
    rv = torch.sum(1 + 2*log_std - mean**2 - torch.exp(2*log_std), dim=-1)
    assert torch.all(torch.isfinite(rv)), log_std
    return 0.5*torch.mean(rv, dim=-1)
