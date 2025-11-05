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



def log_prob_normal(
    X, # BxD
    params,
):
    mean = params[ParamsFields.MEAN]  
    log_std = params[ParamsFields.COVARIANCE]

    assert len(X.shape) == len(mean.shape), f"{X.shape=}, {mean.shape=}"
    assert len(X.shape) == len(log_std.shape), f"{X.shape=}, {log_std.shape=}"
    assert X.shape[-1] == mean.shape[-1], f"{X.shape=}, {mean.shape=}"
    assert X.shape[-1] == log_std.shape[-1], f"{X.shape=}, {log_std.shape=}"

    D = X.shape[-1]
    log_Z = torch.sum(log_std, dim=-1) + 0.5*D*math.log(2*np.pi)
    cov = torch.exp(2*log_std)
    diff = X - mean
    log_potential = -0.5*torch.einsum("...d, ...d -> ...", diff, diff/cov)
    return torch.mean(log_potential - log_Z)


def Dkl_standard_normal(params, eps=1e-15):
    mean = params[ParamsFields.MEAN]
    log_std = params[ParamsFields.COVARIANCE]
    rv = -torch.sum(1 + 2*log_std - mean**2 - torch.exp(2*log_std), dim=-1)
    assert torch.all(torch.isfinite(rv)), log_std
    return 0.5*torch.mean(rv, dim=-1)

def compute_kernel(x, y):
    x_size, dim = x.shape
    y_size = y.shape[0]
    
    # Expand and broadcast
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, dim)  # shape: [x_size, y_size, dim]
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, dim)  # shape: [x_size, y_size, dim]
    
    # Compute squared differences and mean over feature dimension
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim)

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)

