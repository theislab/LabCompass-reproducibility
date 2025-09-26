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


class VAEModule(torch.nn.Module):
    linear_proj_config = {
        "hidden_dims":(),
        "use_dropout": False,
        "use_batchnorm": False,
        "activation_class": torch.nn.Identity,
        "final_activation_class": torch.nn.Identity,
    }

    def __init__(
        self,
        state_dim: int,
        state_latent_dim: int,
        condition_latent_dim: int | None = 2,
        latent_dim: int | None = None,
        encoder_mlp_kwargs=None,
        decoder_mlp_kwargs=None,
        layers_before_pooling: dict | None = None,
        layers_after_pooling: dict | None = None,
        is_conditional: bool = True,
    ):
        if is_conditional:
            assert condition_latent_dim is not None
            assert layers_before_pooling is not None
            assert layers_after_pooling is not None

        super().__init__()
        self.state_dim = state_dim
        self.state_latent_dim = state_latent_dim
        self.condition_latent_dim = condition_latent_dim
        self.latent_dim = latent_dim
        self.encoder_mlp_kwargs = {} if encoder_mlp_kwargs is None else encoder_mlp_kwargs
        self.decoder_mlp_kwargs = {} if decoder_mlp_kwargs is None else decoder_mlp_kwargs
        self.layers_before_pooling = layers_before_pooling
        self.layers_after_pooling = layers_after_pooling
        self.is_conditional = is_conditional
        self.vae_modules = self._init_modules()

    def _init_modules(self):
        modules = {}
        modules["encoder"] = MLPGaussianNoiseModel(
            self.encoder_input_dim,
            self.state_latent_dim,
            latent_dim=self.state_latent_dim,
            cov_estimation_mode="anisotropic",
            encoder_mlp_kwargs=self.encoder_mlp_kwargs,
            mean_mlp_kwargs=self.linear_proj_config,
            cov_mlp_kwargs=self.linear_proj_config,
        )
        modules["decoder"] = MLPGaussianNoiseModel(
            self.decoder_input_dim,
            self.state_dim,
            latent_dim=self.state_dim,
            cov_estimation_mode="anisotropic",
            encoder_mlp_kwargs=self.encoder_mlp_kwargs,
            mean_mlp_kwargs=self.linear_proj_config,
            cov_mlp_kwargs=self.linear_proj_config,
        )
        if self.latent_dim is not None:
            modules["proj"] = torch.nn.Linear(
                self.state_dim + self.condition_latent_dim if self.is_conditional else self.state_dim,
                self.latent_dim,
            )
        if self.is_conditional:
            modules["condition_encoder"] = ConditionEncoder(
                self.condition_latent_dim,
                layers_before_pooling=self.layers_before_pooling,
                layers_after_pooling=self.layers_after_pooling,
            )
        return torch.nn.ModuleDict(modules)
    
    def get_latent_params(
        self,
        x,
        latent_condition=None,
    ):
        # handling input for encoder
        if self.is_conditional:
            assert latent_condition is not None
            encoder_input = torch.concatenate((x, latent_condition), dim=-1)
        else:
            encoder_input = x
        return self.vae_modules["encoder"](encoder_input)

    def get_decoded_params(
        self,
        latent_states,
        latent_condition=None,
    ):
        # handling input for encoder
        if self.is_conditional:
            assert latent_condition is not None
            decoder_input = torch.concatenate((latent_states, latent_condition), dim=-1)
        else:
            latent_condition = None
            decoder_input = latent_states
        if self.latent_dim is not None:
            decoder_input = self.vae_modules["proj"](decoder_input)
        return self.vae_modules["decoder"](decoder_input)

    def encode_condition(
        self,
        condition=None,
    ):
        assert condition is not None
        return self.vae_modules["condition_encoder"](condition)
    
    def sample_latent_states(
        self,
        N,
        x,
        latent_condition=None,
    ):
        # encoding states
        latent_params = self.get_latent_params(x, latent_condition=latent_condition)
        return self.sample_from_params(N, latent_params)
    
    def sample_observations(
        self,
        N,
        latent_states,
        latent_condition=None,
    ):
        # encoding states
        latent_params = self.get_decoded_params(latent_states, latent_condition=latent_condition)
        return self.sample_from_params(N, latent_params)

    def sample_prior_predictive(
        self,
        n_prior_samples,
        n_likelihood_samples,
        condition=None,
    ):
        # encoding conditions
        if self.is_conditional:
            assert condition is not None
            cond_dict = {
                k: v.type(torch.float32).to(self.device)  for k, v in condition.items()
            }
            latent_cond = self.encode_condition(cond_dict)
            latent_cond = latent_cond.unsqueeze(0)
            # print(latent_cond.shape)
        else:
            latent_cond = None
        z_prior = torch.randn((n_prior_samples, self.vae_modules["encoder"].output_dim), device=self.device, dtype=torch.float32)
        # print(z_prior.shape)
        if latent_cond is not None:
            latent_cond = latent_cond.unsqueeze(0).repeat(n_likelihood_samples, *(1 for _ in latent_cond.shape))
        x_params = self.get_decoded_params(z_prior, latent_condition=latent_cond)
        # for k, v in x_params.items():print(k, v.shape)
        x_samples = self.sample_from_params(n_likelihood_samples, x_params)
        return {
            "z_samples": z_prior,
            "x_params": x_params,
            "x_samples": x_samples,
            "latent_cond": latent_cond
        }

    def sample_posterior_predictive(
        self,
        x,
        n_posterior_samples,
        n_likelihood_samples,
        condition=None,
    ): 
        # encoding conditions 
        if self.is_conditional:
            assert condition is not None
            cond_dict = { k: v for k, v in condition.items() }
            latent_cond = self.encode_condition(cond_dict)
        else: latent_cond = None
        z_params = self.get_latent_params(x, latent_condition=latent_cond)
        z_posterior = self.sample_from_params(n_posterior_samples, z_params) # m: (N, B, Z)
        if latent_cond is not None:
            latent_cond = latent_cond.unsqueeze(0).repeat(n_posterior_samples, *(1 for _ in latent_cond.shape))
        x_params = self.get_decoded_params(z_posterior, latent_condition=latent_cond)
        x_samples = self.sample_from_params(n_likelihood_samples, x_params)
        return { "z_params": z_params, "z_samples": z_posterior, "x_params": x_params, "x_samples": x_samples, "latent_cond": latent_cond }

    def to(
        self,
        device
    ):
        for k, v in self.vae_modules.items():
            self.vae_modules[k] = v.to(device)
        self.device = device

    @property
    def encoder_input_dim(
        self,
    ):
        if self.latent_dim is not None:
            return self.latent_dim
        if self.is_conditional:
            return self.state_dim + self.condition_latent_dim
        return self.state_dim

    @property
    def decoder_input_dim(
        self,
    ):
        if self.is_conditional:
            return self.state_latent_dim + self.condition_latent_dim
        return self.state_latent_dim

    @staticmethod
    def sample_from_params(N, params):
        # parsing parameter dictionary
        mean = params[ParamsFields.MEAN]
        std = params[ParamsFields.COVARIANCE]

        # sampling noise and matching shapes
        noise = torch.randn((N, *mean.shape), device=mean.device, dtype=torch.float)
        mean = mean.unsqueeze(0)
        cov = torch.exp(2*std.unsqueeze(0))
        return mean + torch.sqrt(cov)*noise


class VAETrainer(BaseTrainer):
    def __init__(
        self,
        model,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_step: Literal["grad_step", "valid_step"] = "grad_step",
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int | None = 10,
        num_grad_accumulation_steps: int = 20,
        reg_strength = 1e-1,
        n_mc_samples = 100,
        cond_reg_mode = None,
        cond_reg_strength=1e-1,
        eps=1e-15,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_step = lr_scheduler_step
        self.callbacks = callbacks
        self.grad_steps_log_interval = grad_steps_log_interval
        self.num_grad_accumulation_steps = num_grad_accumulation_steps 
        self.reg_strength = reg_strength
        self.n_mc_samples = n_mc_samples
        self.cond_reg_mode = cond_reg_mode
        self.cond_reg_strength = cond_reg_strength
        self.eps = eps

    def _train_step(
        self,
        step_idx: int,
        batch: dict[str, torch.Tensor],
    ):
        # parsing batch
        x = batch[DataFields.STATE_DATA] # (B, D)
        condition = batch.get(DataFields.PERTURBATION_DATA, None) # (B, C)

        # encoding condition
        if condition is not None:
            latent_cond = self.model.encode_condition(condition)
        else:
            latent_cond = None
        
        # encoder and latent regularization
        q_params = self.model.get_latent_params(x, latent_condition=latent_cond) # m: (B, Z), s_i: (B, 1), s_a: (B, Z)
        z_samples = self.model.sample_from_params(self.n_mc_samples, q_params) # m: (N, B, Z)
        dkl = Dkl_standard_normal(q_params, eps=self.eps) # ()
        reg_error = 0.5*self.reg_strength*dkl
        assert len(reg_error.shape) == 0, len(reg_error.shape)
        assert torch.all(torch.isfinite(q_params[ParamsFields.MEAN]))
        assert torch.all(torch.isfinite(q_params[ParamsFields.COVARIANCE]))
        assert torch.all(q_params[ParamsFields.COVARIANCE] + self.eps != 0)
        assert torch.isfinite(reg_error)
        assert torch.all(torch.isfinite(z_samples))
    
        # optional condition regularization
        if self.cond_reg_mode is None or latent_cond is None:
            cond_reg = 0.0
        elif self.cond_reg_mode == "l1":
            cond_reg = torch.mean(
                torch.sum(
                    torch.abs(latent_cond),
                    dim=-1
                )
            )
        elif self.cond_reg_mode == "l2":
            cond_reg =  torch.mean(
                torch.sqrt(
                    torch.sum(
                        torch.square(latent_cond),
                        dim=-1
                    )
                )
            )
        cond_rec_err = self.cond_reg_strength*cond_reg
        if isinstance(cond_reg, torch.Tensor):
            cond_reg = cond_reg.item()

        # decoder and reconstruction
        if latent_cond is not None:
            latent_cond=latent_cond.unsqueeze(0).repeat(self.n_mc_samples, *[1 for _ in z_samples.shape[1:]])
        p_params = self.model.get_decoded_params(
            z_samples,
            latent_condition=latent_cond 
        ) # m: (B, D), s_i: (B, 1), s_a: (B, D))
        # for k, v in p_params.items():
        #     print(k, v.shape)
        # print("x",x.shape)
        # log_px = log_prob_normal(x.unsqueeze(0).unsqueeze(0), p_params)
        log_px = log_prob_normal(x.unsqueeze(0), p_params)
        rec_error = torch.mean(log_px)

        assert len(rec_error.shape) == 0, len(rec_error.shape)
        assert torch.all(torch.isfinite(p_params[ParamsFields.MEAN]))
        assert torch.all(torch.isfinite(p_params[ParamsFields.COVARIANCE]))
        assert torch.all(p_params[ParamsFields.COVARIANCE] + self.eps != 0)
        assert torch.all(torch.isfinite(z_samples))
        assert torch.all(torch.isfinite(x))
        assert torch.isfinite(rec_error)

        # aggregating loss and returning it
        loss = - rec_error - reg_error + cond_rec_err
        return loss, {LossFields.LOSS: loss.item(), "dkl": -dkl.item(), "rec_error": -rec_error.item(), "cond_reg": cond_reg, "grad_step": step_idx}

    def _validation_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor]:
        """"""
        # predictions, target = ..., ...
        # return predictions, target
        raise NotImplementedError
