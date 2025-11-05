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
from sc_exp_design.networks import ConditionEncoder, MLPBlock, MLPGaussianNoiseModel
from sc_exp_design.training.base import BaseTrainer
from sc_exp_design.training.callbacks import BaseCallBack

from vae_utils import compute_mmd, log_prob_normal, Dkl_standard_normal

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
        state_encoder_output_dim: int = 8,
        condition_encoder_output_dim: int = 2,
        joint_latent_dim: int | None = 4,
        decoder_output_dim: int = 4,
        encoder_mlp_kwargs=None,
        decoder_mlp_kwargs=None,
        layers_before_pooling: dict | None = None,
        layers_after_pooling: dict | None = None,
        is_conditional: bool = True,
    ):
        if is_conditional:
            assert layers_before_pooling is not None
            assert layers_after_pooling is not None

        super().__init__()
        self.state_dim = state_dim
        self.state_encoder_output_dim = state_encoder_output_dim
        self.condition_encoder_output_dim = condition_encoder_output_dim
        self.joint_latent_dim = joint_latent_dim
        self.decoder_output_dim = decoder_output_dim
        self.encoder_mlp_kwargs = {} if encoder_mlp_kwargs is None else encoder_mlp_kwargs
        self.decoder_mlp_kwargs = {} if decoder_mlp_kwargs is None else decoder_mlp_kwargs
        self.layers_before_pooling = layers_before_pooling
        self.layers_after_pooling = layers_after_pooling
        self.is_conditional = is_conditional
        self.vae_modules = self._init_modules()

    def _init_state_encoder(self):
        return MLPBlock(
            self.state_dim,
            self.state_encoder_output_dim,
            **self.encoder_mlp_kwargs,
        )

    def _init_condition_encoder(self):
        return ConditionEncoder(
            self.condition_encoder_output_dim,
            layers_before_pooling=self.layers_before_pooling,
            layers_after_pooling=self.layers_after_pooling,
        )

    def _init_encoder_proj(self):
        input_dim = self.state_encoder_output_dim
        if self.is_conditional:
            input_dim = input_dim + self.condition_encoder_output_dim
        return MLPGaussianNoiseModel(
            input_dim,
            self.joint_latent_dim,
            cov_estimation_mode="anisotropic",
            use_shared_representation=False,
            mean_mlp_kwargs=self.linear_proj_config,
            cov_mlp_kwargs=self.linear_proj_config,
        )

    def _init_decoder(self):
        return MLPBlock(
            self.joint_latent_dim,
            self.decoder_output_dim,
            **self.encoder_mlp_kwargs,
        )

    def _init_decoder_proj(self):
        input_dim = self.decoder_output_dim
        if self.is_conditional:
            input_dim = input_dim + self.condition_encoder_output_dim
        return MLPGaussianNoiseModel(
            input_dim,
            self.state_dim,
            cov_estimation_mode="anisotropic",
            use_shared_representation=False,
            mean_mlp_kwargs=self.linear_proj_config,
            cov_mlp_kwargs=self.linear_proj_config,
        )

    def _init_modules(self):
        modules = {}
        if self.is_conditional:
            modules["condition_encoder"] = self._init_condition_encoder()
        modules["state_encoder"] = self._init_state_encoder()
        modules["encoder_proj"] = self._init_encoder_proj()
        modules["decoder"] = self._init_decoder()
        modules["decoder_proj"] = self._init_decoder_proj()
        return torch.nn.ModuleDict(modules)

    def get_latent_params(
        self,
        x,
        latent_condition=None,
    ):
        x = self.vae_modules["state_encoder"](x)
        # handling input for encoder
        if self.is_conditional:
            assert latent_condition is not None
            encoder_input = torch.concatenate((x, latent_condition), dim=-1)
        else:
            encoder_input = x
        return self.vae_modules["encoder_proj"](encoder_input)

    def get_decoded_params(
        self,
        z,
        latent_condition=None,
    ):
        z = self.vae_modules["decoder"](z)
        # handling input for encoder
        if self.is_conditional:
            assert latent_condition is not None
            decoder_input = torch.concatenate((z, latent_condition), dim=-1)
        else:
            decoder_input = z
        return self.vae_modules["decoder_proj"](decoder_input)

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

    @staticmethod
    def sample_from_params(N, params, return_noise=False):
        # parsing parameter dictionary
        mean = params[ParamsFields.MEAN]
        std = params[ParamsFields.COVARIANCE]

        # sampling noise and matching shapes
        noise = torch.randn((N, *mean.shape), device=mean.device, dtype=torch.float)
        mean = mean.unsqueeze(0)
        cov = torch.exp(2*std.unsqueeze(0))
        samples =  mean + torch.sqrt(cov)*noise
        if return_noise:
            return samples, noise
        else:
            return samples


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
        deterministic_reconstruction = False,
        use_reg_linear_schedule = True,
        reg_schedule_num_steps = 100,
        reg_mode: Literal["dkl", "mmd"] = "kldiv",
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
        self.deterministic_reconstruction = deterministic_reconstruction
        self.use_reg_linear_schedule = use_reg_linear_schedule
        self.reg_schedule_num_steps = reg_schedule_num_steps
        self.reg_mode = reg_mode

    def _compute_reg_strength_schedule(
        self,
        step_idx,
    ):
        if self.use_reg_linear_schedule and step_idx<self.reg_schedule_num_steps:
            return (step_idx/ self.reg_schedule_num_steps)*self.reg_strength
        else:
            return self.reg_strength

    def _compute_reconstruction_loss(
        self,
        x,
        p_params,
    ):
        if self.deterministic_reconstruction:
            recon = p_params[ParamsFields.MEAN]
            return torch.mean((x - recon)**2)
        else:
            log_px = log_prob_normal(x.unsqueeze(0), p_params)
            return -torch.mean(log_px)

    def _compute_latent_cond_regularization(
        self,
        latent_cond,
    ):
        # optional condition regularization
        if self.cond_reg_mode is None or latent_cond is None:
            return 0.0
        elif self.cond_reg_mode == "l1":
            return torch.mean(
                torch.sum(
                    torch.abs(latent_cond),
                    dim=-1
                )
            )
        elif self.cond_reg_mode == "l2":
            return torch.mean(
                torch.sqrt(
                    torch.sum(
                        torch.square(latent_cond),
                        dim=-1
                    )
                )
            )

    def _compute_regularization(
        self,
        q_params,
        z_samples, 
        noise,
    ):
        if self.reg_mode == "kldiv":
            return Dkl_standard_normal(q_params, eps=self.eps)
        elif self.reg_mode == "mmd":
            return -compute_mmd(z_samples.reshape(-1, 1), noise.reshape(-1, 1))
        raise ValueError

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
        z_samples, noise = self.model.sample_from_params(self.n_mc_samples, q_params, return_noise=True) # m: (N, B, Z)
        reg = self._compute_regularization(q_params, z_samples, noise)
        reg_strength = self._compute_reg_strength_schedule(step_idx)
        reg_error = reg_strength*reg

        # optional condition regularization
        cond_reg = self.cond_reg_strength*self._compute_latent_cond_regularization(latent_cond)

        # decoder and reconstruction
        if latent_cond is not None:
            latent_cond=latent_cond.unsqueeze(0).repeat(self.n_mc_samples, *[1 for _ in z_samples.shape[1:]])
        p_params = self.model.get_decoded_params(
            z_samples,
            latent_condition=latent_cond 
        )
        rec_error = self._compute_reconstruction_loss(x, p_params)

        # aggregating loss and returning it
        loss = rec_error + reg_error + cond_reg
        return loss, {LossFields.LOSS: loss.item(), "reg": reg_error.item(), "rec_error": rec_error.item(), "cond_reg": cond_reg, "grad_step": step_idx}

    def _validation_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor]:
        """"""
        # predictions, target = ..., ...
        # return predictions, target
        raise NotImplementedError
