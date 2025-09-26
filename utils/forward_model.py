import logging
from collections.abc import Callable, Mapping, Sequence
import os
from typing import Any, Literal

import cloudpickle
import numpy as np
import torch

from sc_exp_design.constants import DataFields, PredictionFields
from sc_exp_design.data.dataloaders import AnnotatedPerturbationData, SequentialDataLoader
from sc_exp_design.networks.blocks import BaseForwardModel

from sc_exp_design.networks.inference_networks import PerturbationApproximatePosterior
from sc_exp_design.models.base import BaseModel
from sc_exp_design.training import BaseCallBack, TargetPredictionTrainer
from sc_exp_design.transforms import Transform

logger = logging.getLogger(__name__)

__all__ = ["ForwardModel"]


class TargetPredictionModel(BaseModel):
    """"""
    def __init__(
        self,
        state_dim: int | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        self.state_dim = state_dim
        self.device_id = device_id
        self.device = torch.device(self.device_id)
        self.target_prediction_model = None
        self.target_prediction_model_trained = False
        self.inverse_model = None

    def prepare_target_prediction_model(
        self,
        target_covariates: str | Sequence[str],
        target_covariates_dims: int | dict[str, int],
        target_covariates_noise_models: Literal["gaussian", "neg_bin"] | dict[str, None | Literal["gaussian", "neg_bin"]] | None = None,
        target_covariates_predictor_kwargs: dict[str, dict[str, Any]] | None = None,
        target_covariates_use_shared_representation: bool = False,
        target_covariates_latent_dim: int = 1024,
        target_covariates_encoder_mlp_kwargs: dict[str, Any] | None = None,
        optimizer_class: torch.optim.Optimizer = torch.optim.AdamW,
        optimizer_kwargs: Mapping[str, Any] = {"lr": 0.001},
        lr_scheduler_class: torch.optim.lr_scheduler.LRScheduler | None = None,
        lr_scheduler_kwargs: Mapping[str, Any] | None = None,
        lr_scheduler_step: Literal["grad_step", "epoch"] = "grad_step",
    ) -> None:
        """"""
        # preparing input with some sanity checks
        if isinstance(target_covariates, str):
            target_covariates = (target_covariates, )

        if isinstance(target_covariates_dims, int):
            msg = f"When `target_covariates_dims` is of type `int`, the respective perturbations should contain only one element, found {len(target_covariates)}"
            assert len(target_covariates) == 1, msg
            target_covariates_dims = {target_covariates[0]: target_covariates_dims}
        
        if isinstance(target_covariates_noise_models, str):
            msg = f"When `target_covariates_noise_models` is of type `str`, the respective perturbations should contain only one element, found {len(target_covariates)}"
            assert len(target_covariates) == 1, msg
            target_covariates_noise_models = {target_covariates[0]: target_covariates_noise_models}
        if target_covariates_noise_models is None:
            target_covariates_noise_models = {target_covariate: None for target_covariate in target_covariates}

        if target_covariates_predictor_kwargs is None:
            target_covariates_predictor_kwargs = {
                target_covariate: {} for target_covariate in target_covariates
            }

        # checking types
        msg = f""
        assert isinstance(target_covariates, Sequence), msg

        msg = f""
        assert isinstance(target_covariates_dims, dict), msg

        msg = f"{target_covariates_noise_models=}, {type(target_covariates_noise_models)=}"
        assert isinstance(target_covariates_noise_models, dict), msg

        msg = f""
        assert isinstance(target_covariates_predictor_kwargs, dict), msg

        # storing the settings here as attributes
        self.target_covariates = target_covariates
        self.target_covariates_dims = target_covariates_dims
        self.target_covariates_noise_models = target_covariates_noise_models
        self.target_covariates_predictor_kwargs = target_covariates_predictor_kwargs
        self.target_covariates_use_shared_representation = target_covariates_use_shared_representation
        self.target_covariates_latent_dim = target_covariates_latent_dim
        self.target_covariates_encoder_mlp_kwargs = target_covariates_encoder_mlp_kwargs

        # initializing the predictor for each target covariate
        self.target_prediction_model = PerturbationApproximatePosterior(
            self.state_dim,
            freeze_grads=False, # we want to backpropagate the gradients from its input
            target_output_dims=self.target_covariates_dims,
            noise_models=self.target_covariates_noise_models,
            covariate_kwargs=self.target_covariates_predictor_kwargs,
            use_shared_representation=self.target_covariates_use_shared_representation,
            latent_dim=self.target_covariates_latent_dim,
            encoder_mlp_kwargs=self.target_covariates_encoder_mlp_kwargs,
        )
        self.target_prediction_model = self.target_prediction_model.float()
        self.target_prediction_model = self.target_prediction_model.to(self.device)

        # optimizer and scheduler 
        self.target_prediction_optimizer = optimizer_class(
            self.target_prediction_model.parameters(),
            **optimizer_kwargs,
        )

        self.target_prediction_lr_scheduler = None
        self.target_prediction_lr_scheduler_step = None
        if lr_scheduler_kwargs is None:
            lr_scheduler_kwargs = {}
        if lr_scheduler_class is not None:
            self.target_prediction_lr_scheduler = lr_scheduler_class(self.target_prediction_optimizer, **lr_scheduler_kwargs)
            self.target_prediction_lr_scheduler_step = lr_scheduler_step

    def train_target_prediction_model(
        self,
        train_data: AnnotatedPerturbationData | None = None,
        validation_data: AnnotatedPerturbationData | None = None,
        num_training_steps: int = 500,
        valid_freq: int | None = None,
        train_batch_size: int = 1024,
        validation_batch_size: int = 512,
        state_transforms: Transform | None = None,
        callbacks: BaseCallBack | None = None,
        grad_steps_log_interval: int = 100,
    ) -> None:
        """"""
        # sanity checks
        msg = f"You need to have instantitated the target predictor model by calling `prepare_target_prediction_model`"
        assert self.target_prediction_model is not None, msg

        if train_data is None:
            msg = f""
            assert self.forward_model is not None, msg
            train_data = self.forward_model.train_data

        msg = f""
        assert isinstance(train_data, AnnotatedPerturbationData), msg

        msg = f""
        assert train_data.target_data is not None, msg

        # initializing data loader
        self.target_predictor_train_data = train_data
        self.target_predictor_train_dataloader = SequentialDataLoader(
            self.target_predictor_train_data,
            train_batch_size,
            state_transforms=state_transforms,
            device_id=self.device_id
        )

        # initialize trainer
        self.target_predictor_trainer = TargetPredictionTrainer(
            self.target_prediction_model,
            self.target_prediction_optimizer,
            lr_scheduler=self.target_prediction_lr_scheduler,
            lr_scheduler_step=self.target_prediction_lr_scheduler_step,
            callbacks=callbacks,
            grad_steps_log_interval=grad_steps_log_interval,
        )

        # optional validation data
        self.target_predictor_validation_dataloader = None
        self.target_predictor_validation_data = validation_data
        if validation_data is not None:

            self.target_predictor_validation_dataloader = SequentialDataLoader(
                self.target_predictor_validation_data,
                validation_batch_size,
                state_transforms=state_transforms,
                device_id=self.device_id,
            )

        # fitting the trainer
        self.target_predictor_trainer.fit(
            num_training_steps,
            self.target_predictor_train_dataloader,
            self.target_predictor_validation_dataloader,
            valid_freq,
        )

        self.target_prediction_model_trained = True

    def save_target_prediction_model(
        self,        
        dump_dir: str,
        model_prefix: str | None = None,
        overwrite: bool = False,
    ) -> None:
        """"""
        # construct file name
        if model_prefix is None:
            model_prefix = ""
        else:
            model_prefix = f"{model_prefix}_"
        file_name = f"{model_prefix}{self.target_prediction_model.__class__.__name__}.pkl"

        # defining path
        dump_path = os.path.join(dump_dir, file_name)

        # checking that the file exists
        if os.path.exists(dump_path):
            if not overwrite:
                msg = f""
                raise RuntimeError(msg)
            msg = f""
            logger.warning(msg)

        # saving the model
        with open(dump_path, "wb") as fp:
            cloudpickle.dump(self.target_prediction_model, fp)

    def load_target_prediction_model(
        self,
        file_name: str,
    ) -> None:
        """"""
        # loading model file
        with open(file_name, "rb") as fp:
            model = cloudpickle.load(fp)
        # veriying types
        if type(model) is not PerturbationApproximatePosterior:
            msg = f""
            raise TypeError(msg)
        self.target_prediction_model = model        

    def predict(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],

        **kwargs,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """"""
        # forward pass on perturbation response prediction model
        target_logits = self.target_prediction_model(batch[DataFields.TARGET_STATE], **kwargs)
        # constructing step output dictionary
        out_dict =  {
            covariate: covariate_pred for covariate, covariate_pred in target_logits.items()
        }
        return out_dict
    