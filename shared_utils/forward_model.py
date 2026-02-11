import logging
from collections.abc import Callable, Mapping, Sequence
import os
from typing import Any, Literal

import cloudpickle
import numpy as np
import torch

from sc_exp_design.constants import PredictionFields
from sc_exp_design.networks.blocks import BaseForwardModel

from sc_exp_design.models.base import BaseModel

logger = logging.getLogger(__name__)

__all__ = ["ForwardModel"]


class ForwardModel(BaseModel):
    """"""
    def __init__(
        self,
        forward_model: BaseForwardModel | None = None,
        target_prediction_model = None,
        state_dim: int | None = None,
        device_id: Literal["cuda", "cpu"] = "cuda",
    ) -> None:
        """"""
        # sanity check on the input 
        if state_dim is None:
            msg = f""
            assert forward_model is not None, msg
            state_dim = forward_model.cvf_config.flow_dim
        else:
            msg = f""
            assert isinstance(state_dim, int), msg
            # if we pass the forward model we take the dimensionality from there
            if (forward_model is not None) and state_dim != forward_model.cvf_config.flow_dim:
                msg = f""
                logger.warning(msg)
                state_dim = forward_model.cvf_config.flow_dim

        self.forward_model = forward_model
        self.target_prediction_model = target_prediction_model
        self.state_dim = state_dim

        self.device_id = device_id
        self.device = torch.device(self.device_id)

    def predict(
        self,
        batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        return_trajectory: bool = False,
        no_grad: bool = True,
        num_samples: int | None = None,
        batch_size: int | None = None,
        num_time_steps: int | None = None,
        solver_kwargs: dict[str, Any] | None = None,
        cfg_guidance_strength: float = 1.0,
        fix_noise: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """"""
        # forward pass on perturbation response prediction model
        x1_hat = self.forward_model.predict(
            batch,
            return_trajectory=return_trajectory,
            no_grad=no_grad,
            num_samples=num_samples,
            batch_size=batch_size,
            num_time_steps=num_time_steps,
            fix_noise=fix_noise,
            solver_kwargs=solver_kwargs,
            cfg_guidance_strength=cfg_guidance_strength,
        )

        # forward pass on target predictor model
        target_logits = self.target_prediction_model(x1_hat, **kwargs)

        # constructing step output dictionary
        out_dict = {
            PredictionFields.PREDICTION_DATA: x1_hat,
            PredictionFields.TARGET_PREDICTION_DATA: {
                covariate: covariate_pred for covariate, covariate_pred in target_logits.items()
            },
        }
        return out_dict
    