from typing import Literal, Type

import torch

from sc_exp_design.constants import DataFields, ParamsFields
from sc_exp_design.models import TargetPredictionModel


TensorParams = dict[Literal[ParamsFields.MEAN, ParamsFields.COVARIANCE], torch.Tensor]


class ModelParam(torch.nn.Module):
    def __init__(self, data: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.data = torch.nn.Parameter(data, requires_grad=True).to(device)


class BaseZNorm(torch.nn.Module):
    def __init__(self, params: TensorParams):
        super().__init__()
        self.params = torch.nn.ModuleDict(
            {
                ParamsFields.MEAN: ModelParam(params[ParamsFields.MEAN]),
                ParamsFields.COVARIANCE: ModelParam(params[ParamsFields.COVARIANCE]),
            }
        )

    @property
    def mean(self) -> torch.nn.Parameter:
        return self.params[ParamsFields.MEAN]

    @property
    def cov(self) -> torch.nn.Parameter:
        return self.params[ParamsFields.COVARIANCE]


class ZNorm(BaseZNorm):
    def __init__(self, params: TensorParams):
        super().__init__(params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return (x - self.params[ParamsFields.MEAN].data)/self.params[ParamsFields.COVARIANCE].data


class IZNorm(BaseZNorm):
    def __init__(self, params: TensorParams):
        super().__init__(params=params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return x*self.params[ParamsFields.COVARIANCE].data + self.params[ParamsFields.MEAN].data


class RescaledTargetPredictionModel(torch.nn.Module):
    def __init__(
        self,
        model: TargetPredictionModel,
        inv_params: IZNorm,
        fwd_params: ZNorm,
    ) -> None:
        """"""
        super().__init__()
        self.resc_model = torch.nn.ModuleDict(
            {
                "inv_params": inv_params,
                "fwd_params": fwd_params,
                "model": model,
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.resc_model["inv_params"](x)
        x = self.resc_model["fwd_params"](x)
        return self.resc_model["model"](x)
