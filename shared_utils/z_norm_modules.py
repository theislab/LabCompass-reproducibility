from typing import Literal, Type

import numpy as np
import scanpy as sc
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


class BaseWhitening(torch.nn.Module):
    def __init__(self, params: TensorParams):
        super().__init__()
        self.params = torch.nn.ModuleDict(
            {
                "mean": ModelParam(params["mean"]),
                "W": ModelParam(params["W"]),
                "iW": ModelParam(params["iW"]),
            }
        )

    @property
    def mean(self) -> torch.nn.Parameter:
        return self.params["mean"]

    @property
    def W(self) -> torch.nn.Parameter:
        return self.params["W"]

    @property
    def iW(self) -> torch.nn.Parameter:
        return self.params["iW"]


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


class Whitening(BaseWhitening):
    def __init__(self, params: TensorParams):
        super().__init__(params=params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return torch.einsum("...n,mn->...m", (x - self.mean), self.W)


class IWhitening(BaseWhitening):
    def __init__(self, params: TensorParams):
        super().__init__(params=params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return torch.einsum("...n,mn->...m", x, self.iW) +  self.mean


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


def get_rescaling(
    adata: sc.AnnData,
    channel_params_key: str = "X_channel_params", 
    scatter_params_key: str = "X_scatter_params",
    inverse: bool = False,
) -> ZNorm | IZNorm:

    X_channel_params_fwd = adata.uns[channel_params_key]
    X_scatter_params_fwd = adata.uns[scatter_params_key]
    params = {
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
    if inverse:
        return IZNorm(params)
    return ZNorm(params)
