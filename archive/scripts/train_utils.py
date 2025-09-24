from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig, OmegaConf

import torch

__all__ = [
    "parse_mlp_config_dictionary",
    "resolve_omegaconf_to_dictionary",
    "TimeSamplers",
    "Optimizers",
    "LRSchedulers",
    "STateTransforms",
    "ActivationFunctions"
]


def parse_mlp_config_dictionary(
    mlp_config_dictionary: dict[str, Any],
    separator: str = "-"
) -> dict[str, Any]:
    """"""
    output_dict = {}
    for key, value in mlp_config_dictionary.items():
        if key == "activation_class":
            value = vars(ActivationFunctions())[value]
        elif key == "final_activation_class":
            value = vars(ActivationFunctions())[value]
        if key == "hidden_dims":
            # single integer
            if isinstance(value, int):
                value = (value, )
            # string with separator
            if isinstance(value, str):
                value = value.split(separator)
                value = [int(val) for val in value]
            # sanity check
            msg = f""
            assert isinstance(value, Sequence), msg
            for val in value:
                msg = f""
                assert isinstance(val, int), msg 
        output_dict[key] = value
    return output_dict


def resolve_omegaconf_to_dictionary(
    conf_dict: dict[str, Any] | None | DictConfig 
) -> dict[str, Any]:
    """"""
    out_dict = {}
    if conf_dict is not None:
        out_dict = conf_dict
        if isinstance(out_dict, dict):
            out_dict = OmegaConf.create(out_dict)
        out_dict = OmegaConf.to_container(out_dict, resolve=True)
    return out_dict


@dataclass(frozen=True)
class TimeSamplers:
    uniform: Callable[[Sequence[int]], torch.Tensor] = torch.rand


@dataclass(frozen=True)
class NoiseDistributions:
    gaussian: Callable[[Sequence[int]], torch.Tensor] = torch.randn


@dataclass(frozen=True)
class Optimizers:
    adam: torch.optim.Optimizer = torch.optim.Adam
    adam_w: torch.optim.Optimizer = torch.optim.AdamW


@dataclass(frozen=True)
class LRSchedulers:
    none: None = None


@dataclass(frozen=True)
class StateTransforms:
    none: None = None


@dataclass(frozen=True)
class ActivationFunctions:
    selu: torch.nn.Module = torch.nn.SELU
    identity: torch.nn.Module = torch.nn.Identity