from collections.abc import Sequence
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch


def parse_mlp_config_dictionary(
    activation_functions: dict[str, torch.optim.Optimizer], 
    mlp_config_dictionary: dict[str, Any] | None,
    separator: str = "-"
) -> dict[str, Any]:
    """"""
    output_dict = {}
    if mlp_config_dictionary is None:
        return output_dict
    for key, value in mlp_config_dictionary.items():
        if key == "activation_class":
            value = activation_functions.get(value, torch.nn.ReLU)
        elif key == "final_activation_class":
            value = activation_functions.get(value, torch.nn.Identity)
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


def parse_nested_mlp_config_dictionary(
    activation_functions: dict[str, torch.optim.Optimizer], 
    mlp_config_dictionary: dict[str, Any] | None,
    separator: str = "-",
):
    mlp_config_dictionary_copy = {}
    for perturbation, perturbation_kwargs in mlp_config_dictionary.items():
        perturbation_kwargs = parse_mlp_config_dictionary(activation_functions, perturbation_kwargs, separator=separator)
        mlp_config_dictionary_copy[perturbation] = perturbation_kwargs
    return mlp_config_dictionary_copy


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
