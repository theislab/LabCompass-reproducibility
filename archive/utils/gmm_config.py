from dataclasses import dataclass, field as dc_field
from functools import partial
import os
from typing import Any

from sc_exp_design.types import MLPConfigFields
from sc_exp_design.config import NeuralVelocityFieldConfig

@dataclass
class GMMNeuralVelocityFieldConfig(NeuralVelocityFieldConfig):
    weight_decoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {},)
    gmm_decoders_mlp_kwargs: dict[int, dict[str, Any]] = dc_field(default_factory=lambda: {},)

    def __post_init__(self):
        super().__post_init__()
    
        # sanity check on mlp configurations
        mlp_kwargs_verifier = partial(
            MLPConfigFields.verify_keys, 
            require_input_dim_key=False,
            require_output_dim_key=False,
        )
        for mlp_kwargs in self.gmm_decoders_mlp_kwargs.values():
            mlp_kwargs_verifier(mlp_kwargs)
    
    @property
    def num_gmm_comps(self) -> int:
        return len(self.gmm_decoders_mlp_kwargs)
