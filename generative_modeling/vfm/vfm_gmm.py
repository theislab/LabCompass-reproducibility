from collections.abc import Callable
import dataclasses
from dataclasses import field as dc_field
from typing import Any

import torch

from sc_exp_design.config import NeuralVelocityFieldConfig
from sc_exp_design.networks import MLPBlock, NeuralVelocityField


@dataclasses.dataclass
class GMMNeuralVelocityFieldConfig(NeuralVelocityFieldConfig):
    """"""
    n_comps: int
    weight_decoder_mlp_kwargs: dict[str, Any] = dc_field(default_factory=lambda: {})


class NeuralGMMVelocityField(NeuralVelocityField):
    """"""
    def __init__(
        self,
        config: GMMNeuralVelocityFieldConfig,
    ) -> None:
        """"""
        super().__init__(config)
    
    def _init_modules(
        self,
    ) -> None:
        """"""
        super()._init_modules()

        # overwrite decoder
        self.decoder = {
            comp: MLPBlock(
                self.config.decoder_input_dim,
                self.config.flow_dim,
                **self.config.decoder_mlp_kwargs
            ) for comp in range(self.n_comps)
        }

        # weight decoder
        self.weight_decoder = MLPBlock(
            self.config.decoder_input_dim,
            self.n_comps,
            **self.config.weight_decoder_mlp_kwargs,
        )

    def forward(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        cond: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        """
        # retrieving latent concat
        latent_concat = self.get_latent_concat(
            t,
            xt,
            cond=cond,
            source=source,
        )

        return {
            "mean": {decoder(latent_concat) for comp, decoder in self.decoder.items()},
            "weight_logits": self.weight_decoder(latent_concat),
        }
    
    def get_vf_fn(
        self,
        cond: dict[str, torch.Tensor] | None = None,
        source: torch.Tensor | None = None,
        cfg_guidance_strength: float = 1.0,
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Returns a velocity field function.

        Args:
            cond (dict[str, Tensor] | None, optional): Conditioning variables. Defaults to None.
            gamma_fn (Callable[[Tensor, Tensor], Tensor] | None, optional): Function for computing diffusion coefficient. Defaults to None.
        
        Returns:
            Callable[[Tensor, Tensor], Tensor]: Velocity field function.
        """
        # sanity checks
        if self.config.use_source_as_condition:
            msg = f""
            assert source is not None, msg

        def vf_fn(
            t: torch.Tensor,
            xt: torch.Tensor,
        ) -> torch.Tensor:
            """"""
            # when using cfg
            if self.config.use_classifier_free_guidance:
                # get null condition token
                null_condition_token = self.get_null_condition_token(cond)
                # computing unguided and guided velocity fields
                vf_unguided = self.forward(t, xt, cond=null_condition_token, source=source)
                vf_guided = self.forward(t, xt, cond=cond, source=source)
                # computing the final velocity field
                vf = vf_unguided + cfg_guidance_strength * (vf_guided - vf_unguided)
                return vf
            # when not using cfg
            return self.forward(t, xt, cond=cond, source=source)

        return vf_fn
