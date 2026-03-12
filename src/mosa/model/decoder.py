from __future__ import annotations

import torch
import torch.nn as nn

from mosa.model.mlp import MLP


class OmicDecoder(nn.Module):
    """Per-view decoder that reconstructs omic features from the joint latent.

    Architecture: ``[joint_latent_dim + cond_dim] -> reversed hidden_dims -> output_dim``.

    The conditional input passes through a dedicated projection layer
    (Linear + BatchNorm + PReLU) before concatenation with z. This allows
    the decoder to learn a non-linear transformation of the metadata,
    rather than feeding raw one-hot vectors directly into the main MLP.
    """

    def __init__(
        self,
        output_dim: int,
        cond_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        reversed_dims = list(reversed(hidden_dims))

        self.has_cond = cond_dim > 0
        if self.has_cond:
            self.cond_fc = nn.Linear(cond_dim, cond_dim)
            self.cond_bn = nn.BatchNorm1d(cond_dim, momentum=0.01, eps=0.001)
            self.cond_act = nn.PReLU()

        self.net = MLP(
            layer_sizes=[latent_dim + cond_dim] + reversed_dims + [output_dim],
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
            activation=nn.PReLU,
            output_activation=None,
        )

    def forward(self, z: torch.Tensor, conditionals: torch.Tensor) -> torch.Tensor:
        """Decode the joint latent vector into reconstructed omic features.

        Parameters
        ----------
        z : Tensor [B, joint_latent_dim]
            Joint latent representation.
        conditionals : Tensor [B, cond_dim]
            Conditional metadata.

        Returns
        -------
        Tensor [B, output_dim]
            Reconstructed omic features.
        """
        if self.has_cond:
            b = self.cond_fc(conditionals)
            b = self.cond_bn(b)
            b = self.cond_act(b)
            h = torch.cat([z, b], dim=1)
        else:
            h = z
        return self.net(h)
