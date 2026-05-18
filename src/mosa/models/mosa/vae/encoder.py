from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mosa.models.mosa.vae.layers import MLP


class ViewDropout(nn.Module):
    """Randomly zeros the input tensor during training with probability ``p``."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply view-level dropout: zero the entire input with probability p."""
        if self.training and torch.rand(1, device=x.device).item() < self.p:
            x = x.clone()
            x.zero_()
        return x


class OmicEncoder(nn.Module):
    """Maps omic features and conditionals to a per-view embedding.

    Architecture: ``[input_dim + cond_dim] -> hidden_dims -> view_latent_dim``
    with BatchNorm, PReLU, and dropout at each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        cond_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        dropout_p: float = 0.1,
        view_dropout_p: float = 0.0,
        use_batch_norm: bool = True,
    ):
        super().__init__()

        self.view_dropout = ViewDropout(p=view_dropout_p) if view_dropout_p > 0 else None

        self.net = MLP(
            layer_sizes=[input_dim + cond_dim] + hidden_dims + [latent_dim],
            dropout_p=dropout_p,
            use_batch_norm=use_batch_norm,
            activation=nn.PReLU,
            output_activation=nn.PReLU,
        )

    def forward(
        self,
        x: torch.Tensor,
        conditionals: torch.Tensor,
    ) -> torch.Tensor:
        """Encode omic features with conditional metadata.

        Parameters
        ----------
        x : Tensor [B, input_dim]
            Omic feature values for this view.
        conditionals : Tensor [B, cond_dim]
            Conditional metadata (model_type, tissue, mutations).

        Returns
        -------
        Tensor [B, view_latent_dim]
            Per-view embedding.
        """
        if self.view_dropout is not None:
            x = self.view_dropout(x)
        h = torch.cat([x, conditionals], dim=1)
        return self.net(h)
