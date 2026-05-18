from __future__ import annotations

import torch
import torch.nn as nn

from mosa.models.mosa.vae.layers import MLP


class Discriminator(nn.Module):
    """Predicts model_type from latent vectors for adversarial training.

    Architecture: three hidden layers of size ``enc_dim`` with ReLU and BatchNorm,
    followed by a linear output layer with ``output_dim`` classes.
    """

    def __init__(self, enc_dim: int, output_dim: int):
        super().__init__()
        self.net = MLP(
            layer_sizes=[enc_dim, enc_dim, enc_dim, output_dim],
            dropout_p=0.0,
            use_batch_norm=True,
            activation=nn.ReLU,
            output_activation=None,
            bn_momentum=0.01,
            bn_eps=0.001,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict model_type logits from latent vector.

        Parameters
        ----------
        x : Tensor [B, enc_dim]
            Latent representations.

        Returns
        -------
        Tensor [B, output_dim]
            Unnormalized class logits.
        """
        return self.net(x)
