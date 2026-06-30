from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Reusable multi-layer perceptron.

    Hidden layers: Linear -> BatchNorm1d (optional) -> Activation -> Dropout (if > 0).
    Final layer:   Linear -> output_activation (optional).
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
        activation: type[nn.Module] = nn.PReLU,
        output_activation: type[nn.Module] | None = None,
    ):
        super().__init__()

        layers: list[nn.Module] = []

        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

            if i < len(layer_sizes) - 1:
                if use_batch_norm:
                    layers.append(
                        nn.BatchNorm1d(
                            layer_sizes[i],
                        )
                    )
                layers.append(activation())
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
            else:
                if output_activation is not None:
                    layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

