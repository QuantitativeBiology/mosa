from __future__ import annotations

import torch
import torch.nn as nn


class MaskedBatchNorm1d(nn.Module):
    """BatchNorm1d that computes statistics only from present samples.

    When a boolean ``mask`` is provided to :meth:`forward`, mean and variance
    are computed exclusively over the samples where ``mask`` is ``True``.
    All samples are then normalised with those statistics so the output is
    well-defined everywhere, but running statistics and affine-parameter
    gradients reflect only the present data.

    In distributed training the masked sums are reduced across all ranks via
    ``torch.distributed.all_reduce``.

    When no mask is supplied the layer behaves identically to
    ``nn.BatchNorm1d``.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    @staticmethod
    def _sync_stats(
        total: torch.Tensor, total_sq: torch.Tensor, count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-reduce masked statistics across distributed ranks."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return total, total_sq, count
        torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(total_sq, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        return total, total_sq, count


    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Normalise *x* using statistics derived from present samples only.

        Parameters
        ----------
        x : Tensor [B, C]
            Input features.
        mask : Tensor [B], optional
            Boolean — ``True`` for samples that should contribute to batch
            statistics.  ``None`` means all samples are present.
        """
        # Eval mode always uses running stats regardless of mask.
        if not self.training and self.track_running_stats:
            return nn.functional.batch_norm(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, False, 0.0, self.eps,
            )

        # Fast path: no mask or every sample present.
        if mask is None or mask.all():
            return nn.functional.batch_norm(
                x,
                self.running_mean if self.track_running_stats else None,
                self.running_var if self.track_running_stats else None,
                self.weight, self.bias, True, self.momentum, self.eps,
            )

        # Masked statistics 
        mask_f = mask.float().unsqueeze(1)              # [B, 1]
        count = mask_f.sum()                             # scalar
        masked_x = x * mask_f                            # zero absent rows
        total = masked_x.sum(dim=0)                      # [C]
        total_sq = (masked_x * masked_x).sum(dim=0)      # [C]

        total, total_sq, count = self._sync_stats(total, total_sq, count)

        # Fall back to running stats when fewer than 2 samples are present
        # across all ranks (cannot compute meaningful variance).
        if count < 2:
            if self.track_running_stats and self.running_mean is not None:
                mean, var = self.running_mean, self.running_var
            else:
                return x
        else:
            mean = total / count
            var = total_sq / count - mean * mean

        # Update running statistics from masked batch stats.
        if self.track_running_stats:
            with torch.no_grad():
                self.num_batches_tracked += 1
                self.running_mean.mul_(1 - self.momentum).add_(mean, alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var, alpha=self.momentum)

        x = (x - mean) / (var + self.eps).sqrt()

        if self.affine:
            x = x * self.weight + self.bias

        return x

    def extra_repr(self) -> str:
        return (
            f"{self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class MLP(nn.Module):
    """Reusable multi-layer perceptron with mask-aware batch normalisation.

    Hidden layers: Linear -> MaskedBatchNorm1d (optional) -> Dropout (if > 0) -> Activation.
    Final layer:   Linear -> output_activation (optional).

    When a ``mask`` is passed to :meth:`forward`, it is forwarded to every
    :class:`MaskedBatchNorm1d` layer so that batch statistics are computed
    only from present samples.  This is required for correct multi-GPU
    training when some samples may be absent for a given data view.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        dropout_p: float = 0.1,
        use_batch_norm: bool = True,
        activation: type[nn.Module] = nn.PReLU,
        output_activation: type[nn.Module] | None = None,
        bn_momentum: float = 0.1,
        bn_eps: float = 1e-5,
    ):
        super().__init__()

        layers: list[nn.Module] = []

        for i in range(1, len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i - 1], layer_sizes[i]))

            if i < len(layer_sizes) - 1:
                if use_batch_norm:
                    layers.append(
                        MaskedBatchNorm1d(
                            layer_sizes[i], momentum=bn_momentum, eps=bn_eps,
                        )
                    )
                if dropout_p > 0.0:
                    layers.append(nn.Dropout(p=dropout_p))
                layers.append(activation())
            else:
                if output_activation is not None:
                    layers.append(output_activation())

        self.net = nn.ModuleList(layers)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pass input through all layers, threading *mask* to BN layers."""
        for layer in self.net:
            if isinstance(layer, MaskedBatchNorm1d):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x

