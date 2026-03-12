from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal, kl_divergence as kl_div
from torch.nn import CrossEntropyLoss


def reconstruction_loss(
    x_hat: dict[str, Tensor],
    x: dict[str, Tensor],
    mask: dict[str, Tensor],
    group: Tensor | None = None,
    sample_weights: Tensor | None = None,
    loss_type: str = "mean",
) -> tuple[Tensor, dict]:
    """Feature-masked reconstruction loss.

    Parameters
    ----------
    x_hat : dict of Tensor [B, D]
    x : dict of Tensor [B, D]
    mask : dict of bool Tensor [B, D]
    group : Tensor [B] or None
        Group labels for balanced loss.
    sample_weights : Tensor [B] or None
    loss_type : str
        "mean" for standard masked MSE, "macro" for group-balanced.

    Returns
    -------
    loss : scalar Tensor
    metrics : dict with per-omic losses
    """
    device = next(iter(x.values())).device
    omic_losses = {}
    group_omic_losses = defaultdict(dict)

    for omic in x:
        feature_mask = mask[omic]  # [B, D]
        recon = x_hat[omic]  # [B, D]
        target = x[omic]  # [B, D]

        mse_per_feature = F.mse_loss(recon, target, reduction="none")  # [B, D]
        mse_masked = mse_per_feature * feature_mask.float()

        n_present = feature_mask.sum(dim=1).clamp(min=1)  # [B]
        per_sample = mse_masked.sum(dim=1) / n_present  # [B]

        if sample_weights is not None:
            per_sample = per_sample * sample_weights

        if loss_type == "macro" and group is not None:
            sample_mask = feature_mask.any(dim=1)
            if not sample_mask.any():
                continue

            per_sample_valid = per_sample[sample_mask]
            group_valid = group[sample_mask]
            unique_groups = torch.unique(group_valid)
            group_losses = []

            for g in unique_groups:
                g_idx = group_valid == g
                if g_idx.any():
                    g_loss = per_sample_valid[g_idx].mean()
                    group_losses.append(g_loss)
                    group_omic_losses[omic][g.item()] = g_loss

            if group_losses:
                omic_losses[omic] = torch.stack(group_losses).mean()
        else:
            omic_losses[omic] = per_sample.mean()

    if not omic_losses:
        loss_total = torch.tensor(0.0, device=device)
    else:
        loss_total = torch.stack(list(omic_losses.values())).mean()

    metrics = {"omic_losses": omic_losses, "group_omic_losses": dict(group_omic_losses)}
    return loss_total, metrics


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence to N(0, I) using torch.distributions."""
    std = torch.exp(0.5 * logvar) + 1e-4
    posterior = Normal(mu, std)
    prior = Normal(torch.zeros_like(mu), torch.ones_like(std))
    return kl_div(posterior, prior).mean()


def contrastive_loss(mu: Tensor, labels: Tensor) -> Tensor:
    """Contrastive loss on joint embeddings using pytorch_metric_learning."""
    from pytorch_metric_learning import losses as pml_losses
    from pytorch_metric_learning.distances import CosineSimilarity

    loss_func = pml_losses.ContrastiveLoss(distance=CosineSimilarity())

    # If labels are one-hot, convert to integer indices
    if labels.dim() == 2:
        label_indices = labels.argmax(dim=1)
    else:
        label_indices = labels

    return loss_func(mu, label_indices)


def adversarial_loss(
    pred: Tensor,
    target: Tensor,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Weighted cross-entropy for adversarial training.

    Parameters
    ----------
    pred : Tensor [B, n_classes]
        Logits from discriminator.
    target : Tensor [B, n_classes] or [B]
        One-hot or class indices.
    class_weights : Tensor [n_classes] or None
    """
    if target.dim() == 2:
        target = torch.argmax(target, dim=1)
    loss_fn = CrossEntropyLoss(weight=class_weights, reduction="mean")
    return loss_fn(pred, target)
