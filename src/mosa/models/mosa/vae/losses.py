from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import Tensor


def reconstruction_loss(
    x_hat: dict[str, Tensor],
    x: dict[str, Tensor],
    mask: dict[str, Tensor],
    group: Tensor | None = None,
    loss_types: dict[str, str] | None = None,
    recon_weights: dict[str, float] | None = None,
) -> tuple[Tensor, dict]:
    """Masked reconstruction loss across omics.

    Parameters
    ----------
    x_hat : dict of Tensor [B, D]
        Reconstructed values.
    x : dict of Tensor [B, D]
        Target values.
    mask : dict of bool Tensor [B, D]
        Feature presence masks.
    group : Tensor [B] or None
        Group labels for macro loss averaging.
    loss_types : dict of str or None
        Per-omic "mean" (sample-weighted MSE) or "macro" (group-balanced MSE).
        Omics not present default to "mean".
    recon_weights : dict of float or None
        Per-omic weight applied before summing into the total loss.
        Omics not present default to weight 1.0.

    Returns
    -------
    loss : Tensor
        Scalar loss.
    metrics : dict
        Per-omic and per-group losses.
    """
    loss_types = loss_types or {}
    device = next(iter(x.values())).device
    omic_losses = {}
    group_omic_losses: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

    for omic in x:
        feature_mask = mask[omic]  # [B, D]
        recon = x_hat[omic]  # [B, D]
        target = x[omic]  # [B, D]

        mse_per_feature = F.mse_loss(recon, target, reduction="none")  # [B, D]
        mse_masked = mse_per_feature * feature_mask.float()

        sample_mask = feature_mask.any(dim=1)
        if not sample_mask.any():
            continue

        n_present = feature_mask.sum(dim=1).clamp(min=1)  # [B]
        per_sample = mse_masked.sum(dim=1) / n_present  # [B]
        per_sample_valid = per_sample[sample_mask]

        if loss_types.get(omic, "mean") == "macro" and group is not None:
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
            omic_losses[omic] = per_sample_valid.mean()

    if not omic_losses:
        loss_total = torch.tensor(0.0, device=device)
    else:
        loss_total = sum(
            (recon_weights.get(omic, 1.0) if recon_weights else 1.0) * loss
            for omic, loss in omic_losses.items()
        )

    metrics = {"omic_losses": omic_losses, "group_omic_losses": dict(group_omic_losses)}
    return loss_total, metrics


def kl_divergence(mu: Tensor, logvar: Tensor) -> Tensor:
    """KL divergence from posterior to standard normal, summed over latent dim.

    logvar is clamped before exponentiating; without it, an unstable posterior
    drives logvar to large values, exp(logvar) overflows to inf, and the
    resulting NaN propagates into a segfault.
    """
    logvar = logvar.clamp(min=-10.0, max=10.0)
    kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()


def contrastive_loss(mu: Tensor, labels: Tensor) -> Tensor:
    """Contrastive loss on latent embeddings.

    Converts one-hot labels to indices if needed.
    """
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
    focal_gamma: float = 0.0,
) -> Tensor:
    """Weighted cross-entropy (optionally focal) for adversarial training.

    Parameters
    ----------
    pred : Tensor [B, n_classes]
        Logits from discriminator.
    target : Tensor [B, n_classes] or [B]
        One-hot or class indices.
    class_weights : Tensor [n_classes] or None
    focal_gamma : float
        Focal loss exponent; 0 gives plain (weighted) cross-entropy.
        Larger values down-weight already-well-classified examples, focusing
        the loss on hard/minority cases (useful under heavy class imbalance).
    """
    if target.dim() == 2:
        target = torch.argmax(target, dim=1)
    ce = F.cross_entropy(pred, target, weight=class_weights, reduction="none")
    if focal_gamma > 0:
        pt = torch.exp(-ce)
        ce = ((1 - pt) ** focal_gamma) * ce
    return ce.mean()
