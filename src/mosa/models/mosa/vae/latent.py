from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from mosa.models.mosa.vae.layers import MLP

_REGISTRY: dict[str, type] = {}


def register_latent(name: str):
    """Decorator to register a latent space subclass."""

    def decorator(cls: type) -> type:
        _REGISTRY[name] = cls
        return cls

    return decorator


class BaseLatentSpace(ABC, nn.Module):
    """Abstract base for latent space fusion methods.

    Subclasses implement _build and forward.
    Instantiate via BaseLatentSpace.create(method, ...).
    """

    def __init__(
        self,
        view_dims: dict[str, int],
        latent_dim: int,
        shared_hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        self.view_dims = view_dims
        self.latent_dim = latent_dim
        self.shared_hidden_dims = shared_hidden_dims or []
        self._build()

    @abstractmethod
    def _build(self) -> None:
        """Construct submodules. Called at the end of ``__init__``."""

    @abstractmethod
    def forward(
        self,
        view_embeddings: dict[str, torch.Tensor],
        view_order: list[str],
        sample_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mu, logvar, z) given per-view embeddings."""

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar) + 1e-8
            return mu + torch.randn_like(std) * std
        return mu

    @classmethod
    def create(
        cls,
        method: str,
        view_dims: dict[str, int],
        latent_dim: int,
        shared_hidden_dims: list[int] | None = None,
        use_shared_head: bool = True,
    ) -> BaseLatentSpace:
        if method not in _REGISTRY:
            raise ValueError(
                f"Unknown fusion method '{method}'. Available: {list(_REGISTRY.keys())}"
            )
        if method == "poe":
            return _REGISTRY[method](
                view_dims, latent_dim, shared_hidden_dims, use_shared_head
            )
        return _REGISTRY[method](view_dims, latent_dim, shared_hidden_dims)


@register_latent("concat")
class ConcatLatentSpace(BaseLatentSpace):
    """Concatenates view embeddings and projects to Gaussian posterior."""

    def _build(self) -> None:
        concat_dim = sum(self.view_dims.values())
        self.fc_mu = nn.Linear(concat_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(concat_dim, self.latent_dim)

    def forward(
        self,
        view_embeddings: dict[str, torch.Tensor],
        view_order: list[str],
        sample_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        concat = torch.cat([view_embeddings[name] for name in view_order], dim=1)
        mu = self.fc_mu(concat)
        logvar = self.fc_logvar(concat)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


@register_latent("poe")
class PoELatentSpace(BaseLatentSpace):
    """Product of Experts: fuses per-view Gaussians via precision-weighted averaging.

    Each view can either project to mu/logvar through a shared head or emit
    mu/logvar directly. The per-view posteriors are then combined with an
    isotropic N(0, I) prior, except when only one view is present, in which
    case that view's posterior is used directly.
    """

    def __init__(
        self,
        view_dims: dict[str, int],
        latent_dim: int,
        shared_hidden_dims: list[int] | None = None,
        use_shared_head: bool = True,
    ):
        self.use_shared_head = use_shared_head
        super().__init__(view_dims, latent_dim, shared_hidden_dims)

    EPS = 1e-8

    def _build(self) -> None:
        dims = set(self.view_dims.values())
        if len(dims) > 1:
            raise ValueError(
                f"PoE fusion requires all views to have the same encoder output dim, "
                f"got {self.view_dims}"
            )
        shared_dim = next(iter(dims))

        if self.use_shared_head:
            # Layer sizes: input, intermediate dims, 2*latent_dim (mu, logvar)
            layer_sizes = [shared_dim] + self.shared_hidden_dims + [self.latent_dim * 2]

            self.shared_head = MLP(
                layer_sizes=layer_sizes,
                dropout_p=0.0,
                use_batch_norm=True,
                activation=nn.PReLU,
                output_activation=None,
                bn_momentum=0.01,
                bn_eps=0.001,
            )
        elif shared_dim != self.latent_dim * 2:
            raise ValueError(
                "PoE with direct per-view mu/logvar requires encoder outputs to have "
                f"dimension 2 * latent_dim ({self.latent_dim * 2}), got {shared_dim}"
            )

    def _project_view(
        self,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.shared_head(embedding) if self.use_shared_head else embedding
        return stats.split(self.latent_dim, dim=1)

    def forward(
        self,
        view_embeddings: dict[str, torch.Tensor],
        view_order: list[str],
        sample_masks: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(view_order) == 1:
            mu, logvar = self._project_view(view_embeddings[view_order[0]])
            z = self.reparameterize(mu, logvar)
            return mu, logvar, z

        B = next(iter(view_embeddings.values())).shape[0]
        device = next(iter(view_embeddings.values())).device

        # Start with isotropic prior: mu=0, precision=1
        precision_sum = torch.ones(B, self.latent_dim, device=device)
        mu_precision_sum = torch.zeros(B, self.latent_dim, device=device)

        for name in view_order:
            view_mask = sample_masks.get(name) if sample_masks is not None else None
            mu_v, logvar_v = self._project_view(view_embeddings[name])
            precision_v = torch.exp(-logvar_v)

            if view_mask is not None:
                mask_f = view_mask.unsqueeze(1)  # [B, 1]
                precision_v = precision_v * mask_f
                mu_v = mu_v * mask_f

            precision_sum = precision_sum + precision_v
            mu_precision_sum = mu_precision_sum + mu_v * precision_v

        mu = mu_precision_sum / (precision_sum + self.EPS)
        logvar = -torch.log(precision_sum + self.EPS)

        z = self.reparameterize(mu, logvar)
        return mu, logvar, z
