from __future__ import annotations

from dataclasses import dataclass

from mosa.config import ModelConfig

_VALID_CONVERGENCE = ("fast", "medium", "slow")


@dataclass
class MOFAConfig(ModelConfig):
    """All MOFA hyperparameters."""

    n_factors: int = 50
    ard_factors: bool = True
    drop_r2: float = 0.001
    scale_views: bool = False
    scale_groups: bool = False
    convergence_mode: str = "fast"

    def __post_init__(self):
        super().__post_init__()
        if self.n_factors <= 0:
            raise ValueError(f"n_factors must be positive, got {self.n_factors}")
        if self.convergence_mode not in _VALID_CONVERGENCE:
            raise ValueError(
                f"convergence_mode must be one of {_VALID_CONVERGENCE}, got '{self.convergence_mode}'"
            )
