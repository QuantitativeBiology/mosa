from __future__ import annotations

from dataclasses import dataclass

from mosa.config import ModelConfig

_VALID_CONVERGENCE = ("fast", "medium", "slow")


@dataclass
class MOFAConfig(ModelConfig):
    """All MOFA hyperparameters."""

    n_factors: int = 50
    ard_factors: bool = True
    drop_r2: float | None = 0.001
    scale_views: bool = False
    scale_groups: bool = False
    convergence_mode: str = "fast"
    iterations: int = 1000
    gpu_mode: bool = False
    gpu_device: int | None = None

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.gpu_mode, bool):
            raise ValueError("gpu_mode must be a boolean")
        if self.gpu_device is not None and (
            isinstance(self.gpu_device, bool)
            or not isinstance(self.gpu_device, int)
            or self.gpu_device < 0
        ):
            raise ValueError("gpu_device must be a non-negative integer or null")
        if self.n_factors <= 0:
            raise ValueError(f"n_factors must be positive, got {self.n_factors}")
        if self.iterations <= 0:
            raise ValueError(f"iterations must be positive, got {self.iterations}")
        if self.convergence_mode not in _VALID_CONVERGENCE:
            raise ValueError(
                f"convergence_mode must be one of {_VALID_CONVERGENCE}, got '{self.convergence_mode}'"
            )

    def validate_against_data(self, data_cfg, eval_cfg, summary: dict) -> list[str]:
        """Held-out Gaussian samples can be projected after fitting."""
        return []
