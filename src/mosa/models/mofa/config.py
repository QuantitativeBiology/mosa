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

    def validate_against_data(self, data_cfg, eval_cfg, summary: dict) -> list[str]:
        """MOFA needs only the structural requirements, already checked by DataConfig.

        The one soft check is the holdout: MOFA's fit() ignores validation
        data, so a nonzero test_size removes samples from training and buys
        nothing.
        """
        if eval_cfg.test_size > 0:
            return [
                f"evaluation.test_size is {eval_cfg.test_size} but MOFA ignores "
                f"validation data; those samples would be held out of training "
                f"for no benefit. Set evaluation.test_size to 0."
            ]
        return []
