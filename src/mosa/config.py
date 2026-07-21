from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Shared data definition. Consumed by every model."""

    path: str = ""
    views: list[str] = field(default_factory=list)
    mask_layer_name: str = "mask"
    discrete_views: set[str] = field(default_factory=set)
    use_tissue: bool = True
    use_mutations: bool = True

    def __post_init__(self):
        if isinstance(self.discrete_views, list):
            self.discrete_views = set(self.discrete_views)
        if not self.views:
            raise ValueError("data.views must not be empty")
        bad = self.discrete_views - set(self.views)
        if bad:
            raise ValueError(f"data.discrete_views not in data.views: {sorted(bad)}")

    def validate_paths(self) -> None:
        """Check that the MuData file/dir exists. Call before training."""
        if not self.path:
            raise FileNotFoundError("data.path is required")
        p = Path(self.path)
        if not (p.is_file() or p.is_dir()):
            raise FileNotFoundError(f"data.path not found: {self.path}")

    def validate_against_data(self, summary: dict) -> list[str]:
        """Check structural requirements against a data summary (see data/io.py:summarize_structure).

        Shared by the `validate` CLI path and load-time structure checks, so both
        raise the same errors. Raises ValueError on the first hard failure. No
        soft/warning-level structural checks exist today; returns [] for a
        uniform interface with model-specific validate_against_data().
        """
        obs_columns = summary.get("obs_columns", [])
        if "model_type" not in obs_columns:
            raise ValueError(
                f"MuData .obs missing 'model_type' column. Available: {obs_columns}"
            )

        modalities = summary.get("modalities", {})
        for view in self.views:
            if view not in modalities:
                raise ValueError(
                    f"View '{view}' not in MuData. Available: {list(modalities.keys())}"
                )
            layers = modalities[view].get("layers", [])
            if self.mask_layer_name not in layers:
                raise ValueError(
                    f"Mask layer '{self.mask_layer_name}' not in '{view}'. Available: {layers}"
                )

        return []


@dataclass
class ModelConfig:
    """Base contract for every model config: orchestration fields shared across all models."""

    output_dir: str = "outputs"
    random_seed: int = 42
    test_size: float = 0.1

    def __post_init__(self):
        if not 0.0 <= self.test_size < 1.0:
            raise ValueError(f"test_size must be in [0, 1), got {self.test_size}")

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "ModelConfig":
        """Build from a raw YAML mapping (model.type already stripped).

        Base implementation passes the mapping straight through. Subclasses
        with nested config objects (e.g. per-view configs) override this to
        parse those before construction, so load_config stays model-agnostic.
        """
        return cls(**raw)


@dataclass
class Config:
    """A complete experiment configuration: shared data + model-specific knobs."""

    data: DataConfig
    model: ModelConfig

    def __post_init__(self):
        # Cross-check: if the model declares per-view architecture, view sets must match data.
        model_views = getattr(self.model, "views", None)
        if isinstance(model_views, dict) and model_views:
            missing = set(model_views) - set(self.data.views)
            extra = set(self.data.views) - set(model_views)
            if missing or extra:
                raise ValueError(
                    f"model.views must match data.views — "
                    f"missing in model: {sorted(missing)}, extra in model: {sorted(extra)}"
                )
