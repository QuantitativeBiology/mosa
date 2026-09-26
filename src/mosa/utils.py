from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import yaml
from torch import Tensor

from mosa.config import Config, DataConfig, EvaluationConfig, check_unknown_keys
from mosa.errors import ConfigError, MissingFileError

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    pl.seed_everything(seed, workers=True)
    logger.debug("Seeded everything with %d", seed)


def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if missing; return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yaml(path: str | Path) -> dict:
    """Read a YAML file into a dict; empty files return {}."""
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError as e:
        raise MissingFileError(f"File not found: {path}") from e
    except yaml.YAMLError as e:
        # PyYAML's message already carries the line and column.
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e


def load_config(yaml_path: str | Path) -> Config:
    """Load a YAML config and return a Config bundling DataConfig + ModelConfig + EvaluationConfig.

    The 'evaluation:' block is optional; omitting it takes EvaluationConfig's
    defaults.
    """
    from mosa.models.registry import model_config_classes

    raw = read_yaml(yaml_path)

    if "data" not in raw or "model" not in raw:
        raise ConfigError(
            f"Config {yaml_path} must contain top-level 'data:' and 'model:' blocks"
        )

    check_unknown_keys(DataConfig, raw["data"], "data:")
    data = DataConfig(**raw["data"])

    model_raw = dict(raw["model"])
    if "type" not in model_raw:
        raise ConfigError("model.type is required (e.g. 'mosa_vae', 'mofa')")
    mtype = model_raw.pop("type")
    model_configs = model_config_classes()
    if mtype not in model_configs:
        raise ConfigError(f"unknown model.type '{mtype}'; valid: {list(model_configs)}")

    model_cfg = model_configs[mtype].from_yaml_dict(model_raw)

    evaluation_raw = raw.get("evaluation", {})
    check_unknown_keys(EvaluationConfig, evaluation_raw, "evaluation:")
    evaluation = EvaluationConfig(**evaluation_raw)

    return Config(data=data, model=model_cfg, evaluation=evaluation)


def validate_config_against_data(cfg: Config) -> list[str]:
    """Check that the data at cfg.data.path satisfies the config's requirements.

    Cost-ordered: path existence, then a lazy structure summary (no matrices
    loaded), then structural and model-specific value checks. Raises on hard
    failures (missing path/view/mask/model_type/target_batch); returns
    collected warning strings for soft ones (tissue, mutations, adversarial
    batch count).
    """
    from mosa.data.io import summarize_structure

    cfg.data.validate_paths()
    summary = summarize_structure(cfg.data.path)
    warnings = cfg.data.validate_against_data(summary)
    warnings += cfg.model.validate_against_data(cfg.data, cfg.evaluation, summary)
    return warnings


def tensors_to_numpy(t: Tensor) -> np.ndarray:
    """Convert tensor to numpy array on CPU."""
    return t.detach().cpu().numpy()


@contextlib.contextmanager
def mudata_set_options(**kwargs):
    import mudata

    if hasattr(mudata, "set_options"):
        with mudata.set_options(**kwargs):
            yield
    else:
        yield
