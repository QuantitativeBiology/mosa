from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import yaml
from torch import Tensor

from mosa.config import Config, DataConfig, ModelConfig
from mosa.models.mofa.config import MOFAConfig
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig

logger = logging.getLogger(__name__)


_MODEL_CONFIGS: dict[str, type[ModelConfig]] = {
    "mosa_vae": MOSAConfig,
    "mofa": MOFAConfig,
}


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    pl.seed_everything(seed, workers=True)
    logger.debug("Seeded everything with %d", seed)


def load_config(yaml_path: str | Path) -> Config:
    """Load a YAML config and return a Config bundling DataConfig + ModelConfig."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if "data" not in raw or "model" not in raw:
        raise ValueError(f"Config {yaml_path} must contain top-level 'data:' and 'model:' blocks")

    data = DataConfig(**raw["data"])

    model_raw = dict(raw["model"])
    if "type" not in model_raw:
        raise ValueError("model.type is required (e.g. 'mosa_vae', 'mofa')")
    mtype = model_raw.pop("type")
    if mtype not in _MODEL_CONFIGS:
        raise ValueError(f"unknown model.type '{mtype}'; valid: {list(_MODEL_CONFIGS)}")

    if mtype == "mosa_vae" and "views" in model_raw:
        model_raw["views"] = {
            name: OmicViewConfig(name=name, **vcfg)
            for name, vcfg in model_raw["views"].items()
        }

    model_cfg = _MODEL_CONFIGS[mtype](**model_raw)

    return Config(data=data, model=model_cfg)


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
    warnings += cfg.model.validate_against_data(cfg.data, summary)
    return warnings


def tensors_to_numpy(t: Tensor) -> np.ndarray:
    """Convert tensor to numpy array on CPU."""
    return t.detach().cpu().numpy()
