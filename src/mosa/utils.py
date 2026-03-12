from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import yaml
from torch import Tensor

from mosa.config import MOSAConfig, TrainerConfig, ViewConfig

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    pl.seed_everything(seed, workers=True)
    logger.debug("Seeded everything with %d", seed)


def load_config(yaml_path: str | Path) -> MOSAConfig:
    """Parse a YAML config file and return a MOSAConfig."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    views_raw = raw.pop("views", {})
    views = {}
    for name, vcfg in views_raw.items():
        vcfg.setdefault("name", name)
        views[name] = ViewConfig(**vcfg)

    trainer_raw = raw.pop("trainer", {})
    trainer = TrainerConfig(**trainer_raw)

    return MOSAConfig(views=views, trainer=trainer, **raw)


def tensors_to_numpy(t: Tensor) -> np.ndarray:
    """Detach a tensor, move to CPU, and convert to numpy."""
    return t.detach().cpu().numpy()
