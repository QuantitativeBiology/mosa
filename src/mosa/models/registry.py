from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from mosa.config import DataConfig, ModelConfig
from mosa.errors import DataError
from mosa.models.api import MultiOmicModel

# Self-contained: no imports of concrete model modules here. Model modules
# import register_model from this file to register themselves, so importing
# them from here would be circular. Registration happens when
# mosa.models.__init__ imports the model modules (see that file).


class _Registration(NamedTuple):
    config_cls: type[ModelConfig]
    model_cls: type[MultiOmicModel]


_REGISTRY: dict[str, _Registration] = {}


def register_model(name: str, config_cls: type[ModelConfig]):
    """Class decorator registering a MultiOmicModel subclass under `name`.

    Pairs the model class with its ModelConfig subclass so both build_model
    (dispatch by config type) and load_config (dispatch by YAML model.type
    string) can resolve the same registration. Also stamps the model class
    with `registered_name` so instances can embed it in saved checkpoints.
    """

    def decorator(model_cls: type[MultiOmicModel]) -> type[MultiOmicModel]:
        _REGISTRY[name] = _Registration(config_cls, model_cls)
        model_cls.registered_name = name
        return model_cls

    return decorator


def model_config_classes() -> dict[str, type[ModelConfig]]:
    """Map of registered model-type name to ModelConfig subclass, for load_config."""
    return {name: reg.config_cls for name, reg in _REGISTRY.items()}


def model_class_for(model_cfg: ModelConfig) -> type[MultiOmicModel]:
    """Registered model class matching model_cfg's exact type, without instantiating it."""
    for reg in _REGISTRY.values():
        if type(model_cfg) is reg.config_cls:
            return reg.model_cls
    raise TypeError(f"Unsupported model_cfg type: {type(model_cfg).__name__}")


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig) -> MultiOmicModel:
    """Instantiate the registered model class matching model_cfg's exact type."""
    return model_class_for(model_cfg)(data_cfg, model_cfg)


def load_model(path: str | Path) -> MultiOmicModel:
    """Load a saved model with the one registered class that owns the file.

    Each model class decides ownership through owns_checkpoint(), so this
    function holds no knowledge of any model's file format.
    """
    path = Path(path)
    owners = [
        name for name, reg in _REGISTRY.items() if reg.model_cls.owns_checkpoint(path)
    ]
    if len(owners) != 1:
        suffixes = {
            name: list(reg.model_cls.checkpoint_suffixes)
            for name, reg in _REGISTRY.items()
        }
        found = f"claimed by {owners}" if owners else "claimed by no model"
        raise DataError(
            f"Cannot determine which model wrote '{path}': {found}. "
            f"Registered file types: {suffixes}"
        )
    return _REGISTRY[owners[0]].model_cls.load(path)
