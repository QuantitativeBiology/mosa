from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from mosa.config import DataConfig, ModelConfig
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


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig) -> MultiOmicModel:
    """Instantiate the registered model class matching model_cfg's exact type."""
    for reg in _REGISTRY.values():
        if type(model_cfg) is reg.config_cls:
            return reg.model_cls(data_cfg, model_cfg)
    raise TypeError(f"Unsupported model_cfg type: {type(model_cfg).__name__}")


def load_model(path: str | Path) -> MultiOmicModel:
    """Load a saved model, dispatching to the registered class that wrote it.

    Artifact-based models are dispatched by file extension. Checkpoint-based
    models embed their registered name under
    hyper_parameters["model_type_name"] on save, but auto-checkpoints written
    by Lightning's ModelCheckpoint callback during training (last.ckpt,
    epoch-NNN.ckpt) go through a different path and carry no such key;
    absence of the key falls back to "mosa_vae".
    """
    path = Path(path)
    if path.suffix == ".hdf5":
        if "mofa" not in _REGISTRY:
            raise ValueError("mofa model type is not registered")
        return _REGISTRY["mofa"].model_cls.load(path)

    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    name = checkpoint.get("hyper_parameters", {}).get("model_type_name", "mosa_vae")
    if name not in _REGISTRY:
        raise ValueError(
            f"Cannot determine registered model type for checkpoint '{path}' "
            f"(found model_type_name={name!r}); known types: {list(_REGISTRY)}"
        )
    return _REGISTRY[name].model_cls.load(path)
