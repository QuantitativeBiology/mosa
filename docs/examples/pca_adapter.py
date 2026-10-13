"""Small executable adapter example; mean-imputed PCA on concatenated views."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mosa.config import DataConfig, ModelConfig
from mosa.models.api import MultiOmicModel
from mosa.models.registry import register_model


@dataclass
class ExamplePCAConfig(ModelConfig):
    n_components: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.n_components < 1:
            raise ValueError("n_components must be positive")


@register_model("example_pca", ExamplePCAConfig)
class ExamplePCA(MultiOmicModel):
    checkpoint_suffixes = (".tutorial-pca",)

    def __init__(self, data_cfg, model_cfg):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.epoch_history = []
        self.training_data = None

    def _matrix(self, data):
        data.validate()
        arrays = []
        for name in self.data_cfg.views:
            if data.feature_names.get(name) != self.features[name]:
                raise ValueError(f"Feature names or order differ for {name}")
            values = np.asarray(data.views[name], dtype=float)
            observed = data.masks[name]
            if not np.isfinite(values[observed]).all():
                raise ValueError("Observed entries must be finite")
            arrays.append(np.where(observed, values, np.nan))
        return np.concatenate(arrays, axis=1)

    def fit(self, train, val=None, resume_from=None):
        if resume_from is not None:
            raise ValueError("This example does not resume training")
        self.features = {
            name: list(train.feature_names[name]) for name in self.data_cfg.views
        }
        x = self._matrix(train)
        if np.isnan(x).all(axis=0).any():
            raise ValueError("Every feature needs an observed training value")
        if self.model_cfg.n_components > min(x.shape):
            raise ValueError("n_components exceeds the training matrix rank bound")
        self.mean = np.nanmean(x, axis=0)
        centered = np.where(np.isnan(x), self.mean, x) - self.mean
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        self.components = vt[: self.model_cfg.n_components]
        self.training_data = train

    def transform(self, data):
        x = self._matrix(data)
        return (np.where(np.isnan(x), self.mean, x) - self.mean) @ self.components.T

    def reconstruct(self, data):
        x = self.transform(data) @ self.components + self.mean
        result, offset = {}, 0
        for name in self.data_cfg.views:
            width = len(self.features[name])
            result[name] = x[:, offset : offset + width]
            offset += width
        return result

    def save(self, path):
        self.require_checkpoint_suffix(path)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data_cfg = asdict(self.data_cfg)
        data_cfg["discrete_views"] = sorted(data_cfg["discrete_views"])
        metadata = {
            "data": data_cfg,
            "model": asdict(self.model_cfg),
            "features": self.features,
        }
        with path.open("wb") as handle:
            np.savez(
                handle,
                mean=self.mean,
                components=self.components,
                metadata=json.dumps(metadata),
            )

    @classmethod
    def load(cls, path, **kwargs):
        with np.load(path, allow_pickle=False) as saved:
            metadata = json.loads(str(saved["metadata"]))
            model = cls(
                DataConfig(**metadata["data"]), ExamplePCAConfig(**metadata["model"])
            )
            model.features = metadata["features"]
            model.mean = saved["mean"].copy()
            model.components = saved["components"].copy()
        return model

    def save_outputs(self, output_dir=None):
        if self.training_data is None:
            raise ValueError("Run exports require the training data retained by fit()")
        target = Path(output_dir or self.model_cfg.output_dir)
        full = target / "full"
        full.mkdir(parents=True, exist_ok=True)
        data = self.training_data
        pd.DataFrame(
            self.transform(data),
            index=data.sample_names,
            columns=[f"z_{i}" for i in range(self.model_cfg.n_components)],
        ).to_parquet(full / "latent.parquet")
        for name, values in self.reconstruct(data).items():
            pd.DataFrame(
                values, index=data.sample_names, columns=self.features[name]
            ).to_parquet(full / f"recon_{name}.parquet")
        self.save(target / "model.tutorial-pca")


if __name__ == "__main__":
    from mosa.cli import main

    main()
