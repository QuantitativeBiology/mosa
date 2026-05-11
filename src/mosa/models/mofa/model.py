from __future__ import annotations

# Optional dependencies: mofapy2 and mofax are not listed in pyproject.toml
# required deps. Install via: pip install mofapy2 mofax

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from mosa.api import MultiOmicModel
from mosa.data.dataset import MultiOmicDataset


class MOFAModel(MultiOmicModel):
    """MultiOmicModel implementation using MOFA+ (mofapy2 / mofax).

    Training uses Automatic Relevance Determination (ARD) for regularization;
    validation data is therefore not used.
    """

    def __init__(
        self,
        n_factors: int = 50,
        ard_factors: bool = True,
        drop_r2: float = 0.001,
        scale_views: bool = False,
        scale_groups: bool = False,
        convergence_mode: str = "fast",
        seed: int = 42,
        save_path: str | None = None,
    ):
        self.n_factors = n_factors
        self.ard_factors = ard_factors
        self.drop_r2 = drop_r2
        self.scale_views = scale_views
        self.scale_groups = scale_groups
        self.convergence_mode = convergence_mode
        self.seed = seed
        self.save_path = save_path
        self._model = None

    def _to_long_df(self, data: MultiOmicDataset) -> pd.DataFrame:
        """Convert MultiOmicDataset to MOFA long-format DataFrame."""
        sample_names = data.sample_names
        group_map = data.metadata["model_type"].to_dict()

        frames: list[pd.DataFrame] = []
        for view_name in data.view_names:
            matrix = data.views[view_name].copy().astype(float)
            mask = data.masks[view_name]
            matrix[~mask] = np.nan

            df = pd.DataFrame(
                matrix,
                index=sample_names,
                columns=data.feature_names[view_name],
            )
            df.index.name = "sample"
            long = df.reset_index().melt(
                id_vars="sample",
                var_name="feature",
                value_name="value",
            )
            long["view"] = view_name
            long["group"] = long["sample"].map(group_map)
            frames.append(long)

        return pd.concat(frames, ignore_index=True)[
            ["sample", "feature", "value", "view", "group"]
        ]

    def fit(self, train: MultiOmicDataset, val: MultiOmicDataset | None = None) -> None:
        """Train the MOFA model. val is ignored."""
        from mofapy2.run.entry_point import entry_point

        ent = entry_point()
        ent.set_data_options(
            scale_views=self.scale_views,
            scale_groups=self.scale_groups,
        )
        ent.set_data_df(self._to_long_df(train))
        ent.set_model_options(factors=self.n_factors, ard_factors=self.ard_factors)
        ent.set_train_options(
            dropR2=self.drop_r2,
            seed=self.seed,
            convergence_mode=self.convergence_mode,
        )
        ent.build()
        ent.run()

        save_path = self.save_path or "mofa_model.hdf5"
        ent.save(save_path, save_data=True)

        import mofax as mfx
        self._model = mfx.mofa_model(save_path)

    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        """Latent factors for training samples. Raises NotImplementedError for unseen data."""
        if self._model is None:
            raise RuntimeError("Model must be fit before calling transform()")

        factors_df = self._model.get_factors(df=True)
        missing = [s for s in data.sample_names if s not in factors_df.index]
        if missing:
            raise NotImplementedError(
                f"MOFA does not support out-of-sample projection. "
                f"Unseen samples: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        return factors_df.loc[data.sample_names].values

    def reconstruct(self, data: MultiOmicDataset) -> dict[str, np.ndarray]:
        """Reconstruct omic views as Z @ W.T per view."""
        if self._model is None:
            raise RuntimeError("Model must be fit before calling reconstruct()")

        Z = self.transform(data)
        result: dict[str, np.ndarray] = {}
        for view_name in data.view_names:
            W_df = self._model.get_weights(views=view_name, df=True)
            result[view_name] = Z @ W_df.values.T
        return result

    def save(self, path: str | Path) -> None:
        """Copy the HDF5 model file to path."""
        if self._model is None:
            raise RuntimeError("Model must be fit before saving")

        src = Path(self.save_path or "mofa_model.hdf5")
        dst = Path(path)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    @classmethod
    def load(cls, path: str | Path) -> MOFAModel:
        """Load a trained MOFA model from an HDF5 file."""
        import mofax as mfx

        instance = cls(save_path=str(path))
        instance._model = mfx.mofa_model(str(path))
        return instance
