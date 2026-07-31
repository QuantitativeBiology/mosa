from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mosa.errors import DataError


@dataclass
class MultiOmicDataset:
    """Format-agnostic container for multi-omic data.

    This is the API contract between data loading and models.
    Data loaders (h5mu, zarr, CSV) produce this; models consume it.
    """

    views: dict[str, np.ndarray]
    """Per-view feature matrices. Keys are view names (e.g. 'gexp_voom'),
    values are float32 arrays of shape [N, D_view]. Missing values are NaN
    (imputed to 0.0 downstream by the datamodule, after the scaler is fit on
    observed values only)."""

    masks: dict[str, np.ndarray]
    """Per-view presence masks. Same keys as views.
    Boolean arrays of shape [N, D_view]. True where data is present."""

    metadata: pd.DataFrame
    """Sample metadata. Index is sample IDs.
    Required column: 'model_type'.
    Optional columns: 'tissue', 'mutation_*'."""

    feature_names: dict[str, list[str]]
    """Per-view feature names. Keys match views. Each list has D_view entries."""

    @property
    def n_samples(self) -> int:
        return len(self.metadata)

    @property
    def view_names(self) -> list[str]:
        return list(self.views.keys())

    @property
    def sample_names(self) -> list[str]:
        return list(self.metadata.index)

    def validate(self) -> None:
        """Check structural consistency of all fields. Raises DataError on failure."""
        view_keys = set(self.views.keys())
        mask_keys = set(self.masks.keys())
        feat_keys = set(self.feature_names.keys())

        if view_keys != mask_keys:
            raise DataError(
                f"views and masks have different keys: {view_keys} vs {mask_keys}"
            )
        if view_keys != feat_keys:
            raise DataError(
                f"views and feature_names have different keys: {view_keys} vs {feat_keys}"
            )

        n = len(self.metadata)
        for k in self.views:
            v_shape = self.views[k].shape
            m_shape = self.masks[k].shape
            if v_shape != m_shape:
                raise DataError(
                    f"views['{k}'] shape {v_shape} != masks['{k}'] shape {m_shape}"
                )
            if v_shape[0] != n:
                raise DataError(
                    f"views['{k}'] has {v_shape[0]} samples but metadata has {n}"
                )
            n_features = len(self.feature_names[k])
            if v_shape[1] != n_features:
                raise DataError(
                    f"views['{k}'] has {v_shape[1]} features but feature_names['{k}'] has {n_features}"
                )

        if "model_type" not in self.metadata.columns:
            raise DataError("metadata is missing required column 'model_type'")

    def subset(self, indices: np.ndarray) -> MultiOmicDataset:
        """Subset by sample indices."""
        return MultiOmicDataset(
            views={k: v[indices] for k, v in self.views.items()},
            masks={k: v[indices] for k, v in self.masks.items()},
            metadata=self.metadata.iloc[indices].copy(),
            feature_names=self.feature_names,
        )
