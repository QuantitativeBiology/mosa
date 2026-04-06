from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

from mosa.config import MOSAConfig
from mosa.data.batch import MOSABatch, collate_fn

logger = logging.getLogger(__name__)


class MOSADataset(Dataset):
    """In-memory dataset with per-sample tensors for a single split."""

    def __init__(
        self,
        omics_data: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        conditionals: np.ndarray,
        tissue_labels: np.ndarray,
        source_ids: np.ndarray,
        sample_weights: np.ndarray,
        sample_names: list[str],
        omic_names: list[str],
    ):
        self.omics = {k: torch.tensor(v, dtype=torch.float32) for k, v in omics_data.items()}
        self.masks = {k: torch.tensor(v, dtype=torch.bool) for k, v in masks.items()}
        self.conditionals = torch.tensor(conditionals, dtype=torch.float32)
        self.tissue_labels = torch.tensor(tissue_labels, dtype=torch.float32)
        self.source_ids = torch.tensor(source_ids, dtype=torch.long)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.sample_names = list(sample_names)
        self.omic_names = omic_names

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> dict:
        return {
            "encoder_inputs": {k: self.omics[k][idx] for k in self.omic_names},
            "decoder_targets": {k: self.omics[k][idx] for k in self.omic_names},
            "missing_masks": {k: self.masks[k][idx] for k in self.omic_names},
            "conditionals": self.conditionals[idx],
            "tissue_labels": self.tissue_labels[idx],
            "source_ids": self.source_ids[idx],
            "sample_weights": self.sample_weights[idx],
            "sample_name": self.sample_names[idx],
        }


class LazyZarrDataset(Dataset):
    """Lazy-loading dataset from MuData zarr store; each worker opens its own handle."""

    def __init__(
        self,
        zarr_path: str,
        view_names: list[str],
        indices: np.ndarray,
        conditionals: np.ndarray,
        tissue_labels: np.ndarray,
        source_ids: np.ndarray,
        sample_weights: np.ndarray,
        sample_names: list[str],
        scalers: dict[str, dict[str, np.ndarray] | None],
        mask_layer_name: str = "mask",
    ):
        self.zarr_path = zarr_path
        self.view_names = view_names
        self.indices = indices
        self.conditionals = conditionals
        self.tissue_labels = tissue_labels
        self.source_ids = source_ids
        self.sample_weights = sample_weights
        self.sample_names = list(sample_names)
        self.scalers = scalers
        self.mask_layer_name = mask_layer_name
        self._store = None

    def _get_store(self):
        if self._store is None:
            self._store = zarr.open_group(self.zarr_path, mode="r")
        return self._store

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        """Batched read: one zarr slice per view for the whole batch."""
        store = self._get_store()
        real_indices = self.indices[indices]

        # Single vectorised zarr read per view
        all_X: dict[str, np.ndarray] = {}
        all_masks: dict[str, np.ndarray] = {}
        for name in self.view_names:
            X_batch = store[f"mod/{name}/X"][real_indices].astype(np.float32)
            mask_batch = store[f"mod/{name}/layers/{self.mask_layer_name}"][real_indices].astype(bool)

            scaler = self.scalers.get(name)
            if scaler is not None:
                X_batch = (X_batch - scaler["mean"]) / scaler["scale"]

            np.nan_to_num(X_batch, nan=0.0, copy=False)
            all_X[name] = X_batch
            all_masks[name] = mask_batch

        results = []
        for i, idx in enumerate(indices):
            t_views = {name: torch.from_numpy(all_X[name][i]) for name in self.view_names}
            results.append({
                "encoder_inputs": t_views,
                "decoder_targets": t_views,
                "missing_masks": {name: torch.from_numpy(all_masks[name][i]) for name in self.view_names},
                "conditionals": torch.from_numpy(self.conditionals[idx].astype(np.float32)),
                "tissue_labels": torch.from_numpy(self.tissue_labels[idx].astype(np.float32)),
                "source_ids": torch.tensor(self.source_ids[idx], dtype=torch.long),
                "sample_weights": torch.tensor(self.sample_weights[idx], dtype=torch.float32),
                "sample_name": self.sample_names[idx],
            })
        return results


class MuDataDataModule(pl.LightningDataModule):
    """Loads MuData (h5mu or zarr) format with in-memory or lazy strategies.

    - h5mu: all data loaded into memory.
    - zarr: metadata at setup, samples loaded lazily per batch.
    """

    def __init__(self, config: MOSAConfig):
        super().__init__()
        self.config = config
        self.scalers: dict[str, StandardScaler | None] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.batch_categories: list[str] = []
        self.tissue_categories: list[str] = []
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.class_weights: np.ndarray | None = None

    # Public API

    def setup(self, stage: str | None = None) -> None:
        data_path = Path(self.config.data_path)
        if data_path.suffix == ".zarr" or (data_path.is_dir() and not data_path.suffix):
            self._setup_zarr()
        else:
            self._setup_h5mu()

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.trainer.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def train_eval_dataloader(self) -> DataLoader:
        """Non-shuffled train dataloader for deterministic inference after training."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def full_dataloader(self) -> DataLoader:
        """DataLoader over all samples (train + val) in a fixed order."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        if self.val_dataset is not None:
            dataset = ConcatDataset([self.train_dataset, self.val_dataset])
        else:
            dataset = self.train_dataset
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Test dataloader not implemented")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("Predict dataloader not implemented")

    # Shared helpers

    def _process_obs(
        self, obs_df: pd.DataFrame, n_samples: int,
    ) -> dict:
        """Process obs metadata into conditionals, labels, weights, and splits.

        Returns
        -------
        dict
            Keys: conditionals, tissue_labels, source_ids, sample_weights,
            train_idx, val_idx, label_codes.
        """
        # Batch (model_type)
        if "model_type" not in obs_df.columns:
            raise ValueError("MuData .obs must contain 'model_type' column")
        batch_dummies = pd.get_dummies(obs_df["model_type"])
        self.batch_categories = list(batch_dummies.columns)

        # Tissue
        if "tissue" in obs_df.columns:
            tissue_dummies = pd.get_dummies(obs_df["tissue"])
            self.tissue_categories = list(tissue_dummies.columns)
        else:
            tissue_dummies = pd.DataFrame()
            logger.warning("No 'tissue' column found in .obs")

        # Mutations
        mutation_cols = [c for c in obs_df.columns if c.startswith("mutation_")]
        mutations = obs_df[mutation_cols].values.astype(np.float32) if mutation_cols else None

        # Concatenate conditionals
        cond_parts = [batch_dummies.values]
        if not tissue_dummies.empty:
            cond_parts.append(tissue_dummies.values)
        if mutations is not None:
            cond_parts.append(mutations)
        conditionals = np.concatenate(cond_parts, axis=1).astype(np.float32)

        # Tissue labels (one-hot)
        if not tissue_dummies.empty:
            tissue_labels = tissue_dummies.values.astype(np.float32)
        else:
            tissue_labels = np.zeros((n_samples, 1), dtype=np.float32)

        # Source IDs & stratified split
        model_type_cats = pd.Categorical(
            obs_df["model_type"],
            categories=sorted(obs_df["model_type"].unique()),
            ordered=True,
        )
        label_codes = np.asarray(model_type_cats.codes, dtype=np.intp)

        if self.config.test_size > 0:
            train_idx, val_idx = train_test_split(
                np.arange(n_samples),
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=label_codes,
            )
        else:
            train_idx = np.arange(n_samples)
            val_idx = np.array([], dtype=int)

        logger.debug("  train=%d, val=%d", len(train_idx), len(val_idx))

        # Class weights
        unique, counts = np.unique(label_codes[train_idx], return_counts=True)
        class_weights = np.zeros(len(unique), dtype=np.float32)
        for i, (cls, count) in enumerate(zip(unique, counts)):
            class_weights[cls] = len(train_idx) / (len(unique) * count)
        self.class_weights = class_weights

        sample_weights = class_weights[label_codes].astype(np.float32)

        return dict(
            conditionals=conditionals,
            tissue_labels=tissue_labels,
            source_ids=label_codes,
            sample_weights=sample_weights,
            train_idx=train_idx,
            val_idx=val_idx,
        )

    def _update_config_dims(self, conditionals: np.ndarray) -> None:
        for view_name in self.config.views:
            vc = self.config.views[view_name]
            if vc.input_dim == 0:
                vc.input_dim = len(self.feature_names[view_name])
            if vc.output_dim == 0:
                vc.output_dim = len(self.feature_names[view_name])
        if self.config.conditional_dim == 0:
            self.config.conditional_dim = conditionals.shape[1]
        if self.config.n_batches == 0:
            self.config.n_batches = len(self.batch_categories)

    # h5mu path (in-memory)

    def _setup_h5mu(self) -> None:
        import mudata

        logger.info("Loading MuData (h5mu) from %s", self.config.data_path)
        mdata = mudata.read(self.config.data_path)
        self._verify_mudata_structure(mdata)

        # Extract dense arrays per modality
        omics_all: dict[str, np.ndarray] = {}
        masks_all: dict[str, np.ndarray] = {}

        for view_name in self.config.views:
            adata = mdata.mod[view_name]
            X = adata.X
            if issparse(X):
                X = X.toarray()
            X = X.astype(np.float32)

            mask = adata.layers[self.config.mask_layer_name]
            if issparse(mask):
                mask = mask.toarray()
            mask = mask.astype(bool)

            if view_name in mdata.obsm:
                presence = np.asarray(mdata.obsm[view_name]).flatten().astype(bool)
                X[~presence] = 0.0
                mask[~presence] = False

            omics_all[view_name] = X
            masks_all[view_name] = mask
            self.feature_names[view_name] = list(adata.var_names)

        obs_df = mdata.obs.loc[:, ~mdata.obs.columns.str.match(r"^Unnamed")]
        sample_names = list(obs_df.index)

        meta = self._process_obs(obs_df, len(sample_names))
        train_idx, val_idx = meta["train_idx"], meta["val_idx"]

        # Fit scalers on training data
        for view_name, X in omics_all.items():
            if self.config.views[view_name].discrete:
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[view_name] = None
            else:
                scaler = StandardScaler()
                scaler.fit(X[train_idx])
                X = scaler.transform(X)
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[view_name] = scaler
            omics_all[view_name] = X

        self._update_config_dims(meta["conditionals"])

        self.train_dataset = self._create_inmemory_dataset(
            omics_all, masks_all, meta, sample_names, train_idx,
        )
        if len(val_idx) > 0:
            self.val_dataset = self._create_inmemory_dataset(
                omics_all, masks_all, meta, sample_names, val_idx,
            )

        logger.info("h5mu setup complete: %d train, %d val", len(train_idx), len(val_idx))

    def _create_inmemory_dataset(
        self,
        omics_all: dict[str, np.ndarray],
        masks_all: dict[str, np.ndarray],
        meta: dict,
        sample_names: list[str],
        indices: np.ndarray,
    ) -> MOSADataset:
        return MOSADataset(
            omics_data={k: v[indices] for k, v in omics_all.items()},
            masks={k: v[indices] for k, v in masks_all.items()},
            conditionals=meta["conditionals"][indices],
            tissue_labels=meta["tissue_labels"][indices],
            source_ids=meta["source_ids"][indices],
            sample_weights=meta["sample_weights"][indices],
            sample_names=[sample_names[i] for i in indices],
            omic_names=list(self.config.views.keys()),
        )

    def _verify_mudata_structure(self, mdata: object) -> None:
        """Validate MuData structure has required columns and views."""
        if "model_type" not in mdata.obs.columns:
            raise ValueError(
                f"MuData .obs missing 'model_type' column. "
                f"Available: {list(mdata.obs.columns)}"
            )
        for view_name in self.config.views:
            if view_name not in mdata.mod:
                raise ValueError(
                    f"View '{view_name}' not in MuData. Available: {list(mdata.mod.keys())}"
                )
            if self.config.mask_layer_name not in mdata.mod[view_name].layers:
                raise ValueError(
                    f"Mask layer '{self.config.mask_layer_name}' not in '{view_name}'. "
                    f"Available: {list(mdata.mod[view_name].layers.keys())}"
                )

    # zarr path (lazy loading)

    @staticmethod
    def _zarr_index_key(group) -> str:
        """Return the key that stores the index for a zarr obs/var group.

        AnnData/MuData zarr stores record the index column name in the
        ``_index`` attribute of the group.  The actual data lives under
        ``group[attrs["_index"]]``, **not** under ``group["_index"]``
        (unless the DataFrame index happened to be named ``_index``).
        """
        return group.attrs.get("_index", "_index")

    @staticmethod
    def _read_zarr_column(group) -> np.ndarray:
        """Decode a single obs/var column from MuData's zarr encoding."""
        if isinstance(group, zarr.Array):
            return np.asarray(group)

        keys = set(group.keys())
        if {"categories", "codes"} <= keys:
            cats_node = group["categories"]
            if isinstance(cats_node, zarr.Group) and "values" in cats_node:
                cats = np.asarray(cats_node["values"])
            else:
                cats = np.asarray(cats_node)
            codes = np.asarray(group["codes"])
            return cats[codes]

        if "values" in keys:
            return np.asarray(group["values"])

        raise ValueError(f"Cannot decode zarr column with keys {keys}")

    def _setup_zarr(self) -> None:
        logger.info("Loading MuData (zarr, lazy) from %s", self.config.data_path)
        store = zarr.open_group(self.config.data_path, mode="r")

        # 1. Read obs metadata only (small)
        obs_group = store["obs"]
        obs_idx_key = self._zarr_index_key(obs_group)
        sample_names = list(self._read_zarr_column(obs_group[obs_idx_key]))
        obs_dict = {"model_type": self._read_zarr_column(obs_group["model_type"])}
        if "tissue" in obs_group:
            obs_dict["tissue"] = self._read_zarr_column(obs_group["tissue"])
        for key in obs_group:
            if key.startswith("mutation_"):
                obs_dict[key] = self._read_zarr_column(obs_group[key])
        obs_df = pd.DataFrame(obs_dict, index=sample_names)

        # 2. Read feature names per view (small)
        for view_name in self.config.views:
            if f"mod/{view_name}" not in store:
                raise ValueError(f"View '{view_name}' not found in zarr store")
            var_group = store[f"mod/{view_name}/var"]
            var_idx_key = self._zarr_index_key(var_group)
            self.feature_names[view_name] = list(self._read_zarr_column(var_group[var_idx_key]))

        # 3. Shared metadata processing
        meta = self._process_obs(obs_df, len(sample_names))
        train_idx, val_idx = meta["train_idx"], meta["val_idx"]

        # 4. Fit scalers on subsample from zarr (avoids loading all data)
        rng = np.random.RandomState(self.config.random_seed)
        frac = self.config.scaler_sample_frac

        for view_name in self.config.views:
            if self.config.views[view_name].discrete:
                self.scalers[view_name] = None
                continue

            X_zarr = store[f"mod/{view_name}/X"]

            if frac < 1.0:
                n_sub = max(1, int(len(train_idx) * frac))
                sub_idx = sorted(rng.choice(train_idx, size=n_sub, replace=False))
            else:
                sub_idx = sorted(train_idx)

            logger.debug("Fitting scaler for '%s' on %d samples", view_name, len(sub_idx))
            X_sub = np.asarray(X_zarr[sub_idx]).astype(np.float32)
            X_sub = np.nan_to_num(X_sub, nan=0.0)

            scaler = StandardScaler()
            scaler.fit(X_sub)
            self.scalers[view_name] = scaler

        del store  # close zarr handle; LazyZarrDataset opens its own

        self._update_config_dims(meta["conditionals"])

        # 5. Package scaler params as plain arrays (pickle-safe for workers)
        scaler_dicts: dict[str, dict[str, np.ndarray] | None] = {}
        for name, scaler in self.scalers.items():
            if scaler is not None:
                scaler_dicts[name] = {
                    "mean": scaler.mean_.astype(np.float32),
                    "scale": scaler.scale_.astype(np.float32),
                }
            else:
                scaler_dicts[name] = None

        # 6. Create lazy datasets
        view_names = list(self.config.views.keys())

        self.train_dataset = LazyZarrDataset(
            zarr_path=self.config.data_path,
            view_names=view_names,
            indices=train_idx,
            conditionals=meta["conditionals"][train_idx],
            tissue_labels=meta["tissue_labels"][train_idx],
            source_ids=meta["source_ids"][train_idx],
            sample_weights=meta["sample_weights"][train_idx],
            sample_names=[sample_names[i] for i in train_idx],
            scalers=scaler_dicts,
            mask_layer_name=self.config.mask_layer_name,
        )

        if len(val_idx) > 0:
            self.val_dataset = LazyZarrDataset(
                zarr_path=self.config.data_path,
                view_names=view_names,
                indices=val_idx,
                conditionals=meta["conditionals"][val_idx],
                tissue_labels=meta["tissue_labels"][val_idx],
                source_ids=meta["source_ids"][val_idx],
                sample_weights=meta["sample_weights"][val_idx],
                sample_names=[sample_names[i] for i in val_idx],
                scalers=scaler_dicts,
                mask_layer_name=self.config.mask_layer_name,
            )

        logger.info("zarr setup complete: %d train, %d val", len(train_idx), len(val_idx))
