from __future__ import annotations

import logging

import numpy as np
import torch
import zarr
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import pytorch_lightning as pl

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.data.io import _zarr_view_mask_key, _zarr_view_x_key
from mosa.models.mosa.config import MOSAConfig

logger = logging.getLogger(__name__)


# Mask-aware standardization: mean/std computed only from observed entries,
# so missing values (NaN placeholders) never bias the fitted statistics.

def _feature_mean(X: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Compute per-feature mean using only observed values."""
    observed = np.where(masks, X, 0.0)
    counts = masks.sum(axis=0).astype(np.float32)
    sums = observed.sum(axis=0)
    mean = np.zeros(X.shape[1], dtype=np.float32)
    valid = counts > 0
    mean[valid] = sums[valid] / counts[valid]
    return mean


def _feature_std(X: np.ndarray, masks: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Compute per-feature standard deviation using only observed values."""
    observed = np.where(masks, X, 0.0)
    counts = masks.sum(axis=0).astype(np.float32)
    centered = observed - mean
    sq_sums = np.square(centered).sum(axis=0)
    var = np.zeros(X.shape[1], dtype=np.float32)
    valid = counts > 0
    var[valid] = sq_sums[valid] / counts[valid]
    scale = np.sqrt(var, dtype=np.float32)
    scale[scale == 0.0] = 1.0
    return scale


def _fit_standardization(X: np.ndarray, masks: np.ndarray) -> dict[str, np.ndarray]:
    """Fit mask-aware mean/std statistics for per-feature standardization."""
    mean = _feature_mean(X, masks)
    scale = _feature_std(X, masks, mean)
    return {"mean": mean, "scale": scale}


def _apply_standardization(
    X: np.ndarray,
    masks: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    """Standardize observed entries and fill missing entries with zero placeholder."""
    mean = stats["mean"]
    scale = stats["scale"]
    standardized = np.where(masks, (X - mean) / scale, 0.0)
    np.nan_to_num(standardized, nan=0.0, copy=False)
    return standardized


# Group centering / imputation (per model_type), mirroring MOFA's group handling.

def _fit_group_centering(
    X: np.ndarray,
    masks: np.ndarray,
    source_ids: np.ndarray,
    n_groups: int,
) -> dict[str, np.ndarray]:
    """Fit per-group feature means plus a global fallback mean.

    Falls back to the global (all-groups) mean on a per-feature basis whenever
    a group has zero observed values for that specific feature (e.g. a group
    absent from a subsample, or a feature entirely unmeasured within a group,
    such as CRISPR having no data at all for a model_type that lacks it).
    """
    global_mean = _feature_mean(X, masks)
    group_means = np.tile(global_mean, (n_groups, 1)).astype(np.float32)
    for g_idx in range(n_groups):
        sample_mask = source_ids == g_idx
        if not np.any(sample_mask):
            continue
        g_counts = masks[sample_mask].sum(axis=0)
        g_mean = _feature_mean(X[sample_mask], masks[sample_mask])
        valid = g_counts > 0
        group_means[g_idx, valid] = g_mean[valid]
    return {"group_means": group_means, "global_mean": global_mean}


def _group_mean_for(g_idx: int, centering: dict[str, np.ndarray]) -> np.ndarray:
    group_means = centering["group_means"]
    if 0 <= int(g_idx) < group_means.shape[0]:
        return group_means[int(g_idx)]
    return centering["global_mean"]


def _apply_group_centering(
    X: np.ndarray,
    masks: np.ndarray,
    source_ids: np.ndarray,
    centering: dict[str, np.ndarray],
) -> np.ndarray:
    """Center observed entries by the mean of their model_type group."""
    centered = np.array(X, copy=True)
    source_ids = np.asarray(source_ids)

    for g_idx in np.unique(source_ids):
        row_mask = source_ids == g_idx
        mean = _group_mean_for(g_idx, centering)
        present = masks[row_mask]
        centered[row_mask] = np.where(present, centered[row_mask] - mean, 0.0)

    np.nan_to_num(centered, nan=0.0, copy=False)
    return centered


def _apply_group_mean_imputation(
    X: np.ndarray,
    masks: np.ndarray,
    source_ids: np.ndarray,
    means: dict[str, np.ndarray],
) -> np.ndarray:
    """Fill missing entries with their model_type group's mean (raw scale).

    Falls back to the global mean (computed across all other groups) when a
    group has no observed values for a feature.
    """
    filled = np.array(X, copy=True)
    source_ids = np.asarray(source_ids)

    for g_idx in np.unique(source_ids):
        row_mask = source_ids == g_idx
        mean = _group_mean_for(g_idx, means)
        present = masks[row_mask]
        filled[row_mask] = np.where(present, filled[row_mask], mean)

    np.nan_to_num(filled, nan=0.0, copy=False)
    return filled


def _apply_group_inverse_centering(
    X: np.ndarray,
    source_ids: np.ndarray,
    centering: dict[str, np.ndarray],
) -> np.ndarray:
    """Undo group centering using the sample's model_type group."""
    restored = np.array(X, copy=True)
    source_ids = np.asarray(source_ids)

    for g_idx in np.unique(source_ids):
        row_mask = source_ids == g_idx
        mean = _group_mean_for(g_idx, centering)
        restored[row_mask] = restored[row_mask] + mean

    return restored


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
        self.omics = {k: torch.from_numpy(v) for k, v in omics_data.items()}
        self.masks = {k: torch.from_numpy(v) for k, v in masks.items()}
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
    """Lazy-loading dataset from MuData zarr store; each worker opens its own handle.

    NOTE: currently unwired — no caller passes ``zarr_path`` to MOSADataModule,
    so this path is never exercised. Before enabling it, verify the index logic:
    ``__getitems__`` reads ``store[...][sorted_real]`` by absolute store
    position, while the train/val split produces randomized/stratified indices,
    so the two must be reconciled or rows will misalign with their labels.
    """

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
        group_centering: dict[str, dict[str, np.ndarray] | None] | None = None,
        impute_means: dict[str, dict[str, np.ndarray] | None] | None = None,
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
        self.group_centering = group_centering or {}
        self.impute_means = impute_means or {}
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
        """Batched read: one zarr slice per view for the whole batch.

        Indices are sorted before the zarr read so the I/O is sequential
        (contiguous chunks). Within-batch order is irrelevant for SGD, so
        the returned list follows the sorted order directly.
        """
        store = self._get_store()

        order = np.argsort(self.indices[indices])
        sorted_indices = [indices[i] for i in order]
        sorted_real = self.indices[sorted_indices]

        all_X: dict[str, np.ndarray] = {}
        all_masks: dict[str, np.ndarray] = {}
        for name in self.view_names:
            X_batch = store[_zarr_view_x_key(name)][sorted_real].astype(np.float32)
            mask_batch = store[_zarr_view_mask_key(name, self.mask_layer_name)][sorted_real].astype(bool)

            centering = self.group_centering.get(name)
            means = self.impute_means.get(name)
            if centering is not None:
                X_batch = _apply_group_centering(
                    X_batch, mask_batch, self.source_ids[sorted_indices], centering,
                )
            elif means is not None:
                X_batch = _apply_group_mean_imputation(
                    X_batch, mask_batch, self.source_ids[sorted_indices], means,
                )
            else:
                scaler = self.scalers.get(name)
                if scaler is not None:
                    X_batch = (X_batch - scaler["mean"]) / scaler["scale"]

            np.nan_to_num(X_batch, nan=0.0, copy=False)
            all_X[name] = X_batch
            all_masks[name] = mask_batch

        results = []
        for i, idx in enumerate(sorted_indices):
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


class MOSADataModule(pl.LightningDataModule):
    """VAE-internal data handler: scaling, batching, and DataLoader creation.

    Receives already-loaded MultiOmicDataset objects. Does not read files.
    """

    def __init__(
        self,
        train_data: MultiOmicDataset | None,
        val_data: MultiOmicDataset | None,
        data_cfg: DataConfig,
        model_cfg: MOSAConfig,
        zarr_path: str | None = None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.zarr_path = zarr_path

        self.scalers: dict[str, dict[str, np.ndarray] | None] = {}
        self.group_centering: dict[str, dict[str, np.ndarray] | None] = {}
        self.impute_means: dict[str, dict[str, np.ndarray] | None] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.batch_categories: list[str] = []
        self.tissue_categories: list[str] = []
        self.mutation_columns: list[str] = []
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.class_weights: np.ndarray | None = None
        self._conditionals_train: np.ndarray | None = None

    # Dimension properties (replace config mutation)

    @property
    def conditional_dim(self) -> int:
        if self._conditionals_train is None:
            raise RuntimeError("Call setup() before accessing conditional_dim")
        return self._conditionals_train.shape[1]

    @property
    def n_batches(self) -> int:
        return len(self.batch_categories)

    @property
    def view_input_dims(self) -> dict[str, int]:
        return {name: len(fnames) for name, fnames in self.feature_names.items()}

    # Public API

    def setup(self, stage: str | None = None) -> None:
        """Fit preprocessing state on train_data and create torch Dataset objects."""
        if self.train_dataset is not None:
            return
        train = self.train_data
        self.feature_names = {k: list(v) for k, v in train.feature_names.items()}

        meta = self._process_obs(train.metadata, train.n_samples)
        self._conditionals_train = meta["conditionals"]

        if self.zarr_path is not None:
            self._setup_lazy_zarr(meta)
        else:
            self._setup_inmemory(train, meta)

    def setup_inference(self, source: "MOSADataModule") -> None:
        """Set up a dataset for inference using another datamodule's fitted state.

        Applies `source`'s scalers, feature names, and batch/tissue categories
        without refitting anything and without mutating `source`. Use this
        instead of `setup()` for transform()/reconstruct() calls so that new
        data is standardized with the training statistics.
        """
        if self.train_dataset is not None:
            return
        if self.zarr_path is not None:
            raise NotImplementedError("Inference setup for lazy zarr data is not implemented")

        self.scalers = source.scalers
        self.group_centering = source.group_centering
        self.impute_means = source.impute_means
        self.feature_names = source.feature_names
        self.batch_categories = source.batch_categories
        self.tissue_categories = source.tissue_categories
        self.mutation_columns = source.mutation_columns
        self.class_weights = source.class_weights

        train = self.train_data
        meta = self._process_obs_readonly(train.metadata)
        self._conditionals_train = meta["conditionals"]
        self.train_dataset = self._build_dataset_readonly(train, meta)

    def _apply_fitted_preprocessing(
        self, view_name: str, X: np.ndarray, mask: np.ndarray, source_ids: np.ndarray,
    ) -> np.ndarray:
        """Apply this view's already-fitted preprocessing (no fitting)."""
        if view_name in self.data_cfg.discrete_views:
            return np.nan_to_num(X, nan=0.0)

        mode = self.model_cfg.preprocessing_mode
        if mode == "center":
            centering = self.group_centering.get(view_name)
            if centering is None:
                return np.nan_to_num(X, nan=0.0)
            return _apply_group_centering(X, mask, source_ids, centering)
        if mode == "none":
            means = self.impute_means.get(view_name)
            if means is None:
                return np.nan_to_num(X, nan=0.0)
            return _apply_group_mean_imputation(X, mask, source_ids, means)

        stats = self.scalers.get(view_name)
        if stats is None:
            return np.nan_to_num(X, nan=0.0)
        return _apply_standardization(X, mask, stats)

    def _build_dataset_readonly(self, data: MultiOmicDataset, meta: dict) -> "MOSADataset":
        """Apply already-fitted preprocessing (no fitting) and build a MOSADataset."""
        omics = {
            view_name: self._apply_fitted_preprocessing(
                view_name, X, data.masks[view_name], meta["source_ids"],
            )
            for view_name, X in data.views.items()
        }

        return MOSADataset(
            omics_data=omics,
            masks=data.masks,
            conditionals=meta["conditionals"],
            tissue_labels=meta["tissue_labels"],
            source_ids=meta["source_ids"],
            sample_weights=meta["sample_weights"],
            sample_names=data.sample_names,
            omic_names=list(self.data_cfg.views),
        )

    def inverse_transform_view(
        self,
        view_name: str,
        values: np.ndarray,
        source_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        """Restore a transformed view to the original scale."""
        mode = self.model_cfg.preprocessing_mode
        if mode == "center":
            centering = self.group_centering.get(view_name)
            if centering is None:
                return np.array(values, copy=True)
            if source_ids is None:
                raise ValueError("source_ids are required to invert group centering")
            return _apply_group_inverse_centering(values, source_ids, centering)

        if mode == "none":
            return np.array(values, copy=True)

        stats = self.scalers.get(view_name)
        if stats is not None:
            return values * stats["scale"] + stats["mean"]
        return np.array(values, copy=True)

    def _setup_inmemory(self, train: MultiOmicDataset, meta: dict) -> None:
        import time
        t0 = time.perf_counter()
        mode = self.model_cfg.preprocessing_mode

        omics_train = {}
        for view_name, X in train.views.items():
            tv = time.perf_counter()
            mask = train.masks[view_name]
            if view_name in self.data_cfg.discrete_views:
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[view_name] = None
                self.group_centering[view_name] = None
                self.impute_means[view_name] = None
            elif mode == "center":
                centering = _fit_group_centering(X, mask, meta["source_ids"], len(self.batch_categories))
                self.group_centering[view_name] = centering
                self.scalers[view_name] = None
                self.impute_means[view_name] = None
                X = _apply_group_centering(X, mask, meta["source_ids"], centering)
            elif mode == "none":
                means = _fit_group_centering(X, mask, meta["source_ids"], len(self.batch_categories))
                self.impute_means[view_name] = means
                self.scalers[view_name] = None
                self.group_centering[view_name] = None
                X = _apply_group_mean_imputation(X, mask, meta["source_ids"], means)
            else:
                stats = _fit_standardization(X, mask)
                self.scalers[view_name] = stats
                self.group_centering[view_name] = None
                self.impute_means[view_name] = None
                X = _apply_standardization(X, mask, stats)
            omics_train[view_name] = X
            logger.debug("  view '%s': preprocessed in %.2fs", view_name, time.perf_counter() - tv)

        self.train_dataset = MOSADataset(
            omics_data=omics_train,
            masks=train.masks,
            conditionals=meta["conditionals"],
            tissue_labels=meta["tissue_labels"],
            source_ids=meta["source_ids"],
            sample_weights=meta["sample_weights"],
            sample_names=train.sample_names,
            omic_names=list(self.data_cfg.views),
        )

        if self.val_data is not None:
            val_meta = self._process_obs_readonly(self.val_data.metadata)
            self.val_dataset = self._build_dataset_readonly(self.val_data, val_meta)

        n_train = self.train_data.n_samples
        n_val = self.val_data.n_samples if self.val_data is not None else 0
        logger.info("Setup complete: %d train, %d val (%.2fs)", n_train, n_val, time.perf_counter() - t0)

    def _setup_lazy_zarr(self, meta: dict) -> None:
        """Set up LazyZarrDataset for large zarr stores."""
        rng = np.random.RandomState(self.model_cfg.random_seed)
        frac = self.model_cfg.scaler_sample_frac
        store = zarr.open_group(self.zarr_path, mode="r")
        mode = self.model_cfg.preprocessing_mode

        n_train = self.train_data.n_samples
        all_train_idx = np.arange(n_train)

        for view_name in self.data_cfg.views:
            if view_name in self.data_cfg.discrete_views:
                self.scalers[view_name] = None
                self.group_centering[view_name] = None
                self.impute_means[view_name] = None
                continue

            X_zarr = store[_zarr_view_x_key(view_name)]
            mask_zarr = store[_zarr_view_mask_key(view_name, self.data_cfg.mask_layer_name)]
            if frac < 1.0:
                n_sub = max(1, int(n_train * frac))
                sub_idx = sorted(rng.choice(all_train_idx, size=n_sub, replace=False))
            else:
                sub_idx = sorted(all_train_idx)

            X_sub = np.asarray(X_zarr[sub_idx]).astype(np.float32)
            M_sub = np.asarray(mask_zarr[sub_idx]).astype(bool)

            if mode == "center":
                logger.debug("Fitting group centering for '%s' on %d samples", view_name, len(sub_idx))
                self.group_centering[view_name] = _fit_group_centering(
                    X_sub, M_sub, meta["source_ids"][sub_idx], len(self.batch_categories),
                )
                self.scalers[view_name] = None
                self.impute_means[view_name] = None
            elif mode == "none":
                logger.debug("Fitting group-mean imputation for '%s' on %d samples", view_name, len(sub_idx))
                self.impute_means[view_name] = _fit_group_centering(
                    X_sub, M_sub, meta["source_ids"][sub_idx], len(self.batch_categories),
                )
                self.scalers[view_name] = None
                self.group_centering[view_name] = None
            else:
                logger.debug("Fitting standardization for '%s' on %d samples", view_name, len(sub_idx))
                self.scalers[view_name] = _fit_standardization(X_sub, M_sub)
                self.group_centering[view_name] = None
                self.impute_means[view_name] = None

        del store

        scaler_dicts: dict[str, dict[str, np.ndarray] | None] = {}
        for name, stats in self.scalers.items():
            if stats is None:
                scaler_dicts[name] = None
            else:
                scaler_dicts[name] = {
                    "mean": stats["mean"].astype(np.float32),
                    "scale": stats["scale"].astype(np.float32),
                }

        view_names = list(self.data_cfg.views)

        self.train_dataset = LazyZarrDataset(
            zarr_path=self.zarr_path,
            view_names=view_names,
            indices=all_train_idx,
            conditionals=meta["conditionals"],
            tissue_labels=meta["tissue_labels"],
            source_ids=meta["source_ids"],
            sample_weights=meta["sample_weights"],
            sample_names=self.train_data.sample_names,
            scalers=scaler_dicts,
            group_centering=self.group_centering,
            impute_means=self.impute_means,
            mask_layer_name=self.data_cfg.mask_layer_name,
        )

        if self.val_data is not None:
            val_meta = self._process_obs_readonly(self.val_data.metadata)
            val_idx = np.arange(self.val_data.n_samples)
            self.val_dataset = LazyZarrDataset(
                zarr_path=self.zarr_path,
                view_names=view_names,
                indices=val_idx,
                conditionals=val_meta["conditionals"],
                tissue_labels=val_meta["tissue_labels"],
                source_ids=val_meta["source_ids"],
                sample_weights=val_meta["sample_weights"],
                sample_names=self.val_data.sample_names,
                scalers=scaler_dicts,
                group_centering=self.group_centering,
                impute_means=self.impute_means,
                mask_layer_name=self.data_cfg.mask_layer_name,
            )

        n_val = self.val_data.n_samples if self.val_data is not None else 0
        logger.info("zarr lazy setup complete: %d train, %d val", n_train, n_val)

    def _loader_kwargs(self) -> dict:
        nw = self.model_cfg.num_workers
        accelerator = self.model_cfg.accelerator
        # pin_memory only speeds up host->GPU transfer; on CPU it's a no-op
        # that still pays for pinned allocation, so skip it there.
        pin_memory = accelerator == "gpu" or (accelerator == "auto" and torch.cuda.is_available())
        kwargs: dict = {
            "num_workers": nw,
            "pin_memory": pin_memory,
        }
        if nw > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def _get_weighted_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        """Build a WeightedRandomSampler from the sample_weights computed in _process_obs()."""
        if isinstance(dataset, MOSADataset):
            sample_weights = dataset.sample_weights.numpy()
        elif isinstance(dataset, LazyZarrDataset):
            sample_weights = dataset.sample_weights
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(dataset),
            replacement=True,
        )
        return sampler

    def teardown(self, stage: str | None = None) -> None:
        """Close any open zarr store handles held by lazy datasets."""
        for ds in [self.train_dataset, self.val_dataset]:
            if isinstance(ds, LazyZarrDataset) and ds._store is not None:
                ds._store.store.close()
                ds._store = None

    def _serialize_group_stats(
        self, stats: dict[str, dict[str, np.ndarray] | None],
    ) -> dict:
        return {
            name: (
                None if s is None else {
                    "group_means": s["group_means"].tolist(),
                    "global_mean": s["global_mean"].tolist(),
                }
            )
            for name, s in stats.items()
        }

    def _deserialize_group_stats(self, state: dict) -> dict[str, dict[str, np.ndarray] | None]:
        return {
            name: (
                None if s is None else {
                    "group_means": np.array(s["group_means"], dtype=np.float32),
                    "global_mean": np.array(s["global_mean"], dtype=np.float32),
                }
            )
            for name, s in state.items()
        }

    def state_dict(self) -> dict:
        """Return serialisable preprocessing state for checkpoint saving."""
        scalers = {}
        for name, stats in self.scalers.items():
            if stats is None:
                scalers[name] = None
            else:
                scalers[name] = {
                    "mean": stats["mean"].tolist(),
                    "scale": stats["scale"].tolist(),
                }
        return {
            "scalers": scalers,
            "group_centering": self._serialize_group_stats(self.group_centering),
            "impute_means": self._serialize_group_stats(self.impute_means),
            "batch_categories": self.batch_categories,
            "tissue_categories": self.tissue_categories,
            "mutation_columns": self.mutation_columns,
            "feature_names": self.feature_names,
            "class_weights": (
                self.class_weights.tolist() if self.class_weights is not None else None
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore preprocessing state from a saved checkpoint."""
        self.batch_categories = state["batch_categories"]
        self.tissue_categories = state["tissue_categories"]
        self.mutation_columns = state.get("mutation_columns", [])
        self.feature_names = state["feature_names"]
        class_weights = state.get("class_weights")
        self.class_weights = (
            np.array(class_weights, dtype=np.float32) if class_weights is not None else None
        )
        self.scalers = {}
        for name, s in state["scalers"].items():
            if s is None:
                self.scalers[name] = None
            else:
                self.scalers[name] = {
                    "mean": np.array(s["mean"], dtype=np.float32),
                    "scale": np.array(s["scale"], dtype=np.float32),
                }
        self.group_centering = self._deserialize_group_stats(state.get("group_centering", {}))
        self.impute_means = self._deserialize_group_stats(state.get("impute_means", {}))

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")

        loader_kwargs = self._loader_kwargs()

        if self.model_cfg.weighted_random_sampler:
            # Create weighted sampler to balance model_type categories in each batch
            sampler = self._get_weighted_sampler(self.train_dataset)
            return DataLoader(
                self.train_dataset,
                batch_size=self.model_cfg.batch_size,
                sampler=sampler,
                **loader_kwargs,
            )
        else:
            # Use default sequential sampling with shuffle
            return DataLoader(
                self.train_dataset,
                batch_size=self.model_cfg.batch_size,
                shuffle=True,
                **loader_kwargs,
            )

    def val_dataloader(self) -> DataLoader | None:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.model_cfg.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def train_eval_dataloader(self) -> DataLoader:
        """Non-shuffled train dataloader for deterministic inference after training."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
        return DataLoader(
            self.train_dataset,
            batch_size=self.model_cfg.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
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
            batch_size=self.model_cfg.batch_size,
            shuffle=False,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader:
        raise NotImplementedError("Test dataloader not implemented")

    def predict_dataloader(self) -> DataLoader:
        raise NotImplementedError("Predict dataloader not implemented")

    # Metadata processing

    def _process_obs(self, obs_df, n_samples: int) -> dict:
        """Fit batch/tissue categories and compute conditionals, labels, weights.

        Mutates self.batch_categories, self.tissue_categories, self.class_weights.

        Returns
        -------
        dict
            Keys: conditionals, tissue_labels, source_ids, sample_weights.
        """
        import pandas as pd

        # Batch (model_type) - always included
        if "model_type" not in obs_df.columns:
            raise ValueError("MuData .obs must contain 'model_type' column")
        batch_dummies = pd.get_dummies(obs_df["model_type"])
        self.batch_categories = list(batch_dummies.columns)

        # Tissue (included if use_tissue=True)
        if self.data_cfg.use_tissue and "tissue" in obs_df.columns:
            tissue_dummies = pd.get_dummies(obs_df["tissue"])
            self.tissue_categories = list(tissue_dummies.columns)
        else:
            tissue_dummies = pd.DataFrame()
            if self.data_cfg.use_tissue and "tissue" not in obs_df.columns:
                logger.warning("use_tissue=True but no 'tissue' column found in .obs")

        # Mutations (included if use_mutations=True)
        if self.data_cfg.use_mutations:
            mutation_cols = [c for c in obs_df.columns if c.startswith("mutation_")]
            mutations = obs_df[mutation_cols].values.astype(np.float32) if mutation_cols else None
        else:
            mutation_cols = []
            mutations = None
        self.mutation_columns = mutation_cols

        cond_parts = [batch_dummies.values]
        if not tissue_dummies.empty:
            cond_parts.append(tissue_dummies.values)
        if mutations is not None:
            cond_parts.append(mutations)
        conditionals = np.concatenate(cond_parts, axis=1).astype(np.float32)

        if not tissue_dummies.empty:
            tissue_labels = tissue_dummies.values.astype(np.float32)
        else:
            tissue_labels = np.zeros((n_samples, 1), dtype=np.float32)

        # Align codes with batch_categories so unused-but-defined Categorical
        # levels do not mismatch the discriminator output size.
        model_type_cats = pd.Categorical(
            obs_df["model_type"],
            categories=self.batch_categories,
            ordered=True,
        )
        label_codes = np.asarray(model_type_cats.codes, dtype=np.intp)

        n_classes = len(self.batch_categories)
        class_weights = np.ones(n_classes, dtype=np.float32)
        unique, counts = np.unique(label_codes, return_counts=True)
        for cls, count in zip(unique, counts):
            class_weights[cls] = n_samples / (len(unique) * count)
        self.class_weights = class_weights

        sample_weights = class_weights[label_codes].astype(np.float32)

        return dict(
            conditionals=conditionals,
            tissue_labels=tissue_labels,
            source_ids=label_codes,
            sample_weights=sample_weights,
        )

    def _process_obs_readonly(self, obs_df) -> dict:
        """Compute conditionals for val data using already-fitted categories.

        Does not update self.batch_categories or self.class_weights.
        """
        import pandas as pd

        batch_dummies = pd.get_dummies(obs_df["model_type"]).reindex(
            columns=self.batch_categories, fill_value=0
        )

        if self.tissue_categories:
            tissue_dummies = pd.get_dummies(obs_df.get("tissue", pd.Series(dtype=str))).reindex(
                columns=self.tissue_categories, fill_value=0
            )
        else:
            tissue_dummies = pd.DataFrame()

        if self.data_cfg.use_mutations and self.mutation_columns:
            # Reindex against the training-time column list so a different
            # order or set of mutation_* columns in new data can't silently
            # misalign the conditional block (or is cleanly zero-filled).
            mutations = obs_df.reindex(
                columns=self.mutation_columns, fill_value=0
            ).values.astype(np.float32)
        else:
            mutations = None

        cond_parts = [batch_dummies.values]
        if not tissue_dummies.empty:
            cond_parts.append(tissue_dummies.values)
        if mutations is not None:
            cond_parts.append(mutations)
        conditionals = np.concatenate(cond_parts, axis=1).astype(np.float32)

        if not tissue_dummies.empty:
            tissue_labels = tissue_dummies.values.astype(np.float32)
        else:
            tissue_labels = np.zeros((len(obs_df), 1), dtype=np.float32)

        model_type_cats = pd.Categorical(
            obs_df["model_type"],
            categories=self.batch_categories,
            ordered=True,
        )
        label_codes = np.asarray(model_type_cats.codes, dtype=np.intp)
        # code -1 marks a model_type absent from the training categories; left
        # unchecked it would negative-index class_weights (wrapping to the last
        # class) and misalign the conditional block. Fail clearly instead.
        if (label_codes < 0).any():
            unseen = sorted(set(obs_df["model_type"][label_codes < 0]))
            raise ValueError(
                f"model_type value(s) {unseen} not seen during fit; known "
                f"categories: {self.batch_categories}"
            )
        if self.class_weights is not None:
            sample_weights = self.class_weights[label_codes].astype(np.float32)
        else:
            sample_weights = np.ones(len(label_codes), dtype=np.float32)

        return dict(
            conditionals=conditionals,
            tissue_labels=tissue_labels,
            source_ids=label_codes,
            sample_weights=sample_weights,
        )
