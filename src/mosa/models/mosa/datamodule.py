from __future__ import annotations

import logging

import numpy as np
import torch
import zarr
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import pytorch_lightning as pl
from sklearn.preprocessing import StandardScaler

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.mosa.config import MOSAVAEConfig

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
            X_batch = store[f"mod/{name}/X"][sorted_real].astype(np.float32)
            mask_batch = store[f"mod/{name}/layers/{self.mask_layer_name}"][sorted_real].astype(bool)

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
        model_cfg: MOSAVAEConfig,
        zarr_path: str | None = None,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self.zarr_path = zarr_path

        self.scalers: dict[str, StandardScaler | None] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.batch_categories: list[str] = []
        self.tissue_categories: list[str] = []
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
        """Fit scalers on train_data and create torch Dataset objects."""
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

    def _setup_inmemory(self, train: MultiOmicDataset, meta: dict) -> None:
        import time
        t0 = time.perf_counter()

        omics_train = {}
        for view_name, X in train.views.items():
            tv = time.perf_counter()
            if view_name in self.data_cfg.discrete_views:
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[view_name] = None
            else:
                scaler = StandardScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[view_name] = scaler
            omics_train[view_name] = X
            logger.debug("  view '%s': scaled and imputed in %.2fs", view_name, time.perf_counter() - tv)

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
            omics_val = {}
            for view_name, X in self.val_data.views.items():
                scaler = self.scalers.get(view_name)
                if scaler is not None:
                    X = scaler.transform(X)
                X = np.nan_to_num(X, nan=0.0)
                omics_val[view_name] = X

            val_meta = self._process_obs_readonly(self.val_data.metadata)
            self.val_dataset = MOSADataset(
                omics_data=omics_val,
                masks=self.val_data.masks,
                conditionals=val_meta["conditionals"],
                tissue_labels=val_meta["tissue_labels"],
                source_ids=val_meta["source_ids"],
                sample_weights=val_meta["sample_weights"],
                sample_names=self.val_data.sample_names,
                omic_names=list(self.data_cfg.views),
            )

        n_train = self.train_data.n_samples
        n_val = self.val_data.n_samples if self.val_data is not None else 0
        logger.info("Setup complete: %d train, %d val (%.2fs)", n_train, n_val, time.perf_counter() - t0)

    def _setup_lazy_zarr(self, meta: dict) -> None:
        """Set up LazyZarrDataset for large zarr stores."""
        rng = np.random.RandomState(self.model_cfg.random_seed)
        frac = self.model_cfg.scaler_sample_frac
        store = zarr.open_group(self.zarr_path, mode="r")

        n_train = self.train_data.n_samples
        all_train_idx = np.arange(n_train)

        for view_name in self.data_cfg.views:
            if view_name in self.data_cfg.discrete_views:
                self.scalers[view_name] = None
                continue

            X_zarr = store[f"mod/{view_name}/X"]
            if frac < 1.0:
                n_sub = max(1, int(n_train * frac))
                sub_idx = sorted(rng.choice(all_train_idx, size=n_sub, replace=False))
            else:
                sub_idx = sorted(all_train_idx)

            logger.debug("Fitting scaler for '%s' on %d samples", view_name, len(sub_idx))
            X_sub = np.asarray(X_zarr[sub_idx]).astype(np.float32)
            X_sub = np.nan_to_num(X_sub, nan=0.0)
            scaler = StandardScaler()
            scaler.fit(X_sub)
            self.scalers[view_name] = scaler

        del store

        scaler_dicts: dict[str, dict[str, np.ndarray] | None] = {}
        for name, scaler in self.scalers.items():
            if scaler is not None:
                scaler_dicts[name] = {
                    "mean": scaler.mean_.astype(np.float32),
                    "scale": scaler.scale_.astype(np.float32),
                }
            else:
                scaler_dicts[name] = None

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
                mask_layer_name=self.data_cfg.mask_layer_name,
            )

        n_val = self.val_data.n_samples if self.val_data is not None else 0
        logger.info("zarr lazy setup complete: %d train, %d val", n_train, n_val)

    def _loader_kwargs(self) -> dict:
        nw = self.model_cfg.num_workers
        kwargs: dict = {
            "num_workers": nw,
            "pin_memory": True,
        }
        if nw > 0:
            kwargs["persistent_workers"] = True
            kwargs["prefetch_factor"] = 2
        return kwargs

    def _get_weighted_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler using pre-computed sample weights.
        
        Uses the sample_weights already calculated in _process_obs() to balance
        model_type categories, ensuring each mini-batch has a balanced distribution
        of organoids, cell lines, and tumors.
        """
        if isinstance(dataset, MOSADataset):
            sample_weights = dataset.sample_weights.numpy()
        elif isinstance(dataset, LazyZarrDataset):
            sample_weights = dataset.sample_weights
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")
        
        # Create sampler using pre-computed weights
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

    def state_dict(self) -> dict:
        """Return serialisable preprocessing state for checkpoint saving."""
        scalers = {}
        for name, scaler in self.scalers.items():
            if scaler is None:
                scalers[name] = None
            else:
                scalers[name] = {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                }
        return {
            "scalers": scalers,
            "batch_categories": self.batch_categories,
            "tissue_categories": self.tissue_categories,
            "feature_names": self.feature_names,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore preprocessing state from a saved checkpoint."""
        self.batch_categories = state["batch_categories"]
        self.tissue_categories = state["tissue_categories"]
        self.feature_names = state["feature_names"]
        self.scalers = {}
        for name, s in state["scalers"].items():
            if s is None:
                self.scalers[name] = None
            else:
                scaler = StandardScaler()
                scaler.mean_ = np.array(s["mean"], dtype=np.float64)
                scaler.scale_ = np.array(s["scale"], dtype=np.float64)
                scaler.n_features_in_ = len(s["mean"])
                self.scalers[name] = scaler

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

        if self.data_cfg.use_mutations:
            mutation_cols = [c for c in obs_df.columns if c.startswith("mutation_")]
            mutations = obs_df[mutation_cols].values.astype(np.float32) if mutation_cols else None
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
        sample_weights = self.class_weights[label_codes].astype(np.float32)

        return dict(
            conditionals=conditionals,
            tissue_labels=tissue_labels,
            source_ids=label_codes,
            sample_weights=sample_weights,
        )
