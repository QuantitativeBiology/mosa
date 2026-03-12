from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mosa.config import MOSAConfig
from mosa.data.batch import MOSABatch, collate_fn

logger = logging.getLogger(__name__)


class MOSADataset(Dataset):
    """Stores per-sample tensors for a single data split (train or val)."""

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
        """Return a single sample as a dict (collated into MOSABatch by collate_fn)."""
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


class MOSADataModule(pl.LightningDataModule):
    """Lightning DataModule for loading and preprocessing multi-omic data.

    Orchestrates: CSV loading, sample alignment, train/val splitting,
    z-score normalization (fit on train only), conditional vector
    construction, and class weight computation.

    Input CSVs are expected in features x samples format and are transposed
    during loading. Data is assumed to be already feature-engineered.
    """

    def __init__(self, config: MOSAConfig):
        super().__init__()
        self.config = config
        self.scalers: dict[str, StandardScaler | None] = {}
        self.feature_names: dict[str, list[str]] = {}
        self.batch_categories: list[str] = []
        self.tissue_categories: list[str] = []
        self.train_dataset: MOSADataset | None = None
        self.val_dataset: MOSADataset | None = None
        self.class_weights: np.ndarray | None = None

    def setup(self, stage: str | None = None) -> None:
        """Load data, split, preprocess, and create train/val datasets."""

        # 1. Load CSVs (features x samples) and transpose to samples x features
        logger.debug("Loading view CSVs")
        omics: dict[str, pd.DataFrame] = {}
        for name, vc in self.config.views.items():
            df = pd.read_csv(vc.path, index_col=0).T.astype(float)
            omics[name] = df
            logger.debug("  view '%s': %d samples x %d features", name, *df.shape)

        # 2. Load samplesheet
        logger.debug("Loading samplesheet from %s", self.config.samplesheet_path)
        samplesheet = pd.read_csv(self.config.samplesheet_path).set_index("model_id")

        # 3. Find common samples across all views and samplesheet
        common = set(samplesheet.index)
        for df in omics.values():
            common &= set(df.index)

        # Include mutations in sample alignment if used
        mutations_df = None
        if self.config.use_mutations_conditional and self.config.mutations_path:
            mutations_df = pd.read_csv(self.config.mutations_path, index_col=0).T
            common &= set(mutations_df.index)

        common_samples = sorted(common)
        logger.debug("Common samples: %d", len(common_samples))

        # 4. Align all data to common samples
        samplesheet = samplesheet.loc[common_samples]
        for name in omics:
            omics[name] = omics[name].loc[common_samples]
        if mutations_df is not None:
            mutations_df = mutations_df.loc[common_samples]

        # Store feature names for each view
        for name, df in omics.items():
            self.feature_names[name] = list(df.columns)

        # 5. Train/val split (stratified by model_type)
        logger.debug("Train/val split (test_size=%.2f)", self.config.test_size)
        model_type_cats = pd.Categorical(
            samplesheet["model_type"],
            categories=sorted(samplesheet["model_type"].unique()),
            ordered=True,
        )
        label_codes = np.asarray(model_type_cats.codes, dtype=np.intp)

        if self.config.test_size > 0:
            train_idx, val_idx = train_test_split(
                np.arange(len(common_samples)),
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=label_codes,
            )
        else:
            train_idx = np.arange(len(common_samples))
            val_idx = np.array([], dtype=int)
        logger.debug("  train=%d, val=%d", len(train_idx), len(val_idx))

        # 6. Build masks, fit scalers (train only), transform
        logger.debug("Building masks and fitting scalers")
        omics_all: dict[str, np.ndarray] = {}
        masks_all: dict[str, np.ndarray] = {}

        for name, df in omics.items():
            X = df.values.astype(np.float32)
            masks_all[name] = ~np.isnan(X)

            if self.config.views[name].discrete:
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[name] = None
            else:
                scaler = StandardScaler()
                scaler.fit(X[train_idx])
                X = scaler.transform(X)
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[name] = scaler

            omics_all[name] = X

        # 7. Build conditionals
        logger.debug("Building conditionals")

        # Model type (batch) — always included
        self.batch_categories = sorted(samplesheet["model_type"].unique())
        batch_dummies = pd.get_dummies(samplesheet["model_type"])
        batch_labels = batch_dummies[self.batch_categories].values.astype(np.float32)

        # Tissue — optional
        self.tissue_categories = sorted(samplesheet["tissue"].unique())
        tissue_dummies = pd.get_dummies(samplesheet["tissue"])
        tissue_labels = tissue_dummies[self.tissue_categories].values.astype(np.float32)

        # Mutations — optional
        mutations_all = None
        if mutations_df is not None:
            mutations_all = mutations_df.values.astype(np.float32)

        # Concatenate conditional vector: [batch, tissue?, mutations?]
        cond_parts = [batch_labels]
        if self.config.use_tissue_conditional:
            cond_parts.append(tissue_labels)
        if mutations_all is not None:
            cond_parts.append(mutations_all)
        conditionals = np.concatenate(cond_parts, axis=1).astype(np.float32)
        logger.debug("  conditionals shape: %s", conditionals.shape)

        # Source IDs (integer model_type index for discriminator/loss)
        source_ids = label_codes

        # 8. Class weights (inverse-frequency balancing)
        unique_classes = np.unique(label_codes)
        n_samples = len(label_codes)
        n_classes = len(unique_classes)
        class_weights = np.zeros(n_classes, dtype=np.float64)
        for i, cls in enumerate(unique_classes):
            class_weights[i] = n_samples / (n_classes * np.sum(label_codes == cls))
        sample_weights = class_weights[label_codes].astype(np.float32)
        self.class_weights = class_weights.astype(np.float32)

        # 9. Update config dims from loaded data
        for name, df in omics.items():
            self.config.views[name].input_dim = df.shape[1]
            if self.config.views[name].output_dim == 0:
                self.config.views[name].output_dim = df.shape[1]
        self.config.conditional_dim = conditionals.shape[1]
        self.config.n_batches = len(self.batch_categories)

        # 10. Create datasets
        omic_names = list(self.config.views.keys())

        self.train_dataset = MOSADataset(
            omics_data={k: v[train_idx] for k, v in omics_all.items()},
            masks={k: v[train_idx] for k, v in masks_all.items()},
            conditionals=conditionals[train_idx],
            tissue_labels=tissue_labels[train_idx],
            source_ids=source_ids[train_idx],
            sample_weights=sample_weights[train_idx],
            sample_names=[common_samples[i] for i in train_idx],
            omic_names=omic_names,
        )

        if len(val_idx) > 0:
            self.val_dataset = MOSADataset(
                omics_data={k: v[val_idx] for k, v in omics_all.items()},
                masks={k: v[val_idx] for k, v in masks_all.items()},
                conditionals=conditionals[val_idx],
                tissue_labels=tissue_labels[val_idx],
                source_ids=source_ids[val_idx],
                sample_weights=sample_weights[val_idx],
                sample_names=[common_samples[i] for i in val_idx],
                omic_names=omic_names,
            )

    def train_dataloader(self) -> DataLoader:
        """Return a DataLoader for the training split."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.trainer.num_workers,
        )

    def val_dataloader(self) -> DataLoader | None:
        """Return a DataLoader for the validation split, or None if no validation set."""
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.trainer.num_workers,
        )
