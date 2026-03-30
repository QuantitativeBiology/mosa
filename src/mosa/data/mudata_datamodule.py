from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse

from mosa.config import MOSAConfig
from mosa.data.batch import MOSABatch, collate_fn
from mosa.data.datamodule import MOSADataset

logger = logging.getLogger(__name__)


class MuDataDataModule(pl.LightningDataModule):
    """Data module for loading MuData (*.h5mu/*.zarr) format.
    
    Replaces MOSADataModule for MuData integration.
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
        """Load MuData, extract modalities, build conditionals, split data."""
        import mudata
        
        # 1. Load MuData
        logger.info("Loading MuData from %s", self.config.data_path)
        mdata = mudata.read(self.config.data_path)
        
        # 2. Verify structure
        self._verify_mudata_structure(mdata)
        
        # 3. Extract data & handle missing modalities
        logger.debug("Extracting data from MuData modalities")
        omics_all: dict[str, np.ndarray] = {}
        masks_all: dict[str, np.ndarray] = {}
        
        for view_name in self.config.views:
            if view_name not in mdata.mod:
                raise ValueError(
                    f"View '{view_name}' configured but not found in MuData modalities. "
                    f"Available modalities: {list(mdata.mod.keys())}"
                )
            
            adata = mdata.mod[view_name]
            
            # Get data (convert sparse to dense if needed)
            X = adata.X
            if issparse(X):
                X = X.toarray()
            X = X.astype(np.float32)
            
            # Get mask from layer
            if self.config.mask_layer_name not in adata.layers:
                raise ValueError(
                    f"Mask layer '{self.config.mask_layer_name}' not found in modality '{view_name}'. "
                    f"Available layers: {list(adata.layers.keys())}"
                )
            
            mask = adata.layers[self.config.mask_layer_name]
            if issparse(mask):
                mask = mask.toarray()
            mask = mask.astype(bool)
            
            # Handle samples missing this modality (not in .obsm[view_name])
            if view_name in mdata.obsm:
                presence = np.asarray(mdata.obsm[view_name]).flatten().astype(bool)
                if len(presence) != X.shape[0]:
                    raise ValueError(
                        f"Presence vector for '{view_name}' has length {len(presence)} "
                        f"but data has {X.shape[0]} samples"
                    )
                missing_idx = ~presence
                X[missing_idx] = 0.0
                mask[missing_idx] = False
            
            omics_all[view_name] = X
            masks_all[view_name] = mask
            
            # Store feature names
            self.feature_names[view_name] = list(adata.var_names)
        
        # 4. Build conditionals from .obs
        logger.debug("Building conditionals from .obs")
        
        # Batch (model_type) – always included
        if "model_type" not in mdata.obs.columns:
            raise ValueError("MuData .obs must contain 'model_type' column")
        batch_dummies = pd.get_dummies(mdata.obs["model_type"])
        self.batch_categories = list(batch_dummies.columns)
        
        # Tissue – always included if column exists
        if "tissue" in mdata.obs.columns:
            tissue_dummies = pd.get_dummies(mdata.obs["tissue"])
            self.tissue_categories = list(tissue_dummies.columns)
        else:
            tissue_dummies = pd.DataFrame()
            logger.warning("No 'tissue' column found in MuData .obs")
        
        # Mutations – columns prefixed "mutation_"
        mutation_cols = [c for c in mdata.obs.columns 
                        if c.startswith("mutation_")]
        if mutation_cols:
            mutations = mdata.obs[mutation_cols].values.astype(np.float32)
        else:
            mutations = None
            logger.debug("No mutation_* columns found in MuData .obs")
        
        # Concatenate conditionals
        cond_parts = [batch_dummies.values]
        if not tissue_dummies.empty:
            cond_parts.append(tissue_dummies.values)
        if mutations is not None:
            cond_parts.append(mutations)
        
        conditionals = np.concatenate(cond_parts, axis=1).astype(np.float32)
        
        # 5. Extract other metadata
        if "tissue" in mdata.obs.columns:
            tissue_labels = pd.get_dummies(mdata.obs["tissue"]).values.astype(np.float32)
        else:
            tissue_labels = np.zeros((len(mdata.obs), 1), dtype=np.float32)
        source_ids = pd.Categorical(mdata.obs["model_type"]).codes.astype(np.int64)
        sample_names = list(mdata.obs.index)
        
        # 6. Train/val split (stratified by model_type)
        logger.debug("Train/val split (test_size=%.2f)", self.config.test_size)
        model_type_cats = pd.Categorical(
            mdata.obs["model_type"],
            categories=sorted(mdata.obs["model_type"].unique()),
            ordered=True,
        )
        label_codes = np.asarray(model_type_cats.codes, dtype=np.intp)
        
        if self.config.test_size > 0:
            train_idx, val_idx = train_test_split(
                np.arange(len(sample_names)),
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=label_codes,
            )
        else:
            train_idx = np.arange(len(sample_names))
            val_idx = np.array([], dtype=int)
        logger.debug("  train=%d, val=%d", len(train_idx), len(val_idx))
        
        # 7. Fit scalers on train only
        logger.debug("Fitting scalers on training data")
        for view_name, X in omics_all.items():
            if self.config.views[view_name].discrete:
                X = np.nan_to_num(X, nan=0.0)
                self.scalers[view_name] = None
            else:
                scaler = StandardScaler()
                scaler.fit(X[train_idx])
                X = scaler.transform(X)
                X = np.nan_to_num(X, nan=0.0)  # zero-fill constant features
                self.scalers[view_name] = scaler
            omics_all[view_name] = X
        
        # 8. Compute class weights for macro loss
        logger.debug("Computing class weights")
        unique, counts = np.unique(label_codes[train_idx], return_counts=True)
        self.class_weights = np.zeros(len(unique))
        for i, (cls, count) in enumerate(zip(unique, counts)):
            self.class_weights[cls] = len(train_idx) / (len(unique) * count)
        
        # 9. Update config dimensions
        self._update_config_dims(conditionals)
        
        # 10. Create datasets
        logger.debug("Creating datasets")
        self.train_dataset = self._create_dataset(
            omics_all, masks_all, conditionals, tissue_labels, 
            source_ids, sample_names, train_idx
        )
        
        if len(val_idx) > 0:
            self.val_dataset = self._create_dataset(
                omics_all, masks_all, conditionals, tissue_labels,
                source_ids, sample_names, val_idx
            )
        else:
            self.val_dataset = None
        
        logger.info("Setup complete: %d training, %d validation samples", 
                   len(train_idx), len(val_idx))
    
    def _verify_mudata_structure(self, mdata) -> None:
        """Verify MuData has required structure."""
        # Check required columns in .obs
        required_cols = ["model_type"]
        missing_cols = [col for col in required_cols if col not in mdata.obs.columns]
        if missing_cols:
            raise ValueError(
                f"MuData .obs missing required columns: {missing_cols}. "
                f"Available columns: {list(mdata.obs.columns)}"
            )
        
        # Check all configured views exist
        configured_views = set(self.config.views.keys())
        available_views = set(mdata.mod.keys())
        missing_views = configured_views - available_views
        if missing_views:
            raise ValueError(
                f"Configured views not found in MuData: {missing_views}. "
                f"Available modalities: {list(available_views)}"
            )
        
        # Check mask layer exists in each modality
        for view_name in self.config.views:
            adata = mdata.mod[view_name]
            if self.config.mask_layer_name not in adata.layers:
                raise ValueError(
                    f"Mask layer '{self.config.mask_layer_name}' not found in modality '{view_name}'. "
                    f"Available layers: {list(adata.layers.keys())}"
                )
    
    def _update_config_dims(self, conditionals: np.ndarray) -> None:
        """Update config dimensions after loading data."""
        # Update input/output dims for each view
        for view_name in self.config.views:
            view_config = self.config.views[view_name]
            if view_config.input_dim == 0:
                view_config.input_dim = len(self.feature_names[view_name])
            if view_config.output_dim == 0:
                view_config.output_dim = len(self.feature_names[view_name])
        
        # Update conditional dim
        if self.config.conditional_dim == 0:
            self.config.conditional_dim = conditionals.shape[1]
        
        # Update n_batches
        if self.config.n_batches == 0:
            self.config.n_batches = len(self.batch_categories)
    
    def _create_dataset(
        self,
        omics_all: dict[str, np.ndarray],
        masks_all: dict[str, np.ndarray],
        conditionals: np.ndarray,
        tissue_labels: np.ndarray,
        source_ids: np.ndarray,
        sample_names: list[str],
        indices: np.ndarray,
    ) -> MOSADataset:
        """Create a MOSADataset for a subset of samples."""
        subset_omics = {k: v[indices] for k, v in omics_all.items()}
        subset_masks = {k: v[indices] for k, v in masks_all.items()}
        subset_conditionals = conditionals[indices]
        subset_tissue_labels = tissue_labels[indices]
        subset_source_ids = source_ids[indices]
        subset_sample_names = [sample_names[i] for i in indices]
        
        # Compute sample weights (for macro loss)
        if self.class_weights is not None:
            sample_weights = self.class_weights[subset_source_ids]
        else:
            sample_weights = np.ones(len(indices), dtype=np.float32)
        
        return MOSADataset(
            omics_data=subset_omics,
            masks=subset_masks,
            conditionals=subset_conditionals,
            tissue_labels=subset_tissue_labels,
            source_ids=subset_source_ids,
            sample_weights=sample_weights,
            sample_names=subset_sample_names,
            omic_names=list(self.config.views.keys()),
        )
    
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
    
    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting dataloaders")
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
    
    def test_dataloader(self) -> DataLoader:
        # Not implemented - use val_dataloader for testing
        raise NotImplementedError("Test dataloader not implemented")
    
    def predict_dataloader(self) -> DataLoader:
        # Not implemented
        raise NotImplementedError("Predict dataloader not implemented")