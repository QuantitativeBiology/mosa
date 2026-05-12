from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from mosa.api import MultiOmicModel
from mosa.config import MOSAConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.mosa.datamodule import MOSADataModule
from mosa.models.mosa.vae.vae_module import MOSAVAE

logger = logging.getLogger(__name__)


class MOSAVAEModel(MultiOmicModel):
    """MultiOmicModel implementation using the MOSAVAE architecture.

    Wraps the MOSAVAE Lightning module, MOSADataModule, and training
    orchestration behind the standard MultiOmicModel interface.
    """

    def __init__(self, config: MOSAConfig):
        self.config = config
        self._model: MOSAVAE | None = None
        self._datamodule: MOSADataModule | None = None

    def fit(
        self,
        train: MultiOmicDataset,
        val: MultiOmicDataset | None = None,
    ) -> None:
        """Train the model on the provided data."""
        self._datamodule = MOSADataModule(
            train_data=train,
            val_data=val,
            config=self.config,
        )
        self._datamodule.setup()

        self._model = MOSAVAE(
            config=self.config,
            view_input_dims=self._datamodule.view_input_dims,
            conditional_dim=self._datamodule.conditional_dim,
            n_batches=self._datamodule.n_batches,
        )

        if self._datamodule.class_weights is not None:
            self._model.class_weights = torch.tensor(
                self._datamodule.class_weights, dtype=torch.float32
            )
        if self._datamodule.batch_categories:
            self._model.model_type_names = self._datamodule.batch_categories

        tc = self.config.trainer
        has_val = val is not None
        callbacks = []
        if has_val:
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=tc.early_stopping_patience,
                    mode="min",
                ),
            )
            callbacks.append(
                ModelCheckpoint(
                    dirpath=self.config.output_dir,
                    filename="mosa-{epoch:03d}-{val/loss:.4f}",
                    monitor="val/loss",
                    mode="min",
                    save_top_k=tc.checkpoint_top_k,
                ),
            )

        use_multi_gpu = isinstance(tc.devices, int) and tc.devices > 1
        trainer_kwargs = dict(
            max_epochs=self.config.num_epochs,
            callbacks=callbacks,
            default_root_dir=self.config.output_dir,
            accelerator=tc.accelerator,
            devices=tc.devices,
            precision=tc.precision,
            gradient_clip_val=tc.gradient_clip_val,
            accumulate_grad_batches=tc.accumulate_grad_batches,
            log_every_n_steps=tc.log_every_n_steps,
            sync_batchnorm=use_multi_gpu,
        )
        if use_multi_gpu:
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
        if not has_val:
            trainer_kwargs["limit_val_batches"] = 0
            trainer_kwargs["num_sanity_val_steps"] = 0

        trainer = pl.Trainer(**trainer_kwargs)
        trainer.fit(self._model, self._datamodule)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            self._save_outputs()

    def _save_outputs(self) -> None:
        """Save latent representations and reconstructions for all splits."""
        output_dir = Path(self.config.output_dir)
        dm = self._datamodule

        self._save_split(dm.train_eval_dataloader(), output_dir / "train")

        val_loader = dm.val_dataloader()
        if val_loader is not None:
            self._save_split(val_loader, output_dir / "val")

        self._save_split(dm.full_dataloader(), output_dir / "full")

        if self.config.inference:
            categories = list(dm.batch_categories)
            target = self.config.target_batch.strip()
            if target and target not in categories:
                raise ValueError(
                    f"target_batch '{target}' not in model_type categories: {categories}"
                )
            target_idx = categories.index(target) if target else 0
            self._save_split(
                dm.full_dataloader(),
                output_dir / "inference",
                force_source_id=target_idx,
            )

    def _save_split(
        self,
        loader,
        out_dir: Path,
        force_source_id: int | None = None,
    ) -> None:
        """Run predict on a dataloader and write latent/recon parquet files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        n_batches = len(self._datamodule.batch_categories) if force_source_id is not None else None
        results = self._model.predict(loader, force_source_id=force_source_id, n_batches=n_batches)

        pd.DataFrame(results["z"], index=results["sample_names"]).to_parquet(out_dir / "latent.parquet")

        for omic, recon in results["x_hat"].items():
            scaler = self._datamodule.scalers.get(omic)
            if scaler is not None:
                recon = scaler.inverse_transform(recon)
            cols = self._datamodule.feature_names.get(omic)
            if cols and len(cols) == recon.shape[1]:
                df = pd.DataFrame(recon, index=results["sample_names"], columns=cols)
            else:
                df = pd.DataFrame(recon, index=results["sample_names"])
            df.to_parquet(out_dir / f"recon_{omic}.parquet")

    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        """Project data into the learned latent space."""
        if self._model is None or self._datamodule is None:
            raise RuntimeError("Model must be fit before calling transform()")

        inf_dm = MOSADataModule(
            train_data=data,
            val_data=None,
            config=self.config,
        )
        inf_dm.scalers = self._datamodule.scalers
        inf_dm.setup()

        loader = inf_dm.train_eval_dataloader()
        results = self._model.predict(loader)
        return results["z"]

    def reconstruct(self, data: MultiOmicDataset) -> dict[str, np.ndarray]:
        """Reconstruct omic views from data passed through the model."""
        if self._model is None or self._datamodule is None:
            raise RuntimeError("Model must be fit before calling reconstruct()")

        inf_dm = MOSADataModule(
            train_data=data,
            val_data=None,
            config=self.config,
        )
        inf_dm.scalers = self._datamodule.scalers
        inf_dm.setup()

        loader = inf_dm.train_eval_dataloader()
        results = self._model.predict(loader)
        return results["x_hat"]

    def save(self, path: str | Path) -> None:
        """Save model weights and architecture dims to disk."""
        if self._model is None:
            raise RuntimeError("Model must be fit before saving")
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "view_input_dims": self._model.view_input_dims,
                "conditional_dim": self._model.conditional_dim,
                "n_batches": self._model.n_batches,
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path, config: MOSAConfig) -> MOSAVAEModel:
        """Load a saved model from disk.

        Parameters
        ----------
        path : str or Path
            Path to saved checkpoint (produced by save()).
        config : MOSAConfig
            Config used to reconstruct the model architecture.
        """
        checkpoint = torch.load(str(path))
        instance = cls(config)
        instance._model = MOSAVAE(
            config=config,
            view_input_dims=checkpoint["view_input_dims"],
            conditional_dim=checkpoint["conditional_dim"],
            n_batches=checkpoint["n_batches"],
        )
        instance._model.load_state_dict(checkpoint["state_dict"])
        instance._model.eval()
        return instance
