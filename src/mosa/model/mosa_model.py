from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from mosa.api import MultiOmicModel
from mosa.config import MOSAConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.model.callbacks import SaveLatentAndReconCallback
from mosa.model.datamodule import MuDataDataModule
from mosa.model.mosavae import MOSAVAE

logger = logging.getLogger(__name__)


class MOSAVAEModel(MultiOmicModel):
    """MultiOmicModel implementation using the MOSAVAE architecture.

    Wraps the MOSAVAE Lightning module, MuDataDataModule, and training
    orchestration behind the standard MultiOmicModel interface.
    """

    def __init__(self, config: MOSAConfig):
        self.config = config
        self._model: MOSAVAE | None = None
        self._datamodule: MuDataDataModule | None = None

    def fit(
        self,
        train: MultiOmicDataset,
        val: MultiOmicDataset | None = None,
    ) -> None:
        """Train the model on the provided data."""
        self._datamodule = MuDataDataModule(
            train_data=train,
            val_data=val,
            config=self.config,
        )
        self._datamodule.setup()

        for view_name, input_dim in self._datamodule.view_input_dims.items():
            vc = self.config.views[view_name]
            if vc.input_dim == 0:
                vc.input_dim = input_dim
            if vc.output_dim == 0:
                vc.output_dim = input_dim
        if self.config.conditional_dim == 0:
            self.config.conditional_dim = self._datamodule.conditional_dim
        if self.config.n_batches == 0:
            self.config.n_batches = self._datamodule.n_batches

        self._model = MOSAVAE(self.config)

        if self._datamodule.class_weights is not None:
            self._model.class_weights = torch.tensor(
                self._datamodule.class_weights, dtype=torch.float32
            )
        if self._datamodule.batch_categories:
            self._model.model_type_names = self._datamodule.batch_categories

        tc = self.config.trainer
        has_val = val is not None
        callbacks = [SaveLatentAndReconCallback(output_dir=self.config.output_dir)]
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

    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        """Project data into the learned latent space."""
        if self._model is None or self._datamodule is None:
            raise RuntimeError("Model must be fit before calling transform()")

        inf_dm = MuDataDataModule(
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

        inf_dm = MuDataDataModule(
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
        """Save model state dict to disk."""
        if self._model is None:
            raise RuntimeError("Model must be fit before saving")
        torch.save(self._model.state_dict(), str(path))

    @classmethod
    def load(cls, path: str | Path, config: MOSAConfig) -> MOSAVAEModel:
        """Load a saved model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved state dict.
        config : MOSAConfig
            Config used to reconstruct the model architecture.
        """
        instance = cls(config)
        instance._model = MOSAVAE(config)
        instance._model.load_state_dict(torch.load(str(path)))
        instance._model.eval()
        return instance
