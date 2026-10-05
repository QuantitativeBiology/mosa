from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.errors import UnsupportedError
from mosa.models.api import MultiOmicModel
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig
from mosa.models.mosa.datamodule import MOSADataModule
from mosa.models.mosa.vae.vae_module import VAE
from mosa.models.registry import register_model
from mosa.utils import ensure_dir

logger = logging.getLogger(__name__)

# MOSAConfig fields that were removed but still sit in older checkpoints'
# saved model_cfg; dropped on load so those checkpoints stay loadable.
_REMOVED_MODEL_CFG_KEYS = ("log_every_n_steps",)


class _EpochHistoryCallback(pl.Callback):
    """Record per-epoch train/val loss so callers can plot learning curves.

    Reads from `trainer.callback_metrics` in `on_train_epoch_end`, not
    `on_validation_epoch_end`: the validation loop runs nested inside the
    training epoch, but the epoch's `train/loss` (on_epoch=True) isn't
    reduced into callback_metrics until the training epoch itself finishes,
    which is after validation. Reading at on_validation_epoch_end would pick
    up a stale train/loss left over from the previous epoch.
    """

    def __init__(self):
        self.history: list[dict] = []

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        entry = {"epoch": trainer.current_epoch}
        if "train/loss" in metrics:
            entry["train_loss"] = float(metrics["train/loss"])
        if "val/loss" in metrics:
            entry["val_loss"] = float(metrics["val/loss"])
        self.history.append(entry)


class _LoggingModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that logs a line each time a checkpoint is written."""

    def _save_checkpoint(self, trainer: pl.Trainer, filepath: str) -> None:
        import time

        t0 = time.perf_counter()
        super()._save_checkpoint(trainer, filepath)
        logger.info(
            "Saved checkpoint %s (epoch %d, %.1fs)",
            Path(filepath).name,
            trainer.current_epoch,
            time.perf_counter() - t0,
        )


@register_model("mosa_vae", MOSAConfig)
class MOSAModel(MultiOmicModel):
    """MultiOmicModel implementation using the VAE architecture.

    Wraps the VAE Lightning module, MOSADataModule, and training
    orchestration behind the standard MultiOmicModel interface.
    """

    checkpoint_suffixes = (".ckpt", ".pt")

    def __init__(self, data_cfg: DataConfig, model_cfg: MOSAConfig):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self._model: VAE | None = None
        self._datamodule: MOSADataModule | None = None
        self._trainer: pl.Trainer | None = None
        self.epoch_history: list[dict] = []

    def fit(
        self,
        train: MultiOmicDataset,
        val: MultiOmicDataset | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        """Train the model on the provided data."""
        self._datamodule = MOSADataModule(
            train_data=train,
            val_data=val,
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
        )
        self._datamodule.setup()

        self._model = VAE(
            config=self.model_cfg,
            view_input_dims=self._datamodule.view_input_dims,
            conditional_dim=self._datamodule.conditional_dim,
            n_batches=self._datamodule.n_batches,
            data_cfg=self.data_cfg,
        )
        # Stamped before training so ModelCheckpoint's files carry it too.
        self._model.hparams["model_type_name"] = self.registered_name

        if self._datamodule.class_weights is not None:
            self._model.class_weights = torch.tensor(
                self._datamodule.class_weights, dtype=torch.float32
            )
        if self._datamodule.batch_categories:
            self._model.model_type_names = self._datamodule.batch_categories

        mc = self.model_cfg
        has_val = val is not None
        callbacks = []
        history_cb = _EpochHistoryCallback() if has_val else None
        if has_val:
            callbacks.append(history_cb)
            callbacks.append(
                EarlyStopping(
                    monitor="val/loss",
                    patience=mc.early_stopping_patience,
                    mode="min",
                ),
            )
            if mc.checkpoint_top_k != 0:
                callbacks.append(
                    _LoggingModelCheckpoint(
                        # Kept apart from the run root, where lightning_logs/
                        # already exists, so Lightning's "directory not empty"
                        # warning only fires when a run reuses an old folder.
                        dirpath=Path(mc.output_dir) / "checkpoints",
                        # auto_insert_metric_name would write "val/loss=",
                        # and the slash turns into a subdirectory.
                        filename="mosa-epoch={epoch:03d}-val_loss={val/loss:.4f}",
                        auto_insert_metric_name=False,
                        monitor="val/loss",
                        mode="min",
                        save_top_k=mc.checkpoint_top_k,
                        save_last=True,
                    ),
                )

        use_multi_gpu = isinstance(mc.devices, int) and mc.devices > 1
        trainer_kwargs = dict(
            max_epochs=mc.num_epochs,
            callbacks=callbacks,
            default_root_dir=mc.output_dir,
            accelerator=mc.accelerator,
            devices=mc.devices,
            precision=mc.precision,
            # gradient_clip_val is applied manually in VAE.training_step;
            # Lightning forbids it on the Trainer under manual optimization.
            accumulate_grad_batches=mc.accumulate_grad_batches,
            # Every metric is epoch-level, so the step interval only drives
            # Lightning's "fewer batches than the logging interval" warning.
            log_every_n_steps=1,
            sync_batchnorm=use_multi_gpu,
        )
        if use_multi_gpu:
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=True)
        if not has_val:
            trainer_kwargs["limit_val_batches"] = 0
            trainer_kwargs["num_sanity_val_steps"] = 0

        self._trainer = pl.Trainer(**trainer_kwargs)
        self._trainer.fit(
            self._model,
            self._datamodule,
            ckpt_path=str(resume_from) if resume_from else None,
        )
        self.epoch_history = history_cb.history if history_cb is not None else []

    def save_outputs(self, output_dir: str | Path | None = None) -> None:
        """Save latent representations and reconstructions for all splits."""
        import time

        if self._model is None or self._datamodule is None:
            raise RuntimeError("Model must be fit before calling save_outputs()")

        output_dir = (
            Path(output_dir)
            if output_dir is not None
            else Path(self.model_cfg.output_dir)
        )
        dm = self._datamodule

        logger.info("Saving outputs to %s", output_dir)
        t0 = time.perf_counter()

        logger.info("Predicting on train split")
        self._save_split(dm.train_eval_dataloader(), output_dir / "train")

        val_loader = dm.val_dataloader()
        if val_loader is not None:
            logger.info("Predicting on val split")
            self._save_split(val_loader, output_dir / "val")

        logger.info("Predicting on full split")
        self._save_split(dm.full_dataloader(), output_dir / "full")

        if self.model_cfg.inference:
            categories = list(dm.batch_categories)
            target = self.model_cfg.target_batch.strip()
            if target and target not in categories:
                raise ValueError(
                    f"target_batch '{target}' not in model_type categories: {categories}"
                )
            target_idx = categories.index(target) if target else 0
            logger.info(
                "Predicting inference split with target_batch='%s'",
                target or categories[0],
            )
            self._save_split(
                dm.full_dataloader(),
                output_dir / "inference",
                force_source_id=target_idx,
            )

        logger.info("Outputs saved in %.1fs", time.perf_counter() - t0)

    def _save_split(
        self,
        loader,
        out_dir: Path,
        force_source_id: int | None = None,
    ) -> None:
        """Run predict on a dataloader and write latent/recon parquet files."""
        import time

        assert self._datamodule is not None
        assert self._model is not None

        ensure_dir(out_dir)
        n_batches = (
            len(self._datamodule.batch_categories)
            if force_source_id is not None
            else None
        )

        t = time.perf_counter()
        results = self._model.predict(
            loader, force_source_id=force_source_id, n_batches=n_batches
        )
        logger.debug(
            "  predict %d samples in %.1fs",
            len(results["sample_names"]),
            time.perf_counter() - t,
        )

        source_ids = results.get("source_ids")
        if force_source_id is not None:
            source_ids = np.full(
                len(results["sample_names"]), force_source_id, dtype=np.int64
            )

        t = time.perf_counter()
        pd.DataFrame(results["z"], index=results["sample_names"]).to_parquet(
            out_dir / "latent.parquet"
        )
        logger.debug("  wrote latent.parquet (%.1fs)", time.perf_counter() - t)

        for omic, recon in results["x_hat"].items():
            t = time.perf_counter()
            recon = self._datamodule.inverse_transform_view(
                omic, recon, source_ids=source_ids
            )
            cols = self._datamodule.feature_names.get(omic)
            if cols and len(cols) == recon.shape[1]:
                df = pd.DataFrame(recon, index=results["sample_names"], columns=cols)
            else:
                df = pd.DataFrame(recon, index=results["sample_names"])
            df.to_parquet(out_dir / f"recon_{omic}.parquet")
            logger.debug(
                "  wrote recon_%s.parquet shape=%s (%.1fs)",
                omic,
                recon.shape,
                time.perf_counter() - t,
            )

    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        """Project data into the learned latent space."""
        if self._model is None or self._datamodule is None:
            raise RuntimeError("Model must be fit before calling transform()")

        inf_dm = MOSADataModule(
            train_data=data,
            val_data=None,
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
        )
        inf_dm.setup_inference(self._datamodule)

        loader = inf_dm.train_eval_dataloader()
        results = self._model.predict(loader)
        return results["z"]

    def reconstruct(self, data: MultiOmicDataset) -> dict[str, np.ndarray]:
        """Reconstruct omic views from data passed through the model, in original scale."""
        if self._model is None or self._datamodule is None:
            raise RuntimeError("Model must be fit before calling reconstruct()")

        inf_dm = MOSADataModule(
            train_data=data,
            val_data=None,
            data_cfg=self.data_cfg,
            model_cfg=self.model_cfg,
        )
        inf_dm.setup_inference(self._datamodule)

        loader = inf_dm.train_eval_dataloader()
        results = self._model.predict(loader)

        source_ids = results.get("source_ids")
        return {
            omic: inf_dm.inverse_transform_view(omic, recon, source_ids=source_ids)
            for omic, recon in results["x_hat"].items()
        }

    def save(self, path: str | Path) -> None:
        """Save model state to a Lightning checkpoint.

        Delegates to trainer.save_checkpoint so the file is identical in
        shape to the .ckpt files Lightning's ModelCheckpoint writes during
        training. Both can be loaded with load() and used for inference.
        """
        if self._trainer is None:
            raise RuntimeError("Model must be fit before saving")
        self.require_checkpoint_suffix(path)

        # Embed the registered model-type name in the module's hyperparameters
        # before writing, so owns_checkpoint can tell this file apart from
        # another torch-based model's, and the checkpoint is serialized in a
        # single pass.
        assert self._model is not None
        self._model.hparams["model_type_name"] = self.registered_name
        self._trainer.save_checkpoint(str(path))

    @classmethod
    def owns_checkpoint(cls, path: str | Path) -> bool:
        """A .ckpt/.pt file naming this model, or an unnamed one written by MOSA's VAE.

        Checkpoints Lightning's ModelCheckpoint wrote before fit() stamped the
        name carry none; view_input_dims, in every VAE's hyperparameters since
        May 2026, marks those as MOSA's rather than another Lightning model's.
        """
        if not super().owns_checkpoint(path):
            return False
        # mmap reads only the pickled metadata here, not every tensor; load()
        # reads the file in full afterwards. Legacy (non-zip) torch files
        # cannot be memory-mapped and are read in full.
        try:
            checkpoint = torch.load(
                str(path), map_location="cpu", weights_only=False, mmap=True
            )
        except RuntimeError:
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        hp = checkpoint.get("hyper_parameters", {})
        name = hp.get("model_type_name")
        if name is None:
            return "view_input_dims" in hp
        return name == cls.registered_name

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> MOSAModel:
        """Load a model from a Lightning checkpoint.

        Accepts any Lightning .ckpt file: those written by save() and those
        written automatically by ModelCheckpoint during training are
        interchangeable. Config, arch dims, and preprocessing state are all
        restored from the file.
        """
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        hp = checkpoint["hyper_parameters"]
        # Checkpoints from before the July 2026 config split store a single
        # "config" dict that no longer maps onto DataConfig/MOSAConfig.
        if "data_cfg" not in hp:
            raise UnsupportedError(
                f"Checkpoint {path} was written by an older MOSA version, "
                "before the config was split into data and model sections, "
                "and cannot be loaded. Retrain the model with the current version."
            )

        data_cfg = DataConfig(**hp["data_cfg"])

        mcfg_raw = dict(hp["model_cfg"])
        for key in _REMOVED_MODEL_CFG_KEYS:
            mcfg_raw.pop(key, None)
        mcfg_raw["views"] = {
            n: OmicViewConfig(**v) for n, v in mcfg_raw["views"].items()
        }
        model_cfg = MOSAConfig(**mcfg_raw)

        instance = cls(data_cfg, model_cfg)
        instance._model = VAE(
            config=model_cfg,
            view_input_dims=hp["view_input_dims"],
            conditional_dim=hp["conditional_dim"],
            n_batches=hp["n_batches"],
        )
        instance._model.load_state_dict(checkpoint["state_dict"])
        instance._model.eval()

        dm_key = MOSADataModule.__name__
        if dm_key in checkpoint:
            dm = MOSADataModule(
                train_data=None,
                val_data=None,
                data_cfg=data_cfg,
                model_cfg=model_cfg,
            )
            dm.load_state_dict(checkpoint[dm_key])
            instance._datamodule = dm

        return instance
