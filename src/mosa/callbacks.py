from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class SaveLatentAndReconCallback(pl.Callback):
    """Save latent representations and reconstructions to CSV at the end of training.

    Creates the following files inside ``output_dir``:

    - ``train/latent.csv`` — joint latent z for the training split
    - ``train/recon_{omic}.csv`` — reconstructed omic for the training split
    - ``val/latent.csv`` — joint latent z for the validation split
    - ``val/recon_{omic}.csv`` — reconstructed omic for the validation split
    - ``full/latent.csv`` — joint latent z for all samples with original conditionals
    - ``full/recon_{omic}.csv`` — reconstructed omic for all samples with original conditionals
    - ``inference/latent.csv`` — corrected latent z for all samples with forced target batch
    - ``inference/recon_{omic}.csv`` — corrected reconstructions with forced target batch
    """

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save train/val/full outputs and optional corrected inference outputs."""
        if not trainer.is_global_zero:
            return
        logger.debug("Saving latent representations and reconstructions to %s", self.output_dir)
        datamodule = trainer.datamodule

        self._save_split(
            pl_module,
            datamodule.train_eval_dataloader(),
            self.output_dir / "train",
            datamodule,
        )

        val_loader = datamodule.val_dataloader()
        if val_loader is not None:
            self._save_split(pl_module, val_loader, self.output_dir / "val", datamodule)

        full_loader = datamodule.full_dataloader()
        self._save_split(pl_module, full_loader, self.output_dir / "full", datamodule)

        if datamodule.config.inference:
            target_idx, target_name = self._resolve_target_batch(datamodule)
            self._save_split(
                pl_module,
                full_loader,
                self.output_dir / "inference",
                datamodule,
                force_source_id=target_idx,
            )
            logger.debug(
                "Saved corrected inference outputs with target model_type='%s' (index=%d)",
                target_name,
                target_idx,
            )

    def _resolve_target_batch(self, datamodule) -> tuple[int, str]:
        """Resolve target model_type for corrected inference."""
        categories = list(datamodule.batch_categories)
        if not categories:
            raise RuntimeError("batch_categories are not available for corrected inference")

        target = datamodule.config.target_batch.strip()
        if target:
            if target not in categories:
                raise ValueError(
                    f"target_batch '{target}' not found in available model_type categories: {categories}"
                )
            return categories.index(target), target

        return 0, categories[0]

    def _inverse_transform(self, omic: str, recon, datamodule) -> tuple:
        """Map reconstructions back to original omic scale when scaler is available."""
        scaler = getattr(datamodule, "scalers", {}).get(omic)
        if scaler is None:
            return recon, False
        return scaler.inverse_transform(recon), True

    def _save_split(
        self,
        model,
        loader,
        out_dir: Path,
        datamodule,
        force_source_id: int | None = None,
    ) -> None:
        """Run model.predict() on a dataloader and write CSV files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        n_batches = len(datamodule.batch_categories) if force_source_id is not None else None
        results = model.predict(loader, force_source_id=force_source_id, n_batches=n_batches)

        pd.DataFrame(results["z"], index=results["sample_names"]).to_csv(out_dir / "latent.csv")
        logger.debug("Saved latent %s to %s", results["z"].shape, out_dir / "latent.csv")

        feature_names = getattr(datamodule, "feature_names", {})
        for omic, recon in results["x_hat"].items():
            recon_out, was_inversed = self._inverse_transform(omic, recon, datamodule)
            cols = feature_names.get(omic)
            if cols is not None and len(cols) == recon_out.shape[1]:
                df = pd.DataFrame(recon_out, index=results["sample_names"], columns=cols)
            else:
                df = pd.DataFrame(recon_out, index=results["sample_names"])

            path = out_dir / f"recon_{omic}.csv"
            df.to_csv(path)
            logger.debug(
                "Saved recon '%s' %s to %s (inverse_transform=%s)",
                omic,
                recon_out.shape,
                path,
                was_inversed,
            )
