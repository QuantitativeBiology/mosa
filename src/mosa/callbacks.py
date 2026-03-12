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

    - ``data/latent.csv`` — joint latent z for the training split
    - ``data/recon_{omic}.csv`` — reconstructed omic for the training split
    - ``inference/latent.csv`` — joint latent z for the validation split
    - ``inference/recon_{omic}.csv`` — reconstructed omic for the validation split
    """

    def __init__(self, output_dir: str | Path):
        super().__init__()
        self.output_dir = Path(output_dir)

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Run inference on train and val splits and save results to CSV."""
        logger.debug("Saving latent representations and reconstructions to %s", self.output_dir)
        self._save_split(pl_module, trainer.datamodule.train_dataloader(), self.output_dir / "data")
        val_loader = trainer.datamodule.val_dataloader()
        if val_loader is not None:
            self._save_split(pl_module, val_loader, self.output_dir / "inference")

    def _save_split(self, model, loader, out_dir: Path) -> None:
        """Run model.predict() on a dataloader and write CSV files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        results = model.predict(loader)

        pd.DataFrame(results["z"], index=results["sample_names"]).to_csv(out_dir / "latent.csv")
        logger.debug("Saved latent %s to %s", results["z"].shape, out_dir / "latent.csv")

        for omic, recon in results["x_hat"].items():
            path = out_dir / f"recon_{omic}.csv"
            pd.DataFrame(recon, index=results["sample_names"]).to_csv(path)
            logger.debug("Saved recon '%s' %s to %s", omic, recon.shape, path)
