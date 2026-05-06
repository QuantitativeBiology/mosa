from __future__ import annotations

import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl

from mosa.config import MOSAConfig
from mosa.data.batch import MOSABatch
from mosa.model.losses import (
    adversarial_loss,
    contrastive_loss,
    kl_divergence,
    reconstruction_loss,
)
from mosa.model.decoder import OmicDecoder
from mosa.model.discriminator import Discriminator
from mosa.model.encoder import OmicEncoder
from mosa.model.latent import BaseLatentSpace

logger = logging.getLogger(__name__)


def _kl_weight_for_epoch(epoch: int, config: MOSAConfig) -> float:
    """Compute KL weight for the given epoch via linear warmup schedule."""
    if not config.use_kl_scheduler:
        return config.kl_weight

    start = config.kl_weight
    final = config.kl_weight_final
    warmup = max(0, config.kl_warmup_epochs)

    if warmup == 0 or final == start:
        return final

    progress = min(1.0, float(epoch + 1) / float(warmup))
    return start + (final - start) * progress


class MOSAVAE(pl.LightningModule):
    """Variational autoencoder for multi-omic data with optional adversarial training.

    Uses manual optimization: discriminator is updated first on detached z,
    then VAE parameters are updated with reconstruction, KL, and adversarial losses.
    """

    def __init__(self, config: MOSAConfig):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.save_hyperparameters({"config": vars(config)})

        self.view_latent_dims: dict[str, int] = {
            name: vc.hidden_layer_dims[-1] for name, vc in config.views.items()
        }
        self.view_order = list(config.views.keys())

        # Per-view encoders and decoders
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for name, vc in config.views.items():
            self.encoders[name] = OmicEncoder(
                input_dim=vc.input_dim,
                cond_dim=config.conditional_dim,
                hidden_dims=vc.hidden_layer_dims,
                latent_dim=self.view_latent_dims[name],
                dropout_p=vc.dropout_p,
                view_dropout_p=config.view_dropout_prob,
            )
            output_dim = vc.output_dim if vc.output_dim > 0 else vc.input_dim
            self.decoders[name] = OmicDecoder(
                output_dim=output_dim,
                cond_dim=config.conditional_dim,
                hidden_dims=vc.hidden_layer_dims,
                latent_dim=config.joint_latent_dim,
                dropout_p=vc.dropout_p,
            )

        # Joint latent space (fusion method selected via config)
        self.latent_space = BaseLatentSpace.create(
            config.fusion_method, self.view_latent_dims, config.joint_latent_dim,
            config.shared_hidden_layer_dims,
        )

        logger.debug("Encoders: %s",
                      {n: f"{vc.input_dim}->{self.view_latent_dims[n]}" for n, vc in config.views.items()})
        logger.debug("Decoders: %s",
                      {n: f"{config.joint_latent_dim}->{vc.output_dim or vc.input_dim}" for n, vc in config.views.items()})
        logger.debug("Latent space: %s -> joint_latent_dim=%d",
                      config.fusion_method, config.joint_latent_dim)

        # Optional adversarial discriminator
        self.discriminator: Discriminator | None = None
        if config.adv_weight > 0 and config.n_batches > 0:
            self.discriminator = Discriminator(config.joint_latent_dim, config.n_batches)
            logger.debug("Discriminator: latent_dim=%d, n_batches=%d",
                         config.joint_latent_dim, config.n_batches)

        # Class weights for adversarial loss (set by datamodule after setup)
        self.register_buffer(
            "class_weights", torch.ones(max(config.n_batches, 1)), persistent=False
        )

        # Model type category names (set by datamodule for per-group logging)
        self.model_type_names: list[str] | None = None

    def forward(self, batch: MOSABatch) -> dict:
        """Encode, fuse, and decode in one forward pass.

        Returns
        -------
        dict with keys:
            x_hat : dict of Tensor [B, D_view]
                Per-view reconstructions.
            z : Tensor [B, joint_latent_dim]
                Sampled latent vector.
            mu : Tensor [B, joint_latent_dim]
                Posterior mean.
            logvar : Tensor [B, joint_latent_dim]
                Posterior log-variance.
        """
        # Encode full batch per view so every DDP rank executes the same ops.
        # Missing samples have zero inputs and produce zero-ish embeddings;
        # their reconstruction loss is masked out downstream.
        view_embeddings = {}
        sample_masks = {}
        for name in self.view_order:
            x = batch.encoder_inputs[name]

            sample_mask = batch.missing_masks[name].any(dim=1)
            sample_masks[name] = sample_mask

            emb = self.encoders[name](x, batch.conditionals)
            emb[~sample_mask] = 0.0
            view_embeddings[name] = emb

        # Fuse view embeddings into joint latent space
        mu, logvar, z = self.latent_space(view_embeddings, self.view_order, sample_masks)

        # Decode each view
        x_hat = {}
        for name in self.view_order:
            x_hat[name] = self.decoders[name](z, batch.conditionals)

        return {"x_hat": x_hat, "z": z, "mu": mu, "logvar": logvar}

    def _current_kl_weight(self) -> float:
        """Return the KL weight for the current training epoch."""
        return _kl_weight_for_epoch(self.current_epoch, self.config)

    def _compute_losses(self, batch: MOSABatch, out: dict) -> dict:
        """Compute reconstruction, KL, and contrastive loss components.

        Returns
        -------
        dict
            Keys: recon, kl, contrastive, recon_metrics.
        """
        recon_loss, recon_metrics = reconstruction_loss(
            x_hat=out["x_hat"],
            x=batch.decoder_targets,
            mask=batch.missing_masks,
            group=batch.source_ids if any(
                vc.loss_type == "macro" for vc in self.config.views.values()
            ) else None,
            sample_weights=batch.sample_weights,
            loss_type="macro" if any(
                vc.loss_type == "macro" for vc in self.config.views.values()
            ) else "mean",
        )

        kl_loss = kl_divergence(out["mu"], out["logvar"])

        c_loss = torch.tensor(0.0, device=self.device)
        if self.config.contrastive_weight > 0:
            c_loss = contrastive_loss(out["mu"], batch.tissue_labels)

        return {
            "recon": recon_loss,
            "kl": kl_loss,
            "contrastive": c_loss,
            "recon_metrics": recon_metrics,
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Ensure MOSABatch is moved to the correct device."""
        if isinstance(batch, MOSABatch):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(self, batch: MOSABatch, batch_idx: int):
        """Two-phase training step: discriminator update, then VAE update."""
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        if isinstance(optimizers, list):
            opt_vae = optimizers[0]
            opt_disc = optimizers[1] if len(optimizers) > 1 else None
        else:
            opt_vae = optimizers
            opt_disc = None

        out = self.forward(batch)
        losses = self._compute_losses(batch, out)
        current_kl_weight = self._current_kl_weight()

        # Phase 1: train discriminator on detached z
        adv_loss_val = torch.tensor(0.0, device=self.device)
        disc_loss_val = torch.tensor(0.0, device=self.device)

        if self.discriminator is not None and opt_disc is not None:
            disc_pred = self.discriminator(out["z"].detach())
            disc_loss_val = adversarial_loss(
                disc_pred, batch.source_ids, self.class_weights
            )
            opt_disc.zero_grad()
            self.manual_backward(disc_loss_val)
            opt_disc.step()

            # Phase 2: adversarial component for VAE (fool discriminator)
            adv_pred = self.discriminator(out["z"])
            adv_loss_val = adversarial_loss(
                adv_pred, batch.source_ids, self.class_weights
            )

        # VAE total loss
        total = (
            losses["recon"]
            + current_kl_weight * losses["kl"]
            + self.config.contrastive_weight * losses["contrastive"]
            - self.config.adv_weight * adv_loss_val
        )

        opt_vae.zero_grad()
        self.manual_backward(total)
        opt_vae.step()

        # Logging
        self.log("train/loss", total, prog_bar=True)
        self.log("train/recon", losses["recon"])
        self.log("train/kl", losses["kl"])
        self.log("train/kl_weight", current_kl_weight)
        for omic_name, omic_loss in losses["recon_metrics"]["omic_losses"].items():
            self.log(f"train/recon_{omic_name}", omic_loss)
        for omic_name, group_losses in losses["recon_metrics"]["group_omic_losses"].items():
            for g_idx, g_loss in group_losses.items():
                g_name = (self.model_type_names[g_idx]
                          if self.model_type_names and g_idx < len(self.model_type_names)
                          else str(g_idx))
                self.log(f"train/recon_{omic_name}_{g_name}", g_loss)
        if self.config.contrastive_weight > 0:
            self.log("train/contrastive", losses["contrastive"])
        if self.discriminator is not None:
            self.log("train/disc_loss", disc_loss_val)
            self.log("train/adv_loss", adv_loss_val)


    def on_train_epoch_end(self) -> None:
        """Step LR schedulers at end of training epoch."""
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return
        if not isinstance(schedulers, list):
            schedulers = [schedulers]
        for sched in schedulers:
            if sched is not None:
                sched.step()

    def validation_step(self, batch: MOSABatch, batch_idx: int):
        """Compute and log validation losses (no adversarial component)."""
        out = self.forward(batch)
        losses = self._compute_losses(batch, out)

        current_kl_weight = self._current_kl_weight()

        total = (
            losses["recon"]
            + current_kl_weight * losses["kl"]
            + self.config.contrastive_weight * losses["contrastive"]
        )

        self.log("val/loss", total, prog_bar=True, sync_dist=True)
        self.log("val/recon", losses["recon"], sync_dist=True)
        self.log("val/kl", losses["kl"], sync_dist=True)
        for omic_name, omic_loss in losses["recon_metrics"]["omic_losses"].items():
            self.log(f"val/recon_{omic_name}", omic_loss, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Log combined train + val epoch summary on rank 0."""
        if self.trainer.is_global_zero:
            metrics = self.trainer.callback_metrics
            logger.debug("epoch %d", self.current_epoch)

            train_parts = ["  train"]
            for key in ("train/loss", "train/recon", "train/kl"):
                if key in metrics:
                    train_parts.append(f"{key.split('/')[-1]}={metrics[key]:.4f}")
            logger.debug(" | ".join(train_parts))

            val_parts = ["  val  "]
            for key in ("val/loss", "val/recon", "val/kl"):
                if key in metrics:
                    val_parts.append(f"{key.split('/')[-1]}={metrics[key]:.4f}")
            logger.debug(" | ".join(val_parts))

    @torch.no_grad()
    def predict(
        self,
        loader: torch.utils.data.DataLoader,
        force_source_id: int | None = None,
        n_batches: int | None = None,
    ) -> dict:
        """Inference on a dataloader.

        Parameters
        ----------
        loader : DataLoader
            Data to infer on.
        force_source_id : int or None
            If set, override batch one-hot with this source ID.
        n_batches : int or None
            Total number of batches (required if force_source_id is set).

        Returns
        -------
        dict
            Keys: z [N, latent_dim], x_hat (per-omic [N, D]), sample_names.
        """
        self.eval()

        all_z: list[torch.Tensor] = []
        all_x_hat: dict[str, list[torch.Tensor]] = {}
        all_names: list[str] = []

        for batch in loader:
            batch = batch.to(self.device)

            if force_source_id is not None:
                if n_batches is None or n_batches <= 0:
                    raise ValueError("n_batches must be provided when force_source_id is used")
                if force_source_id < 0 or force_source_id >= n_batches:
                    raise ValueError(
                        f"force_source_id={force_source_id} out of range for n_batches={n_batches}"
                    )

                # Conditionals start with one-hot model_type block.
                conditionals = batch.conditionals.clone()
                conditionals[:, :n_batches] = 0.0
                conditionals[:, force_source_id] = 1.0
                batch.conditionals = conditionals

            out = self.forward(batch)

            all_z.append(out["z"].cpu())
            all_names.extend(batch.sample_names)

            for omic, recon in out["x_hat"].items():
                all_x_hat.setdefault(omic, []).append(recon.cpu())

        return {
            "z": torch.cat(all_z).numpy(),
            "x_hat": {
                omic: torch.cat(chunks).numpy()
                for omic, chunks in all_x_hat.items()
            },
            "sample_names": all_names,
        }

    def configure_optimizers(self):
        """Set up Adam optimizers and optional StepLR schedulers."""
        vae_params = (
            list(self.encoders.parameters())
            + list(self.decoders.parameters())
            + list(self.latent_space.parameters())
        )
        opt_vae = torch.optim.Adam(vae_params, lr=self.config.learning_rate)
        optimizers = [opt_vae]
        schedulers = []

        if self.discriminator is not None:
            opt_disc = torch.optim.Adam(
                self.discriminator.parameters(), lr=self.config.adv_learning_rate
            )
            optimizers.append(opt_disc)

        if self.config.lr_scheduler == "step":
            for opt in optimizers:
                schedulers.append(
                    torch.optim.lr_scheduler.StepLR(
                        opt,
                        step_size=self.config.lr_step_size,
                        gamma=self.config.lr_gamma,
                    )
                )

        if schedulers:
            return optimizers, schedulers
        return optimizers
