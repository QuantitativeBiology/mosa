from __future__ import annotations

import copy

import numpy as np
import pytorch_lightning as pl
import pytest
import torch

from mosa.models.mosa.datamodule import MOSADataModule
from mosa.models.mosa.vae.losses import adversarial_loss
from mosa.models.mosa.vae.vae_module import VAE


# Helpers


def _make_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path, **dataset_kwargs):
    """Return a freshly constructed (VAE, MOSADataModule) pair after setup()."""
    dataset = make_multi_omic_dataset(**dataset_kwargs)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    dm = MOSADataModule(train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()
    vae = VAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
    )
    return vae, dm


def _make_vae_and_dm_split(make_multi_omic_dataset, make_mosa_config, tmp_path, n_samples=20):
    """Return (VAE, MOSADataModule) with a train/val split."""
    dataset = make_multi_omic_dataset(n_samples=n_samples)
    indices = np.arange(n_samples)
    train = dataset.subset(indices[:16])
    val = dataset.subset(indices[16:])
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    dm = MOSADataModule(train_data=train, val_data=val, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()
    vae = VAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
    )
    return vae, dm


def _silent_trainer(no_val: bool = False, **kwargs) -> pl.Trainer:
    if no_val:
        kwargs.setdefault("limit_val_batches", 0)
        kwargs.setdefault("num_sanity_val_steps", 0)
    return pl.Trainer(
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        **kwargs,
    )


# Smoke tests: one fast_dev_run batch exercises the full training loop
# (forward, loss, backward, optimizer step) to catch broken batch keys,
# shape mismatches, and optimizer configuration bugs cheaply.


def test_vae_fast_dev_run(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    # fast_dev_run forces a val batch even when val_dataloader returns None;
    # expand it manually so this no-val test runs train-only.
    trainer = _silent_trainer(no_val=True, max_epochs=1, limit_train_batches=1)
    trainer.fit(vae, dm)


def test_vae_fast_dev_run_with_val(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_vae_and_dm_split(make_multi_omic_dataset, make_mosa_config, tmp_path)
    trainer = _silent_trainer(fast_dev_run=True)
    trainer.fit(vae, dm)


def test_vae_fast_dev_run_poe(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset()
    data_cfg, model_cfg = make_mosa_config(dataset, fusion_method="poe", output_dir=str(tmp_path))
    dm = MOSADataModule(train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()
    vae = VAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
    )
    _silent_trainer(no_val=True, max_epochs=1, limit_train_batches=1).fit(vae, dm)


# Forward pass sanity: no NaN/Inf in outputs. Catches numerical instability
# from KL log operations, masked mean reductions, and PoE precision
# accumulation before training starts.


def test_vae_forward_no_nan(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    vae.eval()
    batch = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        out = vae(batch)
    assert torch.isfinite(out["mu"]).all(), "NaN/Inf in mu"
    assert torch.isfinite(out["logvar"]).all(), "NaN/Inf in logvar"
    assert torch.isfinite(out["z"]).all(), "NaN/Inf in z"
    for name, x_hat in out["x_hat"].items():
        assert torch.isfinite(x_hat).all(), f"NaN/Inf in x_hat['{name}']"


def test_vae_forward_no_nan_with_missing_data(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Missing views (partial masks) must not produce NaN in any output."""
    vae, dm = _make_vae_and_dm(
        make_multi_omic_dataset, make_mosa_config, tmp_path, missing_frac=0.4
    )
    vae.eval()
    batch = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        out = vae(batch)
    assert torch.isfinite(out["mu"]).all()
    assert torch.isfinite(out["z"]).all()
    for name, x_hat in out["x_hat"].items():
        assert torch.isfinite(x_hat).all(), f"NaN/Inf in x_hat['{name}'] with missing data"


# Missing-view zeroing: a fully-missing view's embedding must be force-zeroed
# before fusion (vae_module.py emb[~sample_mask]=0.0), so arbitrary encoder
# inputs for a missing sample cannot leak into the latent. Uses
# fusion_method="concat"; zeroing happens before fusion so the contract holds
# for poe too.


def test_missing_view_embedding_zeroed(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    vae.eval()

    batch = next(iter(dm.train_dataloader()))
    view = vae.view_order[0]
    sample_idx = 0

    batch1 = copy.deepcopy(batch)
    batch1["missing_masks"][view][sample_idx, :] = False

    batch2 = copy.deepcopy(batch1)
    batch2["encoder_inputs"][view][sample_idx] = torch.randn_like(
        batch2["encoder_inputs"][view][sample_idx]
    )

    with torch.no_grad():
        out1 = vae(batch1)
        out2 = vae(batch2)

    assert torch.equal(out1["mu"][sample_idx], out2["mu"][sample_idx])


# Gradient flow: every trainable parameter in the encoder, decoder, and
# latent space must receive a non-None gradient after one backward pass.
# Catches disconnected computation graphs, misplaced detach() calls, and
# dead code paths. The discriminator is excluded: it uses a separate
# optimizer on the detached z, and its gradient path is exercised by the
# smoke test instead.


def test_vae_params_all_receive_gradients(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    vae.train()

    batch = next(iter(dm.train_dataloader()))
    out = vae(batch)
    losses = vae._compute_losses(batch, out)
    total = losses["recon"] + vae.config.kl_weight * losses["kl"]
    total.backward()

    no_grad = [
        name
        for name, p in vae.named_parameters()
        if p.requires_grad and p.grad is None and "discriminator" not in name
    ]
    assert not no_grad, f"Parameters with no gradient: {no_grad}"


# Adversarial / discriminator wiring: the two-optimizer manual-optimization
# path (discriminator trained on detached z) must wire gradients only where
# intended; the detach must isolate the encoder/decoder/latent space from
# the discriminator's own loss.


def _make_adversarial_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """VAE + datamodule with adv_weight>0 (conftest default gives 2 model_type
    categories, so n_batches=2 and self.discriminator is not None)."""
    dataset = make_multi_omic_dataset()
    data_cfg, model_cfg = make_mosa_config(
        dataset, adv_weight=1.0, adv_learning_rate=1e-3, output_dir=str(tmp_path)
    )
    dm = MOSADataModule(train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()
    vae = VAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
    )
    return vae, dm


def test_vae_fast_dev_run_adversarial(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_adversarial_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    assert vae.discriminator is not None
    trainer = _silent_trainer(no_val=True, max_epochs=1, limit_train_batches=1)
    trainer.fit(vae, dm)


def test_discriminator_params_receive_gradients(make_multi_omic_dataset, make_mosa_config, tmp_path):
    vae, dm = _make_adversarial_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    vae.train()

    batch = next(iter(dm.train_dataloader()))
    out = vae(batch)
    disc_loss = adversarial_loss(
        vae.discriminator(out["z"].detach()), batch["source_ids"], vae.class_weights
    )
    vae.zero_grad()
    disc_loss.backward()

    no_grad = [
        name
        for name, p in vae.discriminator.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert not no_grad, f"Discriminator parameters with no gradient: {no_grad}"


def test_adversarial_z_detach_wiring(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Guards the real training_step call sites (vae_module.py:245, 254): phase 1
    calls the discriminator on a detached z, phase 2 on a non-detached z. A hook
    on the discriminator captures both inputs during one real training step."""
    vae, dm = _make_adversarial_vae_and_dm(make_multi_omic_dataset, make_mosa_config, tmp_path)
    assert vae.discriminator is not None

    captured: list[torch.Tensor] = []
    vae.discriminator.register_forward_pre_hook(lambda module, args: captured.append(args[0]))

    trainer = _silent_trainer(no_val=True, max_epochs=1, limit_train_batches=1)
    trainer.fit(vae, dm)

    assert len(captured) >= 2, f"Discriminator called {len(captured)} times, expected >= 2"
    # Phase 1 (line 245): disc_pred = self.discriminator(out["z"].detach())
    assert captured[0].grad_fn is None and not captured[0].requires_grad, (
        "Phase-1 discriminator input is not detached from the VAE graph"
    )
    # Phase 2 (line 254): adv_pred = self.discriminator(out["z"])
    assert captured[1].grad_fn is not None, (
        "Phase-2 discriminator input is unexpectedly detached from the VAE graph"
    )


def test_vae_fast_dev_run_contrastive(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset()
    data_cfg, model_cfg = make_mosa_config(
        dataset, contrastive_weight=1.0, output_dir=str(tmp_path)
    )
    dm = MOSADataModule(train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()
    vae = VAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
    )
    trainer = _silent_trainer(no_val=True, max_epochs=1, limit_train_batches=1)
    trainer.fit(vae, dm)


# Training-loop wiring: SE-correctness, not model quality. Asserts the
# manual-optimization loop (zero_grad, manual_backward, step, scheduler, KL
# warmup) reduces loss over many steps on a fixed batch; checks direction,
# not magnitude. Catches an inverted LR, an optimizer built on the wrong
# param list, or a missing step(), all of which would still pass
# gradient-existence and fast_dev_run checks. Absolute-loss thresholds
# (convergence quality) are out of scope.


class _LossTracker(pl.Callback):
    """Records train/recon at every training batch end."""

    def __init__(self):
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        if "train/recon" in metrics:
            self.losses.append(float(metrics["train/recon"]))


def test_vae_training_loop_reduces_loss(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """The optimizer loop reduces reconstruction loss over many steps on one batch."""
    dataset = make_multi_omic_dataset(n_samples=8)
    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=50, batch_size=8, output_dir=str(tmp_path)
    )
    dm = MOSADataModule(train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()
    vae = VAE(
        config=model_cfg,
        view_input_dims=dm.view_input_dims,
        conditional_dim=dm.conditional_dim,
        n_batches=dm.n_batches,
    )
    tracker = _LossTracker()
    # Train-only loop over one fixed batch; limit_train_batches=1 keeps the batch
    # constant so the loss trend reflects optimization, not data variation.
    trainer = _silent_trainer(
        no_val=True,
        max_epochs=50,
        limit_train_batches=1,
        log_every_n_steps=1,
        callbacks=[tracker],
    )
    trainer.fit(vae, dm)

    assert len(tracker.losses) >= 2, "No losses were recorded"
    assert tracker.losses[-1] < tracker.losses[0], (
        f"Training loop did not reduce loss over {len(tracker.losses)} steps: "
        f"first={tracker.losses[0]:.4f}, last={tracker.losses[-1]:.4f}"
    )
