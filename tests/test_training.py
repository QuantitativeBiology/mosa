from __future__ import annotations

import numpy as np
import pytorch_lightning as pl
import pytest
import torch

from mosa.models.mosa.datamodule import MOSADataModule
from mosa.models.mosa.vae.vae_module import VAE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Smoke tests — Lightning fast_dev_run
#
# Purpose: verify the full training loop (data → forward → loss → backward →
# optimizer step) runs without error. Catches broken batch keys, shape
# mismatches inside training_step, and optimizer configuration bugs.
# One batch, no checkpointing, no logging overhead.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Forward pass sanity — no NaN / Inf in outputs
#
# Purpose: catch numerical instability before training even starts.
# Relevant for VAEs because log operations in KL, masked mean reductions,
# and PoE precision accumulation are all NaN sources with edge-case inputs.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gradient flow
#
# Purpose: verify that every trainable parameter in the encoder, decoder,
# and latent space receives a non-None gradient after a single backward pass.
# Catches disconnected computation graphs (a common bug when refactoring
# modules), misplaced detach() calls, and dead code paths.
#
# The discriminator is excluded because it uses a separate optimizer and is
# only updated on the detached z — its gradient path is tested implicitly
# by the smoke test (training_step exercises both optimizers).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Overfit test
#
# Purpose: the single most important ML-specific test. If a model cannot
# memorise one batch after many gradient steps, something is fundamentally
# broken — gradient flow, loss computation, or the optimizer step is not
# being applied. Uses Lightning's overfit_batches=1 to force training on a
# fixed single batch.
#
# For standardised N(0,1) data with MSE loss, random-init reconstruction
# loss is ~1.0. After 100 steps of overfit training it must fall below 0.5.
# The threshold is deliberately loose to avoid flakiness; the goal is to
# detect catastrophic failure (e.g. loss stays at 1.0), not measure quality.
# ---------------------------------------------------------------------------


class _LossTracker(pl.Callback):
    """Records train/recon at every training batch end."""

    def __init__(self):
        self.losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        if "train/recon" in metrics:
            self.losses.append(float(metrics["train/recon"]))


def test_vae_can_overfit_single_batch(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=8)
    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=100, batch_size=8, output_dir=str(tmp_path)
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
    # Train-only loop: overfit_batches forces a val step Lightning won't skip
    # when val_dataloader returns None, so set limit_*_batches manually instead.
    trainer = _silent_trainer(
        no_val=True,
        max_epochs=100,
        limit_train_batches=1,
        log_every_n_steps=1,
        callbacks=[tracker],
    )
    trainer.fit(vae, dm)

    assert len(tracker.losses) >= 2, "No losses were recorded"
    assert tracker.losses[-1] < tracker.losses[0], (
        f"Loss did not decrease: first={tracker.losses[0]:.4f}, "
        f"last={tracker.losses[-1]:.4f}"
    )
    assert tracker.losses[-1] < 0.5, (
        f"Final reconstruction loss {tracker.losses[-1]:.4f} is too high "
        f"for a model that should have memorised one batch"
    )
