from __future__ import annotations

import types

import pytest
import torch

from mosa.models.mosa.vae.decoder import OmicDecoder
from mosa.models.mosa.vae.discriminator import Discriminator
from mosa.models.mosa.vae.encoder import OmicEncoder
from mosa.models.mosa.vae.latent import BaseLatentSpace
from mosa.models.mosa.vae.losses import (
    adversarial_loss,
    kl_divergence,
    reconstruction_loss,
)
from mosa.models.mosa.vae.vae_module import _kl_weight_for_epoch

# Encoder


def test_encoder_output_shape():
    encoder = OmicEncoder(input_dim=50, hidden_dims=[32, 16], latent_dim=16)
    x = torch.randn(8, 50)
    out = encoder(x)
    assert out.shape == (8, 16)


def test_encoder_view_dropout():
    torch.manual_seed(0)
    encoder = OmicEncoder(
        input_dim=50,
        hidden_dims=[32, 16],
        latent_dim=16,
        view_dropout_p=1.0,
    )
    x = torch.randn(8, 50)
    x_zero = torch.zeros(8, 50)

    # p=1.0: training always zeros x, so output equals the zero-input forward pass.
    # allclose (not equal): BatchNorm1d's multi-threaded reduction kernel doesn't
    # cancel to exactly 0 for a zero-variance batch the way single-threaded does.
    encoder.train()
    out_train = encoder(x)
    out_zero = encoder(x_zero)
    assert torch.allclose(out_train, out_zero, atol=1e-6)

    # eval: dropout is skipped, so non-zero x produces different output than x_zero
    encoder.eval()
    out_eval = encoder(x)
    out_eval_zero = encoder(x_zero)
    assert not torch.equal(out_eval, out_eval_zero)


# Decoder


def test_decoder_output_shape():
    decoder = OmicDecoder(
        output_dim=50, cond_dim=10, hidden_dims=[32, 16], latent_dim=16
    )
    z = torch.randn(8, 16)
    cond = torch.randn(8, 10)
    out = decoder(z, cond)
    assert out.shape == (8, 50)


def test_decoder_no_cond():
    decoder = OmicDecoder(
        output_dim=50, cond_dim=0, hidden_dims=[32, 16], latent_dim=16
    )
    z = torch.randn(8, 16)
    cond = torch.zeros(8, 0)
    out = decoder(z, cond)
    assert out.shape == (8, 50)


# Latent space


def test_concat_latent_shapes():
    latent = BaseLatentSpace.create("concat", {"a": 16, "b": 16}, latent_dim=8)
    embeddings = {"a": torch.randn(4, 16), "b": torch.randn(4, 16)}
    mu, logvar, z = latent(embeddings, ["a", "b"])
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert z.shape == (4, 8)


def test_poe_latent_shapes():
    latent = BaseLatentSpace.create("poe", {"a": 16, "b": 16}, latent_dim=8)
    embeddings = {"a": torch.randn(4, 16), "b": torch.randn(4, 16)}
    mu, logvar, z = latent(embeddings, ["a", "b"])
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)
    assert z.shape == (4, 8)


def test_poe_masks_missing_view():
    torch.manual_seed(0)
    latent = BaseLatentSpace.create("poe", {"a": 16, "b": 16}, latent_dim=8)
    latent.eval()

    B = 6
    embeddings = {"a": torch.randn(B, 16), "b": torch.randn(B, 16)}
    # Mask out view "b" for the first 3 samples
    masks = {
        "a": torch.ones(B, dtype=torch.bool),
        "b": torch.tensor([False, False, False, True, True, True]),
    }
    mu, logvar, z = latent(embeddings, ["a", "b"], sample_masks=masks)

    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
    assert torch.isfinite(z).all()


def test_poe_masked_view_contributes_zero():
    """A masked view's embedding must not affect mu/logvar (latent.py:174-180:
    precision_v*mask and mu_v*mask zero its contribution before accumulation)."""
    torch.manual_seed(0)
    latent = BaseLatentSpace.create("poe", {"a": 16, "b": 16}, latent_dim=8)
    latent.eval()

    B = 6
    emb_a = torch.randn(B, 16)
    emb_b = torch.randn(B, 16)
    masks = {
        "a": torch.ones(B, dtype=torch.bool),
        "b": torch.tensor([False, False, False, True, True, True]),
    }

    mu1, logvar1, _ = latent({"a": emb_a, "b": emb_b}, ["a", "b"], sample_masks=masks)

    # Change view "b"'s embedding only for the masked samples (first 3 rows).
    emb_b_changed = emb_b.clone()
    emb_b_changed[:3] = torch.randn(3, 16)
    mu2, logvar2, _ = latent(
        {"a": emb_a, "b": emb_b_changed}, ["a", "b"], sample_masks=masks
    )

    assert torch.equal(mu1[:3], mu2[:3])
    assert torch.equal(logvar1[:3], logvar2[:3])


def test_poe_fully_missing_sample_prior_fallback():
    """A sample with every view masked out falls back to the isotropic prior:
    mu=0 exactly, logvar=-log(1+EPS)~=0 (latent.py:166-183)."""
    torch.manual_seed(0)
    latent = BaseLatentSpace.create("poe", {"a": 16, "b": 16}, latent_dim=8)
    latent.eval()

    B = 4
    embeddings = {"a": torch.randn(B, 16), "b": torch.randn(B, 16)}
    masks = {
        "a": torch.tensor([False, True, True, True]),
        "b": torch.tensor([False, True, True, False]),
    }
    mu, logvar, _ = latent(embeddings, ["a", "b"], sample_masks=masks)

    assert torch.allclose(mu[0], torch.zeros(8), atol=1e-6)
    assert torch.allclose(logvar[0], torch.zeros(8), atol=1e-6)


# @pytest.mark.xfail(
#     reason="Unbounded logvar overflows exp(-logvar) in PoE fusion; whether to "
#     "clamp is a modelling decision, not resolved yet.",
#     strict=True,
# )
# def test_poe_extreme_logvar_stays_finite():
#     """Extreme per-view logvar must not overflow exp(-logvar) into inf/nan mu.
#
#     Reproducer for the CV/HPO eval-mode nan: logvar goes far negative,
#     precision = exp(-logvar) overflows float32 to inf, and the fused
#     mu = mu_precision_sum / precision_sum comes out inf/inf = nan.
#     """
#     torch.manual_seed(0)
#     latent = BaseLatentSpace.create("poe", {"a": 16, "b": 16}, latent_dim=8)
#     latent.eval()
#
#     # Force the logvar half of the shared head to a hugely negative constant.
#     last_linear = [m for m in latent.shared_head.net if isinstance(m, torch.nn.Linear)][
#         -1
#     ]
#     with torch.no_grad():
#         last_linear.weight.zero_()
#         last_linear.bias[:8] = 1e3  # mu
#         last_linear.bias[8:] = -1e3  # logvar
#
#     embeddings = {"a": torch.randn(4, 16), "b": torch.randn(4, 16)}
#     masks = {"a": torch.ones(4, dtype=torch.bool), "b": torch.ones(4, dtype=torch.bool)}
#     mu, logvar, z = latent(embeddings, ["a", "b"], sample_masks=masks)
#
#     assert torch.isfinite(mu).all()
#     assert torch.isfinite(logvar).all()
#     assert torch.isfinite(z).all()


def test_latent_registry_unknown():
    with pytest.raises(ValueError, match="Unknown fusion method"):
        BaseLatentSpace.create("nonexistent", {"a": 16}, latent_dim=8)


def test_reparameterize_eval_returns_mu():
    torch.manual_seed(0)
    latent = BaseLatentSpace.create("concat", {"a": 16}, latent_dim=8)
    latent.eval()
    mu = torch.randn(4, 8)
    logvar = torch.zeros(4, 8)
    out = latent.reparameterize(mu, logvar)
    assert torch.equal(out, mu)


# Losses


def test_reconstruction_loss_full_mask():
    torch.manual_seed(0)
    B, D = 8, 20
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    mask = {"omic": torch.ones(B, D, dtype=torch.bool)}
    loss, _ = reconstruction_loss({"omic": x_hat}, {"omic": x}, mask)
    expected = torch.nn.functional.mse_loss(x_hat, x)
    assert torch.isclose(loss, expected, atol=1e-5)


def test_reconstruction_loss_partial_mask():
    torch.manual_seed(0)
    B, D = 8, 20
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    mask_np = torch.ones(B, D, dtype=torch.bool)
    mask_np[:, :5] = False  # first 5 features always absent
    mask = {"omic": mask_np}

    loss, metrics = reconstruction_loss({"omic": x_hat}, {"omic": x}, mask)

    # Loss must be finite and not equal to unmasked MSE (masking changes result)
    unmasked_loss = torch.nn.functional.mse_loss(x_hat, x)
    assert torch.isfinite(loss)
    assert not torch.isclose(loss, unmasked_loss, atol=1e-5)


def test_reconstruction_loss_all_masked():
    B, D = 8, 20
    x = torch.randn(B, D)
    x_hat = torch.randn(B, D)
    mask = {"omic": torch.zeros(B, D, dtype=torch.bool)}
    loss, _ = reconstruction_loss({"omic": x_hat}, {"omic": x}, mask)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_reconstruction_loss_macro():
    torch.manual_seed(0)
    # Group 0: 3 samples, Group 1: 1 sample — unequal counts
    B, D = 4, 10
    x = torch.zeros(B, D)
    x_hat = torch.zeros(B, D)
    # Group 0 error = 1.0 per feature, Group 1 error = 4.0 per feature
    x_hat[0:3] = 1.0  # group 0: squared error = 1.0 per feature
    x_hat[3:4] = 2.0  # group 1: squared error = 4.0 per feature

    mask = {"omic": torch.ones(B, D, dtype=torch.bool)}
    group = torch.tensor([0, 0, 0, 1])

    loss, _ = reconstruction_loss(
        {"omic": x_hat}, {"omic": x}, mask, group=group, loss_types={"omic": "macro"}
    )

    # Macro: mean of (group0_mean=1.0, group1_mean=4.0) = 2.5
    # Sample-weighted mean would be (3*1.0 + 1*4.0)/4 = 1.75
    assert loss.item() == pytest.approx(2.5, abs=1e-5)


def test_reconstruction_loss_recon_weight_scales_omic_contribution():
    """Per-omic recon_weight scales that omic's contribution to the total loss."""
    B, D = 4, 5
    x = torch.zeros(B, D)
    x_hat_a = torch.ones(B, D)  # per-feature squared error = 1.0
    x_hat_b = torch.ones(B, D) * 2.0  # per-feature squared error = 4.0
    mask = {
        "a": torch.ones(B, D, dtype=torch.bool),
        "b": torch.ones(B, D, dtype=torch.bool),
    }

    equal_loss, _ = reconstruction_loss(
        {"a": x_hat_a, "b": x_hat_b},
        {"a": x, "b": x},
        mask,
    )
    assert equal_loss.item() == pytest.approx(1.0 + 4.0, abs=1e-5)

    weighted_loss, _ = reconstruction_loss(
        {"a": x_hat_a, "b": x_hat_b},
        {"a": x, "b": x},
        mask,
        recon_weights={"a": 2.0, "b": 0.5},
    )
    assert weighted_loss.item() == pytest.approx(2.0 * 1.0 + 0.5 * 4.0, abs=1e-5)


def test_kl_divergence_standard_normal():
    mu = torch.zeros(8, 16)
    logvar = torch.zeros(8, 16)
    kl = kl_divergence(mu, logvar)
    assert kl.item() < 1e-2


def test_kl_divergence_positive():
    torch.manual_seed(0)
    mu = torch.randn(8, 16) * 3.0
    logvar = torch.zeros(8, 16)
    kl = kl_divergence(mu, logvar)
    assert kl.item() > 0.0


def test_kl_divergence_clamps_extreme_logvar():
    """Unstable posteriors (huge logvar) must not blow up to inf/NaN."""
    mu = torch.zeros(4, 8)
    logvar = torch.full((4, 8), 1000.0)
    kl = kl_divergence(mu, logvar)
    assert torch.isfinite(kl)


def test_adversarial_loss_shape():
    torch.manual_seed(0)
    B, C = 8, 3
    pred = torch.randn(B, C)
    target = torch.randint(0, C, (B,))
    loss = adversarial_loss(pred, target)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_adversarial_loss_focal_gamma_downweights_easy_examples():
    """With focal_gamma > 0, confidently-correct (easy) predictions contribute
    less to the loss than under plain cross-entropy."""
    B, C = 6, 3
    target = torch.zeros(B, dtype=torch.long)
    easy_pred = torch.zeros(B, C)
    easy_pred[:, 0] = 10.0  # confident, correct logits

    plain = adversarial_loss(easy_pred, target, focal_gamma=0.0)
    focal = adversarial_loss(easy_pred, target, focal_gamma=2.0)
    assert focal.item() < plain.item()


# Discriminator


def test_discriminator_output_shape():
    disc = Discriminator(enc_dim=16, output_dim=3)
    x = torch.randn(8, 16)
    out = disc(x)
    assert out.shape == (8, 3)


# KL weight schedule


def _make_kl_config(**kwargs) -> types.SimpleNamespace:
    defaults = dict(
        use_kl_scheduler=False,
        kl_weight=0.01,
        kl_weight_final=0.01,
        kl_warmup_epochs=0,
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def test_kl_weight_no_scheduler():
    cfg = _make_kl_config(use_kl_scheduler=False, kl_weight=0.5)
    for epoch in [0, 5, 100]:
        assert _kl_weight_for_epoch(epoch, cfg) == pytest.approx(0.5)


def test_kl_weight_warmup():
    cfg = _make_kl_config(
        use_kl_scheduler=True,
        kl_weight=0.0,
        kl_weight_final=1.0,
        kl_warmup_epochs=10,
    )
    # epoch 0: progress = 1/10 = 0.1
    assert _kl_weight_for_epoch(0, cfg) == pytest.approx(0.1, abs=1e-6)
    # epoch 4: progress = 5/10 = 0.5
    assert _kl_weight_for_epoch(4, cfg) == pytest.approx(0.5, abs=1e-6)
    # epoch 9: progress = 10/10 = 1.0
    assert _kl_weight_for_epoch(9, cfg) == pytest.approx(1.0, abs=1e-6)
    # epoch 20: capped at 1.0
    assert _kl_weight_for_epoch(20, cfg) == pytest.approx(1.0, abs=1e-6)
