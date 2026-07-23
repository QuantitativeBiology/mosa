from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mosa.models.api import MultiOmicModel
from mosa.models.mofa import MOFAModel
from mosa.models.mofa.config import MOFAConfig
from mosa.models.mosa import MOSAModel

_mofapy2_available = importlib.util.find_spec("mofapy2") is not None
_mofax_available = importlib.util.find_spec("mofax") is not None
_mofa_available = _mofapy2_available and _mofax_available

skip_mofa = pytest.mark.skipif(
    not _mofa_available,
    reason="mofapy2 and/or mofax not installed",
)


# Helpers


def _split(dataset, n_train=16):
    indices = np.arange(dataset.n_samples)
    return dataset.subset(indices[:n_train]), dataset.subset(indices[n_train:])


# MOSAModel tests


def test_vae_fit(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)
    assert not (tmp_path / "train" / "latent.parquet").exists()
    model.save_outputs()
    assert (tmp_path / "train" / "latent.parquet").exists()


def test_vae_transform_shape(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)
    z = model.transform(train)
    assert z.shape == (train.n_samples, model_cfg.joint_latent_dim)
    assert z.dtype in (np.float32, np.float64)
    assert not np.isnan(z).any()


def test_transform_unseen_model_type_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Inference data with a model_type absent from the fit-time categories must
    raise, not silently miscode it (previously a negative-index wrap)."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, None)

    unseen = make_multi_omic_dataset(n_samples=8, seed=1)
    unseen.metadata["model_type"] = ["NovelType"] * unseen.n_samples
    with pytest.raises(ValueError, match="not seen during fit"):
        model.transform(unseen)
    with pytest.raises(ValueError, match="not seen during fit"):
        model.reconstruct(unseen)


def test_vae_reconstruct_shapes(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)
    recon = model.reconstruct(train)
    assert set(recon.keys()) == set(train.view_names)
    for name in train.view_names:
        expected_dim = train.views[name].shape[1]
        assert recon[name].shape == (train.n_samples, expected_dim)


def test_vae_fit_no_val(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    model.save_outputs()
    assert (tmp_path / "train" / "latent.parquet").exists()


@pytest.mark.parametrize("fusion_method", ["concat", "poe"])
def test_vae_concat_and_poe(
    fusion_method, make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(
        dataset,
        fusion_method=fusion_method,
        joint_latent_dim=16,
        output_dir=str(tmp_path),
    )
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)
    z = model.transform(train)
    assert z.shape == (train.n_samples, model_cfg.joint_latent_dim)


# MOFAModel tests


def _mofa_data_cfg(dataset):
    from mosa.config import DataConfig
    return DataConfig(path="unused", views=list(dataset.view_names))


@skip_mofa
def test_mofa_fit(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    save_path = str(tmp_path / "mofa.hdf5")
    model = MOFAModel(_mofa_data_cfg(dataset), MOFAConfig(n_factors=5), save_path=save_path)
    model.fit(train)
    assert not (tmp_path / "mofa.hdf5").exists()
    model.save_outputs()
    assert (tmp_path / "mofa.hdf5").exists()


@skip_mofa
def test_mofa_transform_shape(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    save_path = str(tmp_path / "mofa.hdf5")
    model = MOFAModel(_mofa_data_cfg(dataset), MOFAConfig(n_factors=5), save_path=save_path)
    model.fit(train)
    model.save_outputs()
    z = model.transform(train)
    # MOFA with ARD prunes uninformative factors, so the surviving count is
    # data-dependent and <= n_factors; the SE guarantee is a 2D [N, k>=1] array.
    assert z.ndim == 2
    assert z.shape[0] == train.n_samples
    assert 1 <= z.shape[1] <= 5


@skip_mofa
def test_mofa_reconstruct_shapes(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    save_path = str(tmp_path / "mofa.hdf5")
    model = MOFAModel(_mofa_data_cfg(dataset), MOFAConfig(n_factors=5), save_path=save_path)
    model.fit(train)
    model.save_outputs()
    recon = model.reconstruct(train)
    assert set(recon.keys()) == set(train.view_names)
    for name in train.view_names:
        expected_dim = train.views[name].shape[1]
        assert recon[name].shape == (train.n_samples, expected_dim)


@skip_mofa
def test_mofa_unseen_data_raises(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    save_path = str(tmp_path / "mofa.hdf5")
    model = MOFAModel(_mofa_data_cfg(dataset), MOFAConfig(n_factors=5), save_path=save_path)
    model.fit(train)
    model.save_outputs()

    other = make_multi_omic_dataset(n_samples=20, seed=99)
    other.metadata.index = [f"other_{i:03d}" for i in range(other.n_samples)]

    with pytest.raises(NotImplementedError):
        model.transform(other)


# Interface compliance


def test_all_models_are_multi_omic_model(sample_dataset, make_mosa_config, tmp_path):
    data_cfg, model_cfg = make_mosa_config(sample_dataset, output_dir=str(tmp_path))
    assert isinstance(MOSAModel(data_cfg, model_cfg), MultiOmicModel)
    assert isinstance(MOFAModel(), MultiOmicModel)


# Error handling: calling API before fit()


def test_transform_before_fit_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.transform(dataset)


def test_reconstruct_before_fit_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.reconstruct(dataset)


def test_save_before_fit_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.save(tmp_path / "model.pt")


# Output quality: no NaN, deterministic eval


def test_vae_transform_no_nan(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z = model.transform(train)
    assert not np.isnan(z).any(), "transform() produced NaN values"
    assert not np.isinf(z).any(), "transform() produced Inf values"


def test_vae_reconstruct_no_nan(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20, missing_frac=0.2)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    recon = model.reconstruct(train)
    for name, arr in recon.items():
        assert not np.isnan(arr).any(), f"reconstruct() produced NaN for view '{name}'"
        assert not np.isinf(arr).any(), f"reconstruct() produced Inf for view '{name}'"


def test_vae_transform_deterministic(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z1 = model.transform(train)
    z2 = model.transform(train)
    np.testing.assert_array_equal(z1, z2)


# Save / load


def test_vae_save_creates_checkpoint(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    save_path = tmp_path / "model.pt"
    model.save(save_path)

    import torch
    checkpoint = torch.load(str(save_path), weights_only=False)
    assert "state_dict" in checkpoint
    hp = checkpoint["hyper_parameters"]
    assert "view_input_dims" in hp
    assert "conditional_dim" in hp
    assert "n_batches" in hp


def test_vae_load_preserves_architecture(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    save_path = tmp_path / "model.pt"
    model.save(save_path)

    loaded = MOSAModel.load(save_path)
    assert loaded._model.view_input_dims == model._model.view_input_dims
    assert loaded._model.conditional_dim == model._model.conditional_dim
    assert loaded._model.n_batches == model._model.n_batches


def test_vae_load_preserves_weights(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Weights survive a save/load round-trip: same batch → identical predictions."""
    from mosa.models.mosa.datamodule import MOSADataModule

    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    save_path = tmp_path / "model.pt"
    model.save(save_path)
    loaded = MOSAModel.load(save_path)

    # Use a fresh datamodule with training scalers to get a proper dataloader
    inf_dm = MOSADataModule(train_data=train, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    inf_dm.setup_inference(model._datamodule)
    loader = inf_dm.train_eval_dataloader()

    result_orig = model._model.predict(loader)
    result_loaded = loaded._model.predict(loader)

    np.testing.assert_allclose(result_orig["z"], result_loaded["z"], atol=1e-5)


# Edge cases


def test_vae_single_view(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20, view_dims={"rna": 40})
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z = model.transform(train)
    assert z.shape == (train.n_samples, model_cfg.joint_latent_dim)
    assert not np.isnan(z).any()


def test_vae_with_missing_data(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Model handles per-feature missingness without NaN in outputs."""
    dataset = make_multi_omic_dataset(n_samples=20, missing_frac=0.3)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z = model.transform(train)
    assert not np.isnan(z).any()


# Inference must reuse training scalers/categories, not refit (regression)


def _shift_dataset(data, shift: float):
    """Copy of `data` with a constant added to every view (same masks/metadata)."""
    from mosa.data.dataset import MultiOmicDataset

    shifted_views = {name: X + shift for name, X in data.views.items()}
    return MultiOmicDataset(shifted_views, data.masks, data.metadata, data.feature_names)


def test_transform_reuses_training_scaler_not_refit(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    """A constant shift in new data must survive standardization with the
    training scaler: transform(A) and transform(A + shift) must differ.

    Before the fix, transform() built a fresh scaler on whatever data was
    passed in, so a constant shift got fully absorbed by the refit mean and
    the two latents came out identical.
    """
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    shifted = _shift_dataset(train, shift=1000.0)

    z1 = model.transform(train)
    z2 = model.transform(shifted)

    # A constant shift, standardized with a scaler *refit on the shifted
    # data itself*, cancels out exactly (the new mean absorbs the shift),
    # so a buggy refit-per-call implementation yields z1 == z2 up to
    # floating-point noise (~1e-6). Reusing the training scaler instead
    # keeps the shift as a large additive offset in standardized space, so
    # the two latents diverge by orders of magnitude more than that noise
    # floor.
    mean_abs_diff = np.abs(z1 - z2).mean()
    assert mean_abs_diff > 0.01, (
        f"transform() produced near-identical latents for shifted input "
        f"(mean abs diff={mean_abs_diff:.2e}); the scaler is being refit on "
        f"the new data instead of reusing the training scaler"
    )


def test_transform_does_not_mutate_trained_scaler(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    """Calling transform()/reconstruct() must not corrupt the model's own scalers."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    means_before = {
        name: scaler.mean_.copy()
        for name, scaler in model._datamodule.scalers.items()
        if scaler is not None
    }

    shifted = _shift_dataset(train, shift=1000.0)
    model.transform(shifted)
    model.reconstruct(shifted)

    for name, mean_before in means_before.items():
        np.testing.assert_array_equal(model._datamodule.scalers[name].mean_, mean_before)


def test_reconstruct_returns_original_scale(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    """reconstruct() output should be in the same scale as the raw input, not
    standardized (near-unit-variance) values.

    The fixture's raw views are already ~N(0, 1), which would make a
    standardized (bugged) reconstruction indistinguishable from a correctly
    inverse-transformed one on scale alone. Rescale the view to a large
    offset/spread so the two are obviously different orders of magnitude.
    """
    from mosa.data.dataset import MultiOmicDataset

    dataset = make_multi_omic_dataset(n_samples=20, view_dims={"view_a": 50})
    rescaled_views = {
        name: X * 200.0 + 5000.0 for name, X in dataset.views.items()
    }
    dataset = MultiOmicDataset(
        rescaled_views, dataset.masks, dataset.metadata, dataset.feature_names
    )
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    recon = model.reconstruct(train)
    raw_scale = np.abs(train.views["view_a"]).mean()
    recon_scale = np.abs(recon["view_a"]).mean()

    # A standardized (bugged) reconstruction would sit near unit scale
    # (~1), orders of magnitude below the raw input's ~5000 scale.
    assert recon_scale > 0
    assert 0.1 * raw_scale < recon_scale < 10 * raw_scale
