from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from mosa.api import MultiOmicModel
from mosa.models.mofa import MOFAModel
from mosa.models.mofa.config import MOFAConfig
from mosa.models.mosa import MOSAVAEModel

_mofapy2_available = importlib.util.find_spec("mofapy2") is not None
_mofax_available = importlib.util.find_spec("mofax") is not None
_mofa_available = _mofapy2_available and _mofax_available

skip_mofa = pytest.mark.skipif(
    not _mofa_available,
    reason="mofapy2 and/or mofax not installed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split(dataset, n_train=16):
    indices = np.arange(dataset.n_samples)
    return dataset.subset(indices[:n_train]), dataset.subset(indices[n_train:])


# ---------------------------------------------------------------------------
# MOSAVAEModel tests
# ---------------------------------------------------------------------------


def test_mosavae_fit(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val)
    assert (tmp_path / "train" / "latent.parquet").exists()


def test_mosavae_transform_shape(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val)
    z = model.transform(train)
    assert z.shape == (train.n_samples, model_cfg.joint_latent_dim)
    assert z.dtype in (np.float32, np.float64)
    assert not np.isnan(z).any()


def test_mosavae_reconstruct_shapes(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val)
    recon = model.reconstruct(train)
    assert set(recon.keys()) == set(train.view_names)
    for name in train.view_names:
        expected_dim = train.views[name].shape[1]
        assert recon[name].shape == (train.n_samples, expected_dim)


def test_mosavae_fit_no_val(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    assert (tmp_path / "train" / "latent.parquet").exists()


@pytest.mark.parametrize("fusion_method", ["concat", "poe"])
def test_mosavae_concat_and_poe(
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
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val)
    z = model.transform(train)
    assert z.shape == (train.n_samples, model_cfg.joint_latent_dim)


# ---------------------------------------------------------------------------
# MOFAModel tests
# ---------------------------------------------------------------------------


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
    assert (tmp_path / "mofa.hdf5").exists()


@skip_mofa
def test_mofa_transform_shape(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    save_path = str(tmp_path / "mofa.hdf5")
    model = MOFAModel(_mofa_data_cfg(dataset), MOFAConfig(n_factors=5), save_path=save_path)
    model.fit(train)
    z = model.transform(train)
    assert z.shape == (train.n_samples, 5)


@skip_mofa
def test_mofa_reconstruct_shapes(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    save_path = str(tmp_path / "mofa.hdf5")
    model = MOFAModel(_mofa_data_cfg(dataset), MOFAConfig(n_factors=5), save_path=save_path)
    model.fit(train)
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

    other = make_multi_omic_dataset(n_samples=20, seed=99)
    other.metadata.index = [f"other_{i:03d}" for i in range(other.n_samples)]

    with pytest.raises(NotImplementedError):
        model.transform(other)


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------


def test_all_models_are_multi_omic_model(sample_dataset, make_mosa_config, tmp_path):
    data_cfg, model_cfg = make_mosa_config(sample_dataset, output_dir=str(tmp_path))
    assert isinstance(MOSAVAEModel(data_cfg, model_cfg), MultiOmicModel)
    assert isinstance(MOFAModel(), MultiOmicModel)


# ---------------------------------------------------------------------------
# Error handling — calling API before fit()
# ---------------------------------------------------------------------------


def test_transform_before_fit_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.transform(dataset)


def test_reconstruct_before_fit_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.reconstruct(dataset)


def test_save_before_fit_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.save(tmp_path / "model.pt")


# ---------------------------------------------------------------------------
# Output quality — no NaN, deterministic eval
# ---------------------------------------------------------------------------


def test_mosavae_transform_no_nan(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z = model.transform(train)
    assert not np.isnan(z).any(), "transform() produced NaN values"
    assert not np.isinf(z).any(), "transform() produced Inf values"


def test_mosavae_reconstruct_no_nan(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20, missing_frac=0.2)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    recon = model.reconstruct(train)
    for name, arr in recon.items():
        assert not np.isnan(arr).any(), f"reconstruct() produced NaN for view '{name}'"
        assert not np.isinf(arr).any(), f"reconstruct() produced Inf for view '{name}'"


def test_mosavae_transform_deterministic(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z1 = model.transform(train)
    z2 = model.transform(train)
    np.testing.assert_array_equal(z1, z2)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_mosavae_save_creates_checkpoint(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
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


def test_mosavae_load_preserves_architecture(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    save_path = tmp_path / "model.pt"
    model.save(save_path)

    loaded = MOSAVAEModel.load(save_path)
    assert loaded._model.view_input_dims == model._model.view_input_dims
    assert loaded._model.conditional_dim == model._model.conditional_dim
    assert loaded._model.n_batches == model._model.n_batches


def test_mosavae_load_preserves_weights(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Weights survive a save/load round-trip: same batch → identical predictions."""
    from mosa.models.mosa.datamodule import MOSADataModule

    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)

    save_path = tmp_path / "model.pt"
    model.save(save_path)
    loaded = MOSAVAEModel.load(save_path)

    # Use a fresh datamodule with training scalers to get a proper dataloader
    inf_dm = MOSADataModule(train_data=train, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    inf_dm.scalers = model._datamodule.scalers
    inf_dm.setup()
    loader = inf_dm.train_eval_dataloader()

    result_orig = model._model.predict(loader)
    result_loaded = loaded._model.predict(loader)

    np.testing.assert_allclose(result_orig["z"], result_loaded["z"], atol=1e-5)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_mosavae_single_view(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20, view_dims={"rna": 40})
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z = model.transform(train)
    assert z.shape == (train.n_samples, model_cfg.joint_latent_dim)
    assert not np.isnan(z).any()


def test_mosavae_with_missing_data(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Model handles per-feature missingness without NaN in outputs."""
    dataset = make_multi_omic_dataset(n_samples=20, missing_frac=0.3)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAVAEModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z = model.transform(train)
    assert not np.isnan(z).any()
