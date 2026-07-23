from __future__ import annotations

import numpy as np
import pytest
import torch

from mosa.models.mosa import MOSAModel
from mosa.models.mosa.datamodule import MOSADataModule


def _split(dataset, n_train=16):
    indices = np.arange(dataset.n_samples)
    return dataset.subset(indices[:n_train]), dataset.subset(indices[n_train:])


# MOSADataModule state serialisation


def test_datamodule_state_dict_scaler_fidelity(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Scaler means and scales survive a state_dict / load_state_dict round-trip."""
    dataset = make_multi_omic_dataset()
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    dm = MOSADataModule(train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()

    state = dm.state_dict()

    dm2 = MOSADataModule(train_data=None, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm2.load_state_dict(state)

    assert dm2.batch_categories == dm.batch_categories
    assert dm2.feature_names == dm.feature_names
    for view in dm.scalers:
        s1, s2 = dm.scalers[view], dm2.scalers[view]
        if s1 is None:
            assert s2 is None
        else:
            np.testing.assert_allclose(s1.mean_, s2.mean_, rtol=1e-6)
            np.testing.assert_allclose(s1.scale_, s2.scale_, rtol=1e-6)


# MOSAModel save / load


def test_vae_save_load_config_preserved(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Config fields and arch dims are reconstructed exactly from the checkpoint."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)

    ckpt = tmp_path / "model.pt"
    model.save(ckpt)

    loaded = MOSAModel.load(ckpt)

    assert loaded.model_cfg.joint_latent_dim == model_cfg.joint_latent_dim
    assert loaded.model_cfg.fusion_method == model_cfg.fusion_method
    assert set(loaded.model_cfg.views.keys()) == set(model_cfg.views.keys())
    assert set(loaded.data_cfg.views) == set(data_cfg.views)
    assert loaded._model.view_input_dims == model._model.view_input_dims
    assert loaded._model.n_batches == model._model.n_batches


def test_vae_save_writes_lightning_format(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """save() produces a Lightning-shaped checkpoint with state_dict, hparams, datamodule."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)

    ckpt = tmp_path / "model.ckpt"
    model.save(ckpt)

    raw = torch.load(str(ckpt), weights_only=False)
    assert "state_dict" in raw
    assert "hyper_parameters" in raw
    assert "MOSADataModule" in raw  # Lightning keys datamodule state by class name
    assert "model_cfg" in raw["hyper_parameters"]
    assert "data_cfg" in raw["hyper_parameters"]
    assert "view_input_dims" in raw["hyper_parameters"]


def test_vae_load_reads_auto_checkpoint(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """load() can read a .ckpt written by Lightning's ModelCheckpoint callback."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)

    # ModelCheckpoint(save_last=True) writes last.ckpt into output_dir during fit()
    last_ckpt = tmp_path / "last.ckpt"
    assert last_ckpt.exists(), "Lightning should have auto-saved last.ckpt"

    loaded = MOSAModel.load(last_ckpt)
    z = loaded.transform(dataset)
    assert z.shape == (dataset.n_samples, model_cfg.joint_latent_dim)


def test_vae_save_load_scaler_fidelity(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Scaler state is preserved so loaded model can apply training-time normalisation."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)

    ckpt = tmp_path / "model.pt"
    model.save(ckpt)

    loaded = MOSAModel.load(ckpt)

    for view in model._datamodule.scalers:
        s1 = model._datamodule.scalers[view]
        s2 = loaded._datamodule.scalers[view]
        if s1 is None:
            assert s2 is None
        else:
            np.testing.assert_allclose(s1.mean_, s2.mean_, rtol=1e-6)


def test_vae_transform_after_load(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """transform() on a loaded model returns the correct shape and finite values."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)

    ckpt = tmp_path / "model.pt"
    model.save(ckpt)

    loaded = MOSAModel.load(ckpt)
    z = loaded.transform(dataset)

    assert z.shape == (dataset.n_samples, model_cfg.joint_latent_dim)
    assert not np.isnan(z).any()


# Resume training


def test_vae_resume_produces_valid_output(make_multi_omic_dataset, make_mosa_config, tmp_path):
    """Resuming from last.ckpt completes without error and transform() still works."""
    dataset = make_multi_omic_dataset(n_samples=20)
    train, val = _split(dataset)

    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path), num_epochs=2)
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val)

    last_ckpt = tmp_path / "last.ckpt"
    assert last_ckpt.exists(), "ModelCheckpoint(save_last=True) must write last.ckpt"

    data_cfg2, model_cfg2 = make_mosa_config(dataset, output_dir=str(tmp_path), num_epochs=4)
    model2 = MOSAModel(data_cfg2, model_cfg2)
    model2.fit(train, val, resume_from=last_ckpt)

    z = model2.transform(dataset)
    assert z.shape == (dataset.n_samples, model_cfg.joint_latent_dim)
    assert not np.isnan(z).any()
