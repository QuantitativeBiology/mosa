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


def test_transform_unseen_model_type_raises(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
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


def test_transform_rejects_reordered_features(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    from mosa.errors import DataError

    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(dataset)

    reordered = dataset.subset(np.arange(dataset.n_samples))
    reordered.feature_names["view_a"] = reordered.feature_names["view_a"][::-1]
    with pytest.raises(DataError, match="view_a"):
        model.transform(reordered)
    with pytest.raises(DataError, match="view_a"):
        model.reconstruct(reordered)


def test_save_rejects_a_suffix_load_model_cannot_route(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(dataset)
    with pytest.raises(ValueError, match=r"\.ckpt"):
        model.save(tmp_path / "model.bin")


def test_legacy_torch_checkpoint_is_still_owned(tmp_path):
    """Non-zip torch files cannot be memory-mapped; the probe must read them anyway."""
    import torch

    path = tmp_path / "legacy.ckpt"
    hp = {"view_input_dims": {"view_a": 3}}
    torch.save({"hyper_parameters": hp}, path, _use_new_zipfile_serialization=False)
    assert MOSAModel.owns_checkpoint(path)


def test_unnamed_checkpoint_of_another_model_is_not_owned(tmp_path):
    """Another Lightning model's auto-checkpoint has no name and no VAE dims."""
    import torch

    path = tmp_path / "last.ckpt"
    torch.save({"hyper_parameters": {"hidden": 8}}, path)
    assert not MOSAModel.owns_checkpoint(path)


# MOFAModel tests


def _mofa_data_cfg(dataset):
    from mosa.config import DataConfig

    return DataConfig(path="unused", views=list(dataset.view_names))


def _fit_mofa(dataset, output_dir, **cfg):
    model = MOFAModel(
        _mofa_data_cfg(dataset),
        MOFAConfig(n_factors=5, output_dir=str(output_dir), **cfg),
    )
    model.fit(dataset)
    return model


def _planted_dataset(n_samples=40):
    """Low-rank views with per-group offsets, a few masked entries, and views and
    groups on different scales, so a missing offset or scale shows in R²."""
    import pandas as pd

    from mosa.data.dataset import MultiOmicDataset

    rng = np.random.default_rng(0)
    samples = [f"s{i:02d}" for i in range(n_samples)]
    groups = np.array(["TypeA", "TypeB"] * (n_samples // 2))
    z = rng.normal(size=(n_samples, 2)) * np.where(groups == "TypeA", 1.0, 3.0)[:, None]
    views, masks, feature_names = {}, {}, {}
    for name, dim, scale in (("view_a", 20, 1.0), ("view_b", 16, 20.0)):
        offsets = {g: rng.normal(size=dim) * 2 for g in ("TypeA", "TypeB")}
        x = z @ rng.normal(size=(dim, 2)).T + np.stack([offsets[g] for g in groups])
        x = scale * (x + 0.05 * rng.normal(size=x.shape))
        mask = np.ones_like(x, dtype=bool)
        mask[3, :4] = False
        x[~mask] = 1e6
        views[name] = x.astype(np.float32)
        masks[name] = mask
        feature_names[name] = [f"{name}_feat_{j}" for j in range(dim)]
    metadata = pd.DataFrame({"model_type": groups}, index=samples)
    return MultiOmicDataset(
        views=views, masks=masks, feature_names=feature_names, metadata=metadata
    )


@skip_mofa
def test_mofa_fit(make_multi_omic_dataset, tmp_path, monkeypatch):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "out"

    model = _fit_mofa(train, out)
    assert not out.exists()
    model.save_outputs()

    assert (out / "mofa_model.hdf5").exists()
    assert (out / "full" / "latent.parquet").exists()
    for name in train.view_names:
        assert (out / "full" / f"recon_{name}.parquet").exists()
    assert list(cwd.iterdir()) == []


@skip_mofa
def test_mofa_save_outputs_explicit_dir_wins(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    model = _fit_mofa(dataset, tmp_path / "configured")
    model.save_outputs(tmp_path / "explicit")
    assert (tmp_path / "explicit" / "mofa_model.hdf5").exists()
    assert not (tmp_path / "configured").exists()


@skip_mofa
def test_mofa_save_outputs_twice_same_dir(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()
    z = model.transform(dataset)
    model.save_outputs()
    np.testing.assert_array_equal(model.transform(dataset), z)


@skip_mofa
def test_mofa_outputs_follow_input_order_and_names(tmp_path):
    import pandas as pd

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()

    latent = pd.read_parquet(tmp_path / "full" / "latent.parquet")
    assert list(latent.index) == dataset.sample_names
    for name in dataset.view_names:
        recon = pd.read_parquet(tmp_path / "full" / f"recon_{name}.parquet")
        assert list(recon.index) == dataset.sample_names
        assert list(recon.columns) == dataset.feature_names[name]


@skip_mofa
def test_mofa_transform_shape(make_multi_omic_dataset, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    model = _fit_mofa(train, tmp_path)
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
    model = _fit_mofa(train, tmp_path)
    model.save_outputs()
    recon = model.reconstruct(train)
    assert set(recon.keys()) == set(train.view_names)
    for name in train.view_names:
        expected_dim = train.views[name].shape[1]
        assert recon[name].shape == (train.n_samples, expected_dim)


_SCALINGS = [
    {},
    {"scale_views": True},
    {"scale_groups": True},
    {"scale_views": True, "scale_groups": True},
]


@skip_mofa
@pytest.mark.parametrize("scaling", _SCALINGS, ids=lambda d: "+".join(d) or "none")
def test_mofa_reconstruct_original_scale(tmp_path, scaling):
    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path, **scaling)
    model.save_outputs()
    recon = model.reconstruct(dataset)
    for name in dataset.view_names:
        x, mask = dataset.views[name], dataset.masks[name]
        resid = ((recon[name] - x)[mask] ** 2).sum()
        total = ((x[mask] - x[mask].mean()) ** 2).sum()
        assert 1 - resid / total > 0.99
        assert (tmp_path / "full" / f"recon_{name}.parquet").exists()


@skip_mofa
@pytest.mark.parametrize("scaling", _SCALINGS, ids=lambda d: "+".join(d) or "none")
def test_mofa_stored_scales_reproduce_processed_data(tmp_path, scaling):
    """Intercepts and scales in the model file invert mofapy2's preprocessing exactly."""
    import h5py

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path, **scaling)
    model.save_outputs()

    with h5py.File(tmp_path / "mofa_model.hdf5", "r") as f:
        for name in dataset.view_names:
            for group in f["samples"]:
                rows = [
                    dataset.sample_names.index(s.decode())
                    for s in f["samples"][group][:]
                ]
                x = dataset.views[name][rows].astype(float)
                x[~dataset.masks[name][rows]] = np.nan
                expected = (
                    (x - f["intercepts"][name][group][:])
                    / f["mosa/view_scale"][name][()]
                    / f["mosa/group_scale"][name][group][()]
                )
                np.testing.assert_allclose(
                    f["data"][name][group][:], expected, rtol=1e-6, equal_nan=True
                )


@skip_mofa
def test_mofa_binary_view_is_not_reconstructed(tmp_path):
    from mosa.errors import UnsupportedError

    dataset = _planted_dataset()
    binary = dataset.views["view_b"] > np.median(dataset.views["view_b"])
    dataset.views["view_b"] = binary.astype(np.float32)
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()

    assert (tmp_path / "full" / "recon_view_a.parquet").exists()
    assert not (tmp_path / "full" / "recon_view_b.parquet").exists()
    with pytest.raises(UnsupportedError, match="bernoulli"):
        model.reconstruct(dataset)


@skip_mofa
def test_mofa_skipped_view_leaves_no_stale_recon(tmp_path):
    _fit_mofa(_planted_dataset(), tmp_path).save_outputs()
    assert (tmp_path / "full" / "recon_view_b.parquet").exists()

    dataset = _planted_dataset()
    binary = dataset.views["view_b"] > np.median(dataset.views["view_b"])
    dataset.views["view_b"] = binary.astype(np.float32)
    _fit_mofa(dataset, tmp_path).save_outputs()
    assert not (tmp_path / "full" / "recon_view_b.parquet").exists()


def _renamed(dataset, views=None, samples=None, features=None):
    """Copy of dataset with views, sample IDs or per-view feature names replaced."""
    from mosa.data.dataset import MultiOmicDataset

    views = views or {v: v for v in dataset.view_names}
    metadata = dataset.metadata.copy()
    if samples is not None:
        metadata.index = samples
    features = features or {}
    return MultiOmicDataset(
        views={new: dataset.views[old] for old, new in views.items()},
        masks={new: dataset.masks[old] for old, new in views.items()},
        feature_names={
            new: features.get(new, list(dataset.feature_names[old]))
            for old, new in views.items()
        },
        metadata=metadata,
    )


@skip_mofa
@pytest.mark.parametrize("kind", ["sample", "feature"])
def test_mofa_rejects_non_ascii_names_before_training(tmp_path, kind):
    """mofax reads names back as ASCII, so such a model could not be reopened."""
    from mosa.errors import DataError

    dataset = _planted_dataset()
    if kind == "sample":
        samples = [f"échantillon_{i:02d}" for i in range(dataset.n_samples)]
        dataset = _renamed(dataset, samples=samples)
    else:
        names = [f"gène_{j}" for j in range(len(dataset.feature_names["view_a"]))]
        dataset = _renamed(dataset, features={"view_a": names})
    with pytest.raises(DataError, match="ASCII"):
        _fit_mofa(dataset, tmp_path)
    assert not (tmp_path / "mofa_model.hdf5").exists()


@skip_mofa
def test_mofa_feature_names_colliding_with_view_suffixes(tmp_path):
    """View a_b with feature x and view b with feature x_a both read as x_a_b
    under a name_view suffix."""
    dataset = _planted_dataset()
    dataset = _renamed(
        dataset,
        views={"view_a": "a_b", "view_b": "b"},
        features={
            "a_b": ["x"] + [f"a{j}" for j in range(19)],
            "b": ["x_a"] + [f"b{j}" for j in range(15)],
        },
    )
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()
    assert model.transform(dataset).shape[0] == dataset.n_samples


@skip_mofa
def test_mofa_transform_rejects_renamed_features(tmp_path):
    from mosa.errors import DataError

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()

    names = [f"renamed_{j}" for j in range(len(dataset.feature_names["view_a"]))]
    renamed = _renamed(dataset, features={"view_a": names})
    with pytest.raises(DataError, match="view_a"):
        model.transform(renamed)


@skip_mofa
def test_mofa_dropped_view_leaves_no_stale_recon(tmp_path):
    _fit_mofa(_planted_dataset(), tmp_path).save_outputs()
    assert (tmp_path / "full" / "recon_view_b.parquet").exists()

    only_a = _renamed(_planted_dataset(), views={"view_a": "view_a"})
    _fit_mofa(only_a, tmp_path).save_outputs()
    assert (tmp_path / "full" / "recon_view_a.parquet").exists()
    assert not (tmp_path / "full" / "recon_view_b.parquet").exists()


@skip_mofa
def test_mofa_save_rejects_a_suffix_load_model_cannot_route(tmp_path):
    model = _fit_mofa(_planted_dataset(), tmp_path)
    model.save_outputs()
    with pytest.raises(ValueError, match=r"\.hdf5"):
        model.save(tmp_path / "model.bin")


@skip_mofa
def test_mofa_values_changed_between_fit_and_save_are_rejected(tmp_path):
    from mosa.errors import DataError

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    dataset.views["view_a"][2, 0] += 1.0
    model.save_outputs()
    with pytest.raises(DataError, match=r"s02"):
        model.transform(dataset)


@skip_mofa
def test_mofa_r_missing_sentinel_does_not_skew_scales(tmp_path):
    """mofapy2 reads -2147483648 as missing; the stored scales must agree."""
    dataset = _planted_dataset()
    sentinel = (5, 3)
    dataset.views["view_b"][sentinel] = -2147483648
    model = _fit_mofa(dataset, tmp_path, scale_views=True, scale_groups=True)
    model.save_outputs()

    x, observed = dataset.views["view_b"], dataset.masks["view_b"].copy()
    observed[sentinel] = False
    recon = model.reconstruct(dataset)["view_b"]
    resid = ((recon - x)[observed] ** 2).sum()
    total = ((x[observed] - x[observed].mean()) ** 2).sum()
    assert 1 - resid / total > 0.99


@skip_mofa
def test_mofa_reconstruct_refuses_files_without_mosa_metadata(tmp_path):
    """A plain mofapy2 file does not say how the data was centered or scaled."""
    import shutil

    import h5py

    from mosa.errors import UnsupportedError

    dataset = _planted_dataset()
    _fit_mofa(dataset, tmp_path / "run").save_outputs()
    plain = tmp_path / "plain.hdf5"
    shutil.copy(tmp_path / "run" / "mofa_model.hdf5", plain)
    with h5py.File(plain, "a") as f:
        del f["mosa"]

    loaded = MOFAModel.load(plain)
    assert loaded.transform(dataset).shape[0] == dataset.n_samples
    with pytest.raises(UnsupportedError, match="preprocessed"):
        loaded.reconstruct(dataset)


@skip_mofa
@pytest.mark.parametrize("labels", [[0, 1], [1]], ids=["two", "single"])
def test_mofa_integer_model_types(tmp_path, labels):
    dataset = _planted_dataset()
    dataset.metadata["model_type"] = [
        labels[i % len(labels)] for i in range(dataset.n_samples)
    ]
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()
    assert (tmp_path / "full" / "recon_view_a.parquet").exists()
    assert model.reconstruct(dataset)["view_a"].shape == dataset.views["view_a"].shape


@skip_mofa
def test_mofa_rejects_labels_that_merge_as_text(tmp_path):
    from mosa.errors import DataError

    dataset = _planted_dataset()
    dataset.metadata["model_type"] = [
        1 if i % 2 else "1" for i in range(dataset.n_samples)
    ]
    with pytest.raises(DataError, match="same group"):
        _fit_mofa(dataset, tmp_path)


@skip_mofa
def test_mofa_load_restores_the_data_config(tmp_path):
    from mosa.config import DataConfig

    dataset = _planted_dataset()
    data_cfg = DataConfig(
        path="data.h5mu",
        views=list(dataset.view_names),
        mask_layer_name="observed",
        use_tissue=False,
    )
    model = MOFAModel(data_cfg, MOFAConfig(n_factors=5, output_dir=str(tmp_path)))
    model.fit(dataset)
    model.save_outputs()
    assert MOFAModel.load(tmp_path / "mofa_model.hdf5").data_cfg == data_cfg


@skip_mofa
def test_mofa_transform_requires_every_trained_view(tmp_path):
    from mosa.errors import DataError

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()
    only_a = _renamed(dataset, views={"view_a": "view_a"})
    with pytest.raises(DataError, match="view_b"):
        model.transform(only_a)


@skip_mofa
def test_mofa_file_saved_without_data_is_owned(tmp_path):
    """mofapy2's save_data=False omits intercepts; the file is still MOFA's."""
    import h5py

    from mosa.models.registry import load_model

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    path = tmp_path / "no_data.hdf5"
    model._ent.save(str(path), save_data=False)
    with h5py.File(path, "r") as f:
        assert "intercepts" not in f
    loaded = load_model(path)
    assert isinstance(loaded, MOFAModel)
    assert loaded.transform(dataset).shape[0] == dataset.n_samples


@skip_mofa
def test_mofa_reconstruct_rejects_reordered_features(tmp_path):
    from mosa.errors import DataError

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()

    dataset.feature_names["view_a"] = dataset.feature_names["view_a"][::-1]
    with pytest.raises(DataError, match="view_a"):
        model.reconstruct(dataset)


@skip_mofa
def test_mofa_reconstruct_after_load_matches(tmp_path):
    from mosa.models.registry import load_model

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path / "run")
    model.save_outputs()
    model.save(tmp_path / "copy.hdf5")

    loaded = load_model(tmp_path / "copy.hdf5")
    expected = model.reconstruct(dataset)
    for name, arr in loaded.reconstruct(dataset).items():
        np.testing.assert_allclose(arr, expected[name])


@skip_mofa
def test_mofa_all_factors_dropped_raises_data_error(
    make_multi_omic_dataset, tmp_path, monkeypatch
):
    """mofapy2 calls exit() when every factor is dropped; fit must raise instead."""
    from mofapy2.core.BayesNet import BayesNet

    from mosa.errors import DataError

    def drop_all(self, **kwargs):
        self.dim["K"] = 0
        exit()

    monkeypatch.setattr(BayesNet, "removeInactiveFactors", drop_all)
    with pytest.raises(DataError, match="drop_r2"):
        _fit_mofa(make_multi_omic_dataset(n_samples=20), tmp_path)


@skip_mofa
def test_mofa_iterations_caps_training(make_multi_omic_dataset, tmp_path, caplog):
    import h5py

    model = _fit_mofa(make_multi_omic_dataset(n_samples=20), tmp_path, iterations=3)
    model.save_outputs()
    with h5py.File(tmp_path / "mofa_model.hdf5", "r") as f:
        # time[0] is the initialisation; each later entry is one update.
        updates = np.count_nonzero(~np.isnan(f["training_stats"]["time"][:])) - 1
    assert updates == 3
    assert "iteration cap (3)" in caplog.text


@skip_mofa
def test_mofa_interrupt_is_not_swallowed(
    make_multi_omic_dataset, tmp_path, monkeypatch
):
    from mofapy2.core.BayesNet import BayesNet

    def interrupted(self, **kwargs):
        raise KeyboardInterrupt

    # Before the training loop, mofapy2's run() wrapper turns Ctrl-C into
    # exit(); inside the loop, BayesNet swallows it. Both must surface.
    for method in ("precompute", "removeInactiveFactors"):
        with monkeypatch.context() as m:
            m.setattr(BayesNet, method, interrupted)
            with pytest.raises(KeyboardInterrupt):
                _fit_mofa(make_multi_omic_dataset(n_samples=20), tmp_path)


@skip_mofa
def test_mofa_unseen_data_raises(make_multi_omic_dataset, tmp_path):
    from mosa.errors import UnsupportedError

    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    model = _fit_mofa(train, tmp_path)
    model.save_outputs()

    other = make_multi_omic_dataset(n_samples=20, seed=99)
    other.metadata.index = [f"other_{i:03d}" for i in range(other.n_samples)]

    with pytest.raises(UnsupportedError, match="out-of-sample"):
        model.transform(other)


@skip_mofa
def test_mofa_transform_rejects_changed_values(tmp_path):
    from mosa.errors import DataError

    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()

    first = dataset.subset(np.arange(10))
    model.transform(first)
    first.views["view_a"][2, 0] += 1.0
    with pytest.raises(DataError, match=r"s02"):
        model.transform(first)


@skip_mofa
def test_mofa_transform_ignores_masked_values(tmp_path):
    """A masked entry carries no information, so changing it is not a new sample."""
    dataset = _planted_dataset()
    model = _fit_mofa(dataset, tmp_path)
    model.save_outputs()

    loaded = MOFAModel.load(tmp_path / "mofa_model.hdf5")
    dataset.views["view_a"][3, 0] = -123.0
    np.testing.assert_array_equal(loaded.transform(dataset), model.transform(dataset))


# Interface compliance


def test_all_models_are_multi_omic_model(sample_dataset, make_mosa_config, tmp_path):
    data_cfg, model_cfg = make_mosa_config(sample_dataset, output_dir=str(tmp_path))
    assert isinstance(MOSAModel(data_cfg, model_cfg), MultiOmicModel)
    assert isinstance(MOFAModel(), MultiOmicModel)


# Error handling: calling API before fit()


def test_transform_before_fit_raises(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    with pytest.raises(RuntimeError, match="fit"):
        model.transform(dataset)


def test_reconstruct_before_fit_raises(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
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


def test_vae_transform_deterministic(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20)
    train, _ = _split(dataset)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))
    model = MOSAModel(data_cfg, model_cfg)
    model.fit(train, val=None)
    z1 = model.transform(train)
    z2 = model.transform(train)
    np.testing.assert_array_equal(z1, z2)


# Save / load


def test_vae_save_creates_checkpoint(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
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


def test_vae_load_preserves_architecture(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
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


def test_vae_load_preserves_weights(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
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
    inf_dm = MOSADataModule(
        train_data=train, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg
    )
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
    return MultiOmicDataset(
        shifted_views, data.masks, data.metadata, data.feature_names
    )


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
        name: scaler["mean"].copy()
        for name, scaler in model._datamodule.scalers.items()
        if scaler is not None
    }

    shifted = _shift_dataset(train, shift=1000.0)
    model.transform(shifted)
    model.reconstruct(shifted)

    for name, mean_before in means_before.items():
        np.testing.assert_array_equal(
            model._datamodule.scalers[name]["mean"], mean_before
        )


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
    rescaled_views = {name: X * 200.0 + 5000.0 for name, X in dataset.views.items()}
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
