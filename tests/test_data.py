from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.data.io import load_mudata, _dearrow_mudata
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig
from mosa.models.mosa.datamodule import MOSADataModule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs_df(n_samples: int, rng: np.random.RandomState) -> pd.DataFrame:
    index = [f"sample_{i:03d}" for i in range(n_samples)]
    model_types = ["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_samples)]
    tissues = ["tissue_0" if i % 3 == 0 else "tissue_1" for i in range(n_samples)]
    return pd.DataFrame({"model_type": model_types, "tissue": tissues}, index=index)


def _create_test_h5mu(path: Path, n_samples: int, view_specs: dict[str, int]) -> Path:
    import anndata
    import mudata

    anndata.settings.allow_write_nullable_strings = True

    rng = np.random.RandomState(42)
    adatas = {}
    for view_name, n_features in view_specs.items():
        X = rng.randn(n_samples, n_features).astype(np.float32)
        mask = np.ones((n_samples, n_features), dtype=bool)
        flat = mask.ravel()
        n_false = max(1, int(len(flat) * 0.2))
        false_idx = rng.choice(len(flat), size=n_false, replace=False)
        flat[false_idx] = False
        mask = flat.reshape(n_samples, n_features)
        obs = pd.DataFrame(index=[f"sample_{i:03d}" for i in range(n_samples)])
        var = pd.DataFrame(
            index=[f"{view_name}_feat_{j}" for j in range(n_features)]
        )
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers["mask"] = mask
        adatas[view_name] = adata

    mdata = mudata.MuData(adatas)
    obs_df = _make_obs_df(n_samples, rng)
    obs_df.index = obs_df.index.astype(object)
    mdata.obs = obs_df.copy()
    _dearrow_mudata(mdata)
    mdata.write(str(path))
    return path


def _create_test_zarr(path: Path, n_samples: int, view_specs: dict[str, int]) -> Path:
    import anndata
    import mudata

    anndata.settings.allow_write_nullable_strings = True

    rng = np.random.RandomState(42)
    adatas = {}
    for view_name, n_features in view_specs.items():
        X = rng.randn(n_samples, n_features).astype(np.float32)
        mask = np.ones((n_samples, n_features), dtype=bool)
        flat = mask.ravel()
        n_false = max(1, int(len(flat) * 0.2))
        false_idx = rng.choice(len(flat), size=n_false, replace=False)
        flat[false_idx] = False
        mask = flat.reshape(n_samples, n_features)
        obs = pd.DataFrame(index=[f"sample_{i:03d}" for i in range(n_samples)])
        var = pd.DataFrame(
            index=[f"{view_name}_feat_{j}" for j in range(n_features)]
        )
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers["mask"] = mask
        adatas[view_name] = adata

    mdata = mudata.MuData(adatas)
    obs_df = _make_obs_df(n_samples, rng)
    obs_df.index = obs_df.index.astype(object)
    mdata.obs = obs_df.copy()
    _dearrow_mudata(mdata)
    mdata.write_zarr(str(path))
    return path


def _make_configs(view_specs: dict[str, int], n_samples: int, tmp_path: Path, discrete_views: set[str] | None = None) -> tuple[DataConfig, MOSAConfig]:
    discrete_views = discrete_views or set()
    data_cfg = DataConfig(
        path=str(tmp_path / "dummy.h5mu"),
        views=list(view_specs.keys()),
        discrete_views=discrete_views,
    )
    view_configs = {
        name: OmicViewConfig(name=name, hidden_layer_dims=[32, 16])
        for name in view_specs
    }
    model_cfg = MOSAConfig(
        views=view_configs,
        joint_latent_dim=8,
        batch_size=8,
        num_epochs=1,
        output_dir=str(tmp_path / "out"),
        weighted_random_sampler=False,
    )
    return data_cfg, model_cfg


def _make_dataset(
    n_samples: int,
    view_specs: dict[str, int],
    seed: int = 42,
    model_types: list[str] | None = None,
) -> MultiOmicDataset:
    rng = np.random.RandomState(seed)
    views = {
        name: rng.randn(n_samples, n_features).astype(np.float32)
        for name, n_features in view_specs.items()
    }
    masks = {
        name: np.ones((n_samples, n_features), dtype=bool)
        for name, n_features in view_specs.items()
    }
    feature_names = {
        name: [f"{name}_feat_{j}" for j in range(n_features)]
        for name, n_features in view_specs.items()
    }
    if model_types is None:
        model_types = ["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_samples)]
    index = [f"sample_{i:03d}" for i in range(n_samples)]
    metadata = pd.DataFrame(
        {
            "model_type": model_types,
            "tissue": ["tissue_0" if i % 2 == 0 else "tissue_1" for i in range(n_samples)],
        },
        index=index,
    )
    return MultiOmicDataset(views=views, masks=masks, metadata=metadata, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Tests for load_mudata() — h5mu
# ---------------------------------------------------------------------------

def test_load_h5mu_basic(tmp_path):
    view_specs = {"view_a": 20, "view_b": 15}
    p = _create_test_h5mu(tmp_path / "test.h5mu", 20, view_specs)
    dataset = load_mudata(str(p), ["view_a", "view_b"])

    assert isinstance(dataset, MultiOmicDataset)
    assert dataset.n_samples == 20
    assert dataset.view_names == ["view_a", "view_b"]
    assert dataset.views["view_a"].shape == (20, 20)
    assert dataset.views["view_b"].shape == (20, 15)
    assert dataset.masks["view_a"].shape == (20, 20)
    assert dataset.masks["view_b"].shape == (20, 15)
    assert "model_type" in dataset.metadata.columns


def test_load_h5mu_missing_view_raises(tmp_path):
    p = _create_test_h5mu(tmp_path / "test.h5mu", 10, {"view_a": 10})
    with pytest.raises(ValueError, match="not in MuData"):
        load_mudata(str(p), ["view_a", "view_missing"])


def test_load_h5mu_sparse_data(tmp_path):
    import anndata
    import mudata
    from scipy.sparse import csr_matrix

    anndata.settings.allow_write_nullable_strings = True
    rng = np.random.RandomState(42)
    n_samples, n_features = 10, 8
    X = csr_matrix(rng.randn(n_samples, n_features).astype(np.float32))
    mask = np.ones((n_samples, n_features), dtype=bool)
    obs = pd.DataFrame(index=[f"sample_{i:03d}" for i in range(n_samples)])
    var = pd.DataFrame(index=[f"feat_{j}" for j in range(n_features)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["mask"] = mask
    mdata = mudata.MuData({"view_a": adata})
    obs_df = _make_obs_df(n_samples, rng)
    obs_df.index = obs_df.index.astype(object)
    mdata.obs = obs_df.copy()
    _dearrow_mudata(mdata)
    p = tmp_path / "sparse.h5mu"
    mdata.write(str(p))

    dataset = load_mudata(str(p), ["view_a"])
    assert isinstance(dataset.views["view_a"], np.ndarray)
    assert dataset.views["view_a"].dtype == np.float32
    assert dataset.views["view_a"].shape == (n_samples, n_features)


# ---------------------------------------------------------------------------
# Tests for load_mudata() — zarr
# ---------------------------------------------------------------------------

def test_load_zarr_basic(tmp_path):
    view_specs = {"view_a": 20, "view_b": 15}
    p = _create_test_zarr(tmp_path / "test.zarr", 20, view_specs)
    dataset = load_mudata(str(p), ["view_a", "view_b"])

    assert isinstance(dataset, MultiOmicDataset)
    assert dataset.n_samples == 20
    assert dataset.view_names == ["view_a", "view_b"]
    assert dataset.views["view_a"].shape == (20, 20)
    assert dataset.views["view_b"].shape == (20, 15)
    assert dataset.masks["view_a"].shape == (20, 20)
    assert dataset.masks["view_b"].shape == (20, 15)
    assert "model_type" in dataset.metadata.columns


def test_load_zarr_matches_h5mu(tmp_path):
    view_specs = {"view_a": 12, "view_b": 8}
    n_samples = 15
    h5mu_path = _create_test_h5mu(tmp_path / "test.h5mu", n_samples, view_specs)
    zarr_path = _create_test_zarr(tmp_path / "test.zarr", n_samples, view_specs)

    ds_h5 = load_mudata(str(h5mu_path), list(view_specs.keys()))
    ds_zarr = load_mudata(str(zarr_path), list(view_specs.keys()))

    for view_name in view_specs:
        assert ds_h5.feature_names[view_name] == ds_zarr.feature_names[view_name]
        assert np.allclose(ds_h5.views[view_name], ds_zarr.views[view_name])
        assert np.array_equal(ds_h5.masks[view_name], ds_zarr.masks[view_name])


# ---------------------------------------------------------------------------
# Tests for MultiOmicDataset.validate()
# ---------------------------------------------------------------------------

def test_validate_passes():
    dataset = _make_dataset(10, {"view_a": 5})
    dataset.validate()


def test_validate_mismatched_keys():
    dataset = _make_dataset(10, {"view_a": 5, "view_b": 3})
    del dataset.masks["view_b"]
    with pytest.raises(ValueError, match="different keys"):
        dataset.validate()


def test_validate_wrong_n_samples():
    dataset = _make_dataset(10, {"view_a": 5})
    dataset.views["view_a"] = dataset.views["view_a"][:8]
    dataset.masks["view_a"] = dataset.masks["view_a"][:8]
    with pytest.raises(ValueError, match="samples but metadata has"):
        dataset.validate()


def test_validate_missing_model_type():
    dataset = _make_dataset(10, {"view_a": 5})
    dataset.metadata = dataset.metadata.drop(columns=["model_type"])
    with pytest.raises(ValueError, match="model_type"):
        dataset.validate()


# ---------------------------------------------------------------------------
# Tests for MultiOmicDataset.subset()
# ---------------------------------------------------------------------------

def test_subset_correct_size():
    dataset = _make_dataset(10, {"view_a": 5, "view_b": 3})
    sub = dataset.subset(np.array([0, 2, 4]))
    assert sub.n_samples == 3


def test_subset_preserves_features():
    dataset = _make_dataset(10, {"view_a": 5, "view_b": 3})
    sub = dataset.subset(np.array([0, 2, 4]))
    assert sub.feature_names is dataset.feature_names


def test_subset_metadata_matches():
    dataset = _make_dataset(10, {"view_a": 5})
    indices = np.array([1, 3, 7])
    sub = dataset.subset(indices)
    expected_index = dataset.metadata.iloc[indices].index
    assert list(sub.metadata.index) == list(expected_index)


# ---------------------------------------------------------------------------
# Tests for MOSADataModule
# ---------------------------------------------------------------------------

def _make_datamodule(
    train_data: MultiOmicDataset,
    val_data: MultiOmicDataset,
    tmp_path: Path,
    view_specs: dict[str, int],
    discrete_views: set[str] | None = None,
) -> MOSADataModule:
    data_cfg, model_cfg = _make_configs(view_specs, train_data.n_samples, tmp_path, discrete_views)
    return MOSADataModule(
        train_data=train_data, val_data=val_data,
        data_cfg=data_cfg, model_cfg=model_cfg,
    )


def test_datamodule_scaler_train_only(tmp_path):
    view_specs = {"view_a": 20}
    rng = np.random.RandomState(0)

    # Train: N(0, 1), Val: N(10, 1) — deliberately different means
    n_train, n_features = 30, 20
    train_X = rng.randn(n_train, n_features).astype(np.float32)
    train_model_types = ["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_train)]
    train_data = _make_dataset(n_train, view_specs, model_types=train_model_types)
    train_data.views["view_a"] = train_X

    n_val = 10
    val_X = (rng.randn(n_val, n_features) + 10.0).astype(np.float32)
    val_model_types = ["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_val)]
    val_data = _make_dataset(n_val, view_specs, model_types=val_model_types)
    val_data.views["view_a"] = val_X

    dm = _make_datamodule(train_data, val_data, tmp_path, view_specs)
    dm.setup()

    # Scaler was fitted on train
    scaler = dm.scalers["view_a"]
    assert scaler is not None
    assert np.allclose(scaler.mean_, train_X.mean(axis=0), atol=1e-4)

    # Train data is standardized (mean ~0 per feature, overall)
    train_tensor = dm.train_dataset.omics["view_a"].numpy()
    assert abs(train_tensor.mean()) < 0.5


def test_datamodule_discrete_view_no_scaling(tmp_path):
    view_specs = {"view_a": 10, "view_b": 8}
    n = 20
    train_data = _make_dataset(n, view_specs)
    val_data = _make_dataset(10, view_specs, seed=99)

    dm = _make_datamodule(train_data, val_data, tmp_path, view_specs, discrete_views={"view_a"})
    dm.setup()

    assert dm.scalers["view_a"] is None
    assert dm.scalers["view_b"] is not None


def test_datamodule_class_weights(tmp_path):
    view_specs = {"view_a": 10}
    n_total = 20
    # 15 TypeA, 5 TypeB
    model_types = ["TypeA"] * 15 + ["TypeB"] * 5
    train_data = _make_dataset(n_total, view_specs, model_types=model_types)

    data_cfg, model_cfg = _make_configs(view_specs, n_total, tmp_path)
    dm = MOSADataModule(train_data=train_data, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()

    # sorted categories: TypeA=0, TypeB=1
    weights = dm.class_weights
    assert weights[1] > weights[0], f"TypeB weight {weights[1]} should exceed TypeA weight {weights[0]}"


def test_datamodule_batch_categories(tmp_path):
    view_specs = {"view_a": 10}
    model_types = ["TypeB", "TypeA", "TypeC", "TypeA", "TypeB", "TypeC"] * 4
    n = len(model_types)
    train_data = _make_dataset(n, view_specs, model_types=model_types)

    data_cfg, model_cfg = _make_configs(view_specs, n, tmp_path)
    dm = MOSADataModule(train_data=train_data, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg)
    dm.setup()

    expected = sorted(set(model_types))
    assert dm.batch_categories == expected
