"""Tests for config/data requirements validation (mosa validate + pre-train path)."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from mosa.config import Config, DataConfig, EvaluationConfig
from mosa.data.io import _dearrow_mudata, summarize_structure
from mosa.models.mofa.config import MOFAConfig
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig
from mosa.utils import validate_config_against_data

# Helper: build a custom h5mu file with controllable obs columns.


def _write_h5mu(
    tmp_path,
    view_specs: dict[str, int],
    n_samples: int = 20,
    model_types: list[str] | None = None,
    include_model_type: bool = True,
    tissue: bool = True,
    mutation_cols: list[str] | None = None,
    mask_name: str = "mask",
):
    import anndata
    import mudata

    anndata.settings.allow_write_nullable_strings = True
    rng = np.random.RandomState(42)

    adatas = {}
    for view_name, n_features in view_specs.items():
        X = rng.randn(n_samples, n_features).astype(np.float32)
        mask: np.ndarray = np.ones((n_samples, n_features), dtype=bool)
        obs = pd.DataFrame(index=[f"sample_{i:03d}" for i in range(n_samples)])
        var = pd.DataFrame(index=[f"{view_name}_feat_{j}" for j in range(n_features)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        adata.layers[mask_name] = mask
        adatas[view_name] = adata

    index = [f"sample_{i:03d}" for i in range(n_samples)]

    obs_data: dict[str, list] = {}
    if include_model_type:
        if model_types is None:
            model_types = ["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_samples)]
        obs_data["model_type"] = model_types
    if tissue:
        obs_data["tissue"] = [
            "tissue_0" if i % 2 == 0 else "tissue_1" for i in range(n_samples)
        ]
    for col in mutation_cols or []:
        obs_data[col] = [i % 2 for i in range(n_samples)]

    obs_df = pd.DataFrame(obs_data, index=index)
    obs_df.index = obs_df.index.astype(object)

    path = tmp_path / "data.h5mu"
    with mudata.set_options(pull_on_update=False):
        mdata = mudata.MuData(adatas)
        mdata.obs = obs_df.copy()
        _dearrow_mudata(mdata)
        mdata.write(str(path))
    return path


def _data_and_model_cfg(path, views, **data_overrides):
    data_cfg = DataConfig(path=str(path), views=views, **data_overrides)
    view_cfgs = {
        name: OmicViewConfig(name=name, hidden_layer_dims=[16, 8]) for name in views
    }
    model_cfg = MOSAConfig(
        views=view_cfgs, joint_latent_dim=8, batch_size=4, num_epochs=1
    )
    return data_cfg, model_cfg


# 1. Valid config + matching data: no error, no warnings.


def test_valid_config_matching_data_no_warnings(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10, "view_b": 8})
    data_cfg, model_cfg = _data_and_model_cfg(
        path, ["view_a", "view_b"], use_mutations=False
    )
    cfg = Config(data=data_cfg, model=model_cfg)

    warnings = validate_config_against_data(cfg)
    assert warnings == []


# 2. Wrong-case view name: ValueError naming available modalities.


def test_view_name_wrong_case_raises(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10})
    data_cfg, model_cfg = _data_and_model_cfg(path, ["View_A"])
    cfg = Config(data=data_cfg, model=model_cfg)

    with pytest.raises(ValueError, match="not in MuData"):
        validate_config_against_data(cfg)


# 3. mask_layer_name not present in a view: ValueError.


def test_mask_layer_missing_raises(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10}, mask_name="mask")
    data_cfg, model_cfg = _data_and_model_cfg(
        path, ["view_a"], mask_layer_name="missing_mask"
    )
    cfg = Config(data=data_cfg, model=model_cfg)

    with pytest.raises(ValueError, match="Mask layer 'missing_mask' not in"):
        validate_config_against_data(cfg)


# 4. model_type column absent: ValueError.


def test_model_type_column_absent_raises(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10}, include_model_type=False)
    data_cfg, model_cfg = _data_and_model_cfg(path, ["view_a"])
    cfg = Config(data=data_cfg, model=model_cfg)

    with pytest.raises(ValueError, match="missing 'model_type' column"):
        validate_config_against_data(cfg)


# 5. target_batch case-mismatch: ValueError; correct case passes.


def test_target_batch_wrong_case_raises(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10})
    data_cfg, model_cfg = _data_and_model_cfg(path, ["view_a"], use_mutations=False)
    model_cfg.inference = True
    model_cfg.target_batch = "typea"
    cfg = Config(data=data_cfg, model=model_cfg)

    with pytest.raises(
        ValueError, match="target_batch 'typea' not in model_type categories"
    ):
        validate_config_against_data(cfg)


def test_target_batch_correct_case_passes(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10})
    data_cfg, model_cfg = _data_and_model_cfg(path, ["view_a"], use_mutations=False)
    model_cfg.inference = True
    model_cfg.target_batch = "TypeA"
    cfg = Config(data=data_cfg, model=model_cfg)

    warnings = validate_config_against_data(cfg)
    assert warnings == []


# 6. contrastive_weight > 0 (or use_tissue) with no tissue column: warning.


def test_contrastive_without_tissue_warns(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10}, tissue=False)
    data_cfg, model_cfg = _data_and_model_cfg(
        path,
        ["view_a"],
        use_mutations=False,
        use_tissue=False,
    )
    model_cfg.contrastive_weight = 1.0
    cfg = Config(data=data_cfg, model=model_cfg)

    warnings = validate_config_against_data(cfg)
    assert any("tissue" in w for w in warnings)


def test_use_tissue_without_tissue_column_warns(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10}, tissue=False)
    data_cfg, model_cfg = _data_and_model_cfg(
        path, ["view_a"], use_mutations=False, use_tissue=True
    )
    cfg = Config(data=data_cfg, model=model_cfg)

    warnings = validate_config_against_data(cfg)
    assert any("tissue" in w for w in warnings)


# 7. use_mutations=True with no mutation_* column: warning.


def test_use_mutations_without_mutation_columns_warns(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10})
    data_cfg, model_cfg = _data_and_model_cfg(path, ["view_a"], use_mutations=True)
    cfg = Config(data=data_cfg, model=model_cfg)

    warnings = validate_config_against_data(cfg)
    assert any("mutation" in w for w in warnings)


# 8. adv_weight > 0 with a single model_type category: warning.


def test_adv_weight_single_batch_warns(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10}, model_types=["TypeA"] * 20)
    data_cfg, model_cfg = _data_and_model_cfg(path, ["view_a"], use_mutations=False)
    model_cfg.adv_weight = 1.0
    model_cfg.adv_learning_rate = 1e-3
    cfg = Config(data=data_cfg, model=model_cfg)

    warnings = validate_config_against_data(cfg)
    assert any("adv" in w.lower() for w in warnings)


# 9. summarize_structure does not load matrices; returns expected keys.


def test_summarize_structure_returns_metadata_only(tmp_path):
    path = _write_h5mu(
        tmp_path,
        {"view_a": 10, "view_b": 8},
        mutation_cols=["mutation_TP53"],
    )
    summary = summarize_structure(str(path))

    assert summary["format"] == "h5mu"
    assert set(summary["modalities"]) == {"view_a", "view_b"}
    assert summary["modalities"]["view_a"]["n_features"] == 10
    assert "mask" in summary["modalities"]["view_a"]["layers"]
    assert summary["modalities"]["view_b"]["n_features"] == 8
    assert "mask" in summary["modalities"]["view_b"]["layers"]
    assert set(summary["obs_columns"]) >= {"model_type", "tissue", "mutation_TP53"}
    assert summary["model_type_categories"] == ["TypeA", "TypeB"]


def test_summarize_structure_zarr(tmp_path):
    import anndata
    import mudata

    anndata.settings.allow_write_nullable_strings = True
    rng = np.random.RandomState(0)
    n_samples, n_features = 6, 4
    X = rng.randn(n_samples, n_features).astype(np.float32)
    mask = np.ones((n_samples, n_features), dtype=bool)
    obs = pd.DataFrame(index=[f"s{i}" for i in range(n_samples)])
    var = pd.DataFrame(index=[f"f{j}" for j in range(n_features)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["mask"] = mask
    obs_df = pd.DataFrame(
        {"model_type": ["TypeA", "TypeB"] * 3},
        index=[f"s{i}" for i in range(n_samples)],
    )
    obs_df.index = obs_df.index.astype(object)
    path = tmp_path / "data.zarr"
    with mudata.set_options(pull_on_update=False):
        mdata = mudata.MuData({"view_a": adata})
        mdata.obs = obs_df.copy()
        _dearrow_mudata(mdata)
        mdata.write_zarr(str(path))

    summary = summarize_structure(str(path))
    assert summary["format"] == "zarr"
    assert summary["modalities"]["view_a"]["n_features"] == n_features
    assert summary["modalities"]["view_a"]["layers"] == ["mask"]
    assert "model_type" in summary["obs_columns"]
    assert set(summary["model_type_categories"]) == {"TypeA", "TypeB"}


# 10. mosa validate on a config with a missing path raises a MosaError. The
#     exit code and stderr contract it gets turned into lives at the CLI
#     boundary; see tests/test_cli_errors.py.


def test_cli_validate_missing_path_raises_mosa_error(tmp_path):
    from mosa.cli import _validate
    from mosa.errors import MissingFileError

    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(
        "data:\n"
        "  path: does/not/exist.h5mu\n"
        "  views: [view_a]\n"
        "model:\n"
        "  type: mosa_vae\n"
        "  views:\n"
        "    view_a:\n"
        "      hidden_layer_dims: [16, 8]\n"
        "evaluation:\n"
        "  test_size: 0.0\n"
    )
    args = argparse.Namespace(config=str(yaml_path))

    with pytest.raises(MissingFileError, match="data.path not found"):
        _validate(args)


# 11. Existing load-time structural errors are unchanged (regression guard;
#     also exercised directly in tests/test_data.py).


def test_load_time_structure_error_unchanged(tmp_path):
    from mosa.data.io import load_mudata

    path = _write_h5mu(tmp_path, {"view_a": 10})
    with pytest.raises(ValueError, match="not in MuData"):
        load_mudata(str(path), ["view_a", "view_missing"])


# MOFAConfig.validate_against_data: held-out projection is supported.


def test_mofa_validate_against_data_returns_empty(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10})
    data_cfg = DataConfig(path=str(path), views=["view_a"])
    cfg = Config(
        data=data_cfg,
        model=MOFAConfig(),
        evaluation=EvaluationConfig(test_size=0.0),
    )

    warnings = validate_config_against_data(cfg)
    assert warnings == []


def test_mofa_allows_heldout_samples(tmp_path):
    path = _write_h5mu(tmp_path, {"view_a": 10})
    data_cfg = DataConfig(path=str(path), views=["view_a"])
    cfg = Config(
        data=data_cfg,
        model=MOFAConfig(),
        evaluation=EvaluationConfig(test_size=0.2),
    )

    warnings = validate_config_against_data(cfg)
    assert warnings == []
