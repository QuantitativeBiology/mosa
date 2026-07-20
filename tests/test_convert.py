"""Tests for csv_to_mudata() and its helpers (src/mosa/data/io.py).

Grouped so the future multi-format conversion refactor can reuse Group A:
  - Group A: format-agnostic contract (output structure, alignment, NaN
    handling, mutations, round-trip, h5mu/zarr parity).
  - Group B: CSV-adapter specifics (conditionals parsing, view orientation,
    numeric validation, missing files, format-extension warning).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mosa.data.dataset import MultiOmicDataset
from mosa.data.io import csv_to_mudata, load_mudata


# ---------------------------------------------------------------------------
# CSV-writing helpers
# ---------------------------------------------------------------------------

def _write_view_csv(path, features, samples, nan_cells=None):
    """Write a features x samples view CSV (rows=features, cols=sample IDs)."""
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.randn(len(features), len(samples)), index=features, columns=samples)
    if nan_cells:
        for feat, sample in nan_cells:
            df.loc[feat, sample] = np.nan
    df.to_csv(path)
    return path


def _write_conditionals_csv(path, model_ids, model_types, tissues=None):
    data = {"model_id": model_ids, "model_type": model_types}
    if tissues is not None:
        data["tissue"] = tissues
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def _write_mutations_csv(path, features, samples, values):
    pd.DataFrame(values, index=features, columns=samples).to_csv(path)
    return path


# ===========================================================================
# Group A -- format-agnostic contract
# ===========================================================================

def test_output_structure(tmp_path):
    samples = [f"S{i:02d}" for i in range(6)]
    features_a = ["gA_0", "gA_1", "gA_2"]
    features_b = ["gB_0", "gB_1"]

    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv",
        model_ids=list(reversed(samples)),
        model_types=["TypeA", "TypeB"] * 3,
        tissues=["tissueX", "tissueY"] * 3,
    )
    view_a_path = _write_view_csv(tmp_path / "view_a.csv", features_a, samples)
    view_b_path = _write_view_csv(tmp_path / "view_b.csv", features_b, samples)

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path),
        [("view_a", str(view_a_path)), ("view_b", str(view_b_path))],
        str(out_path),
        format="h5mu",
    )

    import mudata
    mdata = mudata.read(str(out_path))

    assert set(mdata.mod.keys()) == {"view_a", "view_b"}

    adata_a = mdata.mod["view_a"]
    assert adata_a.X.dtype == np.float32
    assert list(adata_a.var_names) == features_a
    assert np.array_equal(adata_a.layers["mask"], ~np.isnan(adata_a.X))

    assert "model_type" in mdata.obs.columns
    assert "tissue" in mdata.obs.columns


def test_sample_alignment_and_ordering(tmp_path):
    cond_ids = ["S03", "S01", "S04", "S00", "S02"]  # scrambled order in the CSV
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=cond_ids, model_types=["TypeA"] * 5
    )
    features_a = ["gA_0", "gA_1"]
    features_b = ["gB_0"]
    view_a_samples = ["S00", "S01", "S02", "S03", "S04"]
    view_b_samples = ["S01", "S02", "S03", "S04", "S05"]  # missing S00; extra S05 not in conditionals

    view_a_path = _write_view_csv(tmp_path / "view_a.csv", features_a, view_a_samples)
    view_b_path = _write_view_csv(tmp_path / "view_b.csv", features_b, view_b_samples)

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path),
        [("view_a", str(view_a_path)), ("view_b", str(view_b_path))],
        str(out_path),
    )

    import mudata
    mdata = mudata.read(str(out_path))

    expected = ["S00", "S01", "S02", "S03", "S04"]  # sorted intersection; S05 excluded
    assert list(mdata.obs_names) == expected

    # S00: present in view_a, absent from view_b -> NaN row + mask False (absent).
    idx_s00 = expected.index("S00")
    adata_b = mdata.mod["view_b"]
    assert np.all(np.isnan(adata_b.X[idx_s00]))
    assert not adata_b.layers["mask"][idx_s00].any()

    # S01: present in both views -> mask True (present) in view_b.
    idx_s01 = expected.index("S01")
    assert adata_b.layers["mask"][idx_s01].any()


def test_nan_kept_not_imputed(tmp_path):
    samples = ["S00", "S01", "S02"]
    features = ["g0", "g1"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 3
    )
    view_path = _write_view_csv(
        tmp_path / "view_a.csv", features, samples, nan_cells=[("g0", "S01")]
    )

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(out_path))

    import mudata
    mdata = mudata.read(str(out_path))
    adata = mdata.mod["view_a"]

    sample_idx = list(mdata.obs_names).index("S01")
    feat_idx = list(adata.var_names).index("g0")
    other_feat_idx = list(adata.var_names).index("g1")

    assert np.isnan(adata.X[sample_idx, feat_idx])
    assert not adata.layers["mask"][sample_idx, feat_idx]
    # Only the missing cell is affected; the rest of the row is untouched.
    assert not np.isnan(adata.X[sample_idx, other_feat_idx])


def test_mutations_reindexed_with_zero_fill(tmp_path):
    cond_samples = ["S00", "S01", "S02", "S03", "S04"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=cond_samples, model_types=["TypeA"] * 5
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], cond_samples)

    mut_samples = ["S00", "S01", "S02"]  # missing S03, S04
    mut_path = _write_mutations_csv(
        tmp_path / "mutations.csv", ["mut1", "mut2"], mut_samples,
        values=[[1, 0, 1], [0, 1, 0]],
    )

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path), [("view_a", str(view_path))], str(out_path),
        mutations_path=str(mut_path),
    )

    import mudata
    mdata = mudata.read(str(out_path))
    obs = mdata.obs

    assert "mutation_mut1" in obs.columns
    assert "mutation_mut2" in obs.columns
    assert obs.loc["S00", "mutation_mut1"] == 1
    assert obs.loc["S03", "mutation_mut1"] == 0  # not in mutations CSV -> filled 0
    assert obs.loc["S04", "mutation_mut2"] == 0


def test_round_trip_load_mudata(tmp_path):
    samples = [f"S{i:02d}" for i in range(8)]
    features_a = [f"gA_{i}" for i in range(4)]
    features_b = [f"gB_{i}" for i in range(3)]

    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples,
        model_types=["TypeA", "TypeB"] * 4,
        tissues=["tissueX"] * 8,
    )
    view_a_path = _write_view_csv(tmp_path / "view_a.csv", features_a, samples)
    view_b_path = _write_view_csv(tmp_path / "view_b.csv", features_b, samples)

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path),
        [("view_a", str(view_a_path)), ("view_b", str(view_b_path))],
        str(out_path),
    )

    dataset = load_mudata(str(out_path), ["view_a", "view_b"])

    assert isinstance(dataset, MultiOmicDataset)
    assert dataset.n_samples == 8
    assert dataset.views["view_a"].shape == (8, 4)
    assert dataset.views["view_b"].shape == (8, 3)
    assert dataset.masks["view_a"].shape == (8, 4)
    assert "model_type" in dataset.metadata.columns


def test_h5mu_zarr_parity(tmp_path):
    samples = [f"S{i:02d}" for i in range(6)]
    features_a = [f"gA_{i}" for i in range(3)]

    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 6
    )
    # S02: single partial-NaN cell. S04: fully missing (every feature NaN) --
    # exercises the h5mu/zarr divergence around fully-missing-sample handling.
    nan_cells = [("gA_0", "S02")] + [(f, "S04") for f in features_a]
    view_a_path = _write_view_csv(
        tmp_path / "view_a.csv", features_a, samples, nan_cells=nan_cells
    )

    h5mu_path = tmp_path / "out.h5mu"
    zarr_path = tmp_path / "out.zarr"
    csv_to_mudata(str(cond_path), [("view_a", str(view_a_path))], str(h5mu_path), format="h5mu")
    csv_to_mudata(str(cond_path), [("view_a", str(view_a_path))], str(zarr_path), format="zarr")

    ds_h5 = load_mudata(str(h5mu_path), ["view_a"])
    ds_zarr = load_mudata(str(zarr_path), ["view_a"])

    assert np.allclose(
        ds_h5.views["view_a"], ds_zarr.views["view_a"], equal_nan=True
    )
    assert np.array_equal(ds_h5.masks["view_a"], ds_zarr.masks["view_a"])
    assert ds_h5.feature_names["view_a"] == ds_zarr.feature_names["view_a"]


# ===========================================================================
# Group B -- CSV-adapter specifics
# ===========================================================================

def test_conditionals_missing_model_id_column(tmp_path):
    cond_path = tmp_path / "conditionals.csv"
    pd.DataFrame({"model_type": ["TypeA", "TypeB"]}).to_csv(cond_path, index=False)

    with pytest.raises(ValueError, match="model_id"):
        csv_to_mudata(str(cond_path), [], str(tmp_path / "out.h5mu"))


def test_conditionals_missing_model_type_column(tmp_path):
    cond_path = tmp_path / "conditionals.csv"
    pd.DataFrame({"model_id": ["S00", "S01"]}).to_csv(cond_path, index=False)

    with pytest.raises(ValueError, match="model_type"):
        csv_to_mudata(str(cond_path), [], str(tmp_path / "out.h5mu"))


def test_conditionals_duplicate_model_id(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv",
        model_ids=["S00", "S01", "S00"],
        model_types=["TypeA", "TypeB", "TypeA"],
    )
    with pytest.raises(ValueError, match="duplicate"):
        csv_to_mudata(str(cond_path), [], str(tmp_path / "out.h5mu"))


def test_conditionals_missing_tissue_warns_but_succeeds(tmp_path, caplog):
    samples = ["S00", "S01", "S02"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 3
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0", "g1"], samples)
    out_path = tmp_path / "out.h5mu"

    with caplog.at_level(logging.WARNING):
        csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(out_path))

    assert out_path.exists()
    assert "tissue" in caplog.text


def test_view_csv_transposed_raises(tmp_path):
    samples = [f"S{i:02d}" for i in range(15)]  # >10 overlapping IDs to trip the threshold
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 15
    )
    # Wrong orientation: sample IDs as the row index, features as columns.
    transposed = pd.DataFrame(
        np.random.RandomState(0).randn(15, 3), index=samples, columns=["g0", "g1", "g2"]
    )
    view_path = tmp_path / "view_a.csv"
    transposed.to_csv(view_path)

    with pytest.raises(ValueError, match="samples x features"):
        csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"))


def test_view_csv_non_numeric_cell_raises(tmp_path):
    samples = ["S00", "S01", "S02"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 3
    )
    view_path = tmp_path / "view_a.csv"
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], ["oops", 5.0, 6.0]], index=["g0", "g1"], columns=samples
    )
    df.to_csv(view_path)

    with pytest.raises(ValueError, match="non-numeric"):
        csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"))


def test_missing_view_csv_raises_file_not_found(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    with pytest.raises(FileNotFoundError):
        csv_to_mudata(
            str(cond_path),
            [("view_a", str(tmp_path / "does_not_exist.csv"))],
            str(tmp_path / "out.h5mu"),
        )


def test_missing_mutations_csv_raises_file_not_found(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)

    with pytest.raises(FileNotFoundError):
        csv_to_mudata(
            str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"),
            mutations_path=str(tmp_path / "does_not_exist_mut.csv"),
        )


def test_disjoint_ids_raises_no_samples(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv",
        model_ids=["S00", "S01", "S02"],
        model_types=["TypeA"] * 3,
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], ["T00", "T01", "T02"])

    with pytest.raises(ValueError, match="No samples"):
        csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"))


def test_format_extension_mismatch_warns_but_writes(tmp_path, caplog):
    samples = ["S00", "S01", "S02"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 3
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)
    out_path = tmp_path / "out.h5mu"  # .h5mu extension, but format="zarr"

    with caplog.at_level(logging.WARNING):
        csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(out_path), format="zarr")

    assert out_path.is_dir()  # zarr stores are directories, despite the .h5mu name
    assert "zarr" in caplog.text.lower()
