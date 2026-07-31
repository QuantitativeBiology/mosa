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


# CSV-writing helpers

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


# Group A: format-agnostic contract

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

    # S00: present in view_a, absent from view_b: NaN row + mask False (absent).
    idx_s00 = expected.index("S00")
    adata_b = mdata.mod["view_b"]
    assert np.all(np.isnan(adata_b.X[idx_s00]))
    assert not adata_b.layers["mask"][idx_s00].any()

    # S01: present in both views: mask True (present) in view_b.
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
    assert obs.loc["S03", "mutation_mut1"] == 0  # not in mutations CSV, filled 0
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
    # S02: single partial-NaN cell. S04: fully missing (every feature NaN);
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


def test_multi_file_view_unions_features(tmp_path):
    """One omic assembled from two files: samples concatenate, features union."""
    samples_a = ["S00", "S01"]
    samples_b = ["S02", "S03"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv",
        model_ids=samples_a + samples_b, model_types=["TypeA"] * 4,
    )
    # g0/g1 shared; g2 only in file a; g3 only in file b
    path_a = _write_view_csv(tmp_path / "gexp_a.csv", ["g0", "g1", "g2"], samples_a)
    path_b = _write_view_csv(tmp_path / "gexp_b.csv", ["g0", "g1", "g3"], samples_b)

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path),
        [("gexp", str(path_a)), ("gexp", str(path_b))],
        str(out_path),
    )

    ds = load_mudata(str(out_path), ["gexp"])
    features = ds.feature_names["gexp"]

    assert sorted(features) == ["g0", "g1", "g2", "g3"]
    assert ds.views["gexp"].shape == (4, 4)

    # g2 came only from file a, so file b's samples have it missing, not dropped.
    i_s02 = list(ds.metadata.index).index("S02")
    i_g2 = features.index("g2")
    assert np.isnan(ds.views["gexp"][i_s02, i_g2])
    assert not ds.masks["gexp"][i_s02, i_g2]

    # A shared feature is observed for every sample.
    i_g0 = features.index("g0")
    assert ds.masks["gexp"][:, i_g0].all()


def test_multi_file_view_duplicate_sample_raises(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    path_a = _write_view_csv(tmp_path / "gexp_a.csv", ["g0"], samples)
    path_b = _write_view_csv(tmp_path / "gexp_b.csv", ["g1"], ["S01"])  # S01 repeats

    with pytest.raises(ValueError, match="occur more than once"):
        csv_to_mudata(
            str(cond_path),
            [("gexp", str(path_a)), ("gexp", str(path_b))],
            str(tmp_path / "out.h5mu"),
        )


def _write_id_map_csv(path, pairs):
    pd.DataFrame(pairs, columns=["source_id", "model_id"]).to_csv(path, index=False)
    return path


def test_id_map_renames_view_samples(tmp_path):
    """A crosswalk lets two providers' names for one sample line up."""
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv",
        model_ids=["SIDM01", "SIDM02"], model_types=["TypeA"] * 2,
    )
    # gexp already uses canonical IDs; crispr uses the provider's WTS names.
    gexp_path = _write_view_csv(tmp_path / "gexp.csv", ["g0", "g1"], ["SIDM01", "SIDM02"])
    crispr_path = _write_view_csv(tmp_path / "crispr.csv", ["c0"], ["WTS-1", "WTS-2"])
    id_map_path = _write_id_map_csv(
        tmp_path / "id_map.csv", [("WTS-1", "SIDM01"), ("WTS-2", "SIDM02")]
    )

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path),
        [("gexp", str(gexp_path)), ("crispr", str(crispr_path))],
        str(out_path),
        id_map_path=str(id_map_path),
    )

    ds = load_mudata(str(out_path), ["gexp", "crispr"])

    # Without the map, crispr's samples would not have matched at all.
    assert list(ds.metadata.index) == ["SIDM01", "SIDM02"]
    assert ds.masks["crispr"].all()
    assert ds.masks["gexp"].all()


def test_id_map_collision_errors_by_default(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=["P01"], model_types=["TypeA"]
    )
    view_path = _write_view_csv(tmp_path / "gexp.csv", ["g0"], ["P01-A", "P01-B"])
    id_map_path = _write_id_map_csv(
        tmp_path / "id_map.csv", [("P01-A", "P01"), ("P01-B", "P01")]
    )

    with pytest.raises(ValueError, match="occur more than once"):
        csv_to_mudata(
            str(cond_path), [("gexp", str(view_path))], str(tmp_path / "out.h5mu"),
            id_map_path=str(id_map_path),
        )


def test_id_map_collision_first_keeps_first(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=["P01"], model_types=["TypeA"]
    )
    view_path = tmp_path / "gexp.csv"
    pd.DataFrame(
        [[1.0, 2.0]], index=["g0"], columns=["P01-A", "P01-B"]
    ).to_csv(view_path)
    id_map_path = _write_id_map_csv(
        tmp_path / "id_map.csv", [("P01-A", "P01"), ("P01-B", "P01")]
    )

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path), [("gexp", str(view_path))], str(out_path),
        id_map_path=str(id_map_path), on_collision="first",
    )

    ds = load_mudata(str(out_path), ["gexp"])
    assert ds.views["gexp"].shape == (1, 1)
    assert ds.views["gexp"][0, 0] == pytest.approx(1.0)  # P01-A, not P01-B


def test_id_map_missing_column_raises(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=["S00"], model_types=["TypeA"]
    )
    view_path = _write_view_csv(tmp_path / "gexp.csv", ["g0"], ["S00"])
    bad_map = tmp_path / "id_map.csv"
    pd.DataFrame({"from": ["a"], "to": ["b"]}).to_csv(bad_map, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        csv_to_mudata(
            str(cond_path), [("gexp", str(view_path))], str(tmp_path / "out.h5mu"),
            id_map_path=str(bad_map),
        )


def test_invalid_on_collision_raises(tmp_path):
    with pytest.raises(ValueError, match="on_collision"):
        csv_to_mudata("unused", [], str(tmp_path / "out.h5mu"), on_collision="mean")


def test_has_view_distinguishes_absent_from_all_nan(tmp_path):
    """A sample missing from an omic and one measured as all-NaN differ."""
    samples = ["S00", "S01", "S02"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 3
    )
    view_a_path = _write_view_csv(tmp_path / "view_a.csv", ["g0", "g1"], samples)
    # view_b omits S00 entirely, and carries S01 as all-NaN.
    view_b_path = _write_view_csv(
        tmp_path / "view_b.csv", ["h0", "h1"], ["S01", "S02"],
        nan_cells=[("h0", "S01"), ("h1", "S01")],
    )

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path),
        [("view_a", str(view_a_path)), ("view_b", str(view_b_path))],
        str(out_path),
    )

    import mudata
    obs = mudata.read(str(out_path)).obs

    assert not obs.loc["S00", "has_view_b"]  # absent from the file
    assert obs.loc["S01", "has_view_b"]      # present, but every value missing
    assert obs.loc["S02", "has_view_b"]
    assert obs["has_view_a"].all()

    # The mask cannot tell S00 and S01 apart.
    ds = load_mudata(str(out_path), ["view_a", "view_b"])
    mask_b = ds.masks["view_b"]
    axis = list(ds.metadata.index)
    assert not mask_b[axis.index("S00")].any()
    assert not mask_b[axis.index("S01")].any()


def test_min_views_boundaries(tmp_path):
    samples = ["S00", "S01", "S02"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 3
    )
    # S00: view_a only. S01: both. S02: view_b only.
    view_a_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], ["S00", "S01"])
    view_b_path = _write_view_csv(tmp_path / "view_b.csv", ["h0"], ["S01", "S02"])
    specs = [("view_a", str(view_a_path)), ("view_b", str(view_b_path))]

    out1 = tmp_path / "out1.h5mu"
    csv_to_mudata(str(cond_path), specs, str(out1), min_views=1)
    assert list(load_mudata(str(out1), ["view_a", "view_b"]).metadata.index) == samples

    out2 = tmp_path / "out2.h5mu"
    csv_to_mudata(str(cond_path), specs, str(out2), min_views=2)
    assert list(load_mudata(str(out2), ["view_a", "view_b"]).metadata.index) == ["S01"]


def test_min_views_counts_observed_data_not_file_presence(tmp_path):
    """A view counts toward min_views only if it holds at least one real value.

    A sample listed in a view's file but measured as all-NaN does not count,
    though has_<view> still records that it was listed.
    """
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    view_a_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)
    # S00 is listed in view_b but every value is missing; S01 is absent entirely.
    view_b_path = _write_view_csv(
        tmp_path / "view_b.csv", ["h0"], ["S00"], nan_cells=[("h0", "S00")]
    )
    specs = [("view_a", str(view_a_path)), ("view_b", str(view_b_path))]

    # min_views=1: both survive, and has_view_b still distinguishes them.
    out1 = tmp_path / "out1.h5mu"
    csv_to_mudata(str(cond_path), specs, str(out1), min_views=1)
    ds1 = load_mudata(str(out1), ["view_a", "view_b"])
    assert list(ds1.metadata.index) == ["S00", "S01"]
    assert list(ds1.metadata["has_view_b"]) == [True, False]

    # min_views=2: S00's view_b is empty, so it has one real omic, not two.
    with pytest.raises(ValueError, match="at least 2 views"):
        csv_to_mudata(str(cond_path), specs, str(tmp_path / "out2.h5mu"), min_views=2)


def test_min_views_above_view_count_raises(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)

    with pytest.raises(ValueError, match="at least 3 views"):
        csv_to_mudata(
            str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"),
            min_views=3,
        )


def test_presence_survives_zarr_round_trip(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    view_a_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)
    view_b_path = _write_view_csv(tmp_path / "view_b.csv", ["h0"], ["S01"])
    specs = [("view_a", str(view_a_path)), ("view_b", str(view_b_path))]

    zarr_path = tmp_path / "out.zarr"
    csv_to_mudata(str(cond_path), specs, str(zarr_path), format="zarr")

    ds = load_mudata(str(zarr_path), ["view_a", "view_b"])
    assert list(ds.metadata["has_view_b"]) == [False, True]


def test_inspect_handles_presence_columns(tmp_path, capsys):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)
    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(out_path))

    from mosa.data.io import inspect_mudata
    inspect_mudata(str(out_path))

    assert "has_view_a" in capsys.readouterr().out


def test_conversion_report_attributes_every_dropped_sample(tmp_path, capsys):
    # 5 metadata samples; S04 has no omic data. GHOST is in a view but has no
    # metadata. S00 has only one view, so min_views=2 drops it too.
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv",
        model_ids=["S00", "S01", "S02", "S03", "S04"], model_types=["TypeA"] * 5,
    )
    view_a_path = _write_view_csv(
        tmp_path / "view_a.csv", ["g0"], ["S00", "S01", "S02", "S03", "GHOST"]
    )
    view_b_path = _write_view_csv(tmp_path / "view_b.csv", ["h0"], ["S01", "S02", "S03"])

    csv_to_mudata(
        str(cond_path),
        [("view_a", str(view_a_path)), ("view_b", str(view_b_path))],
        str(tmp_path / "out.h5mu"),
        min_views=2,
    )

    out = capsys.readouterr().out
    assert "in metadata:" in out
    # 5 in metadata, 5 in views (4 real + GHOST), 4 matched, 1 no-metadata,
    # 1 no-view-data, 1 dropped by min_views, 3 final.
    assert "dropped, no metadata:" in out
    assert "dropped, min_views=2:" in out

    def value(label):
        line = next(ln for ln in out.splitlines() if label in ln)
        return int(line.split()[-1])

    assert value("in metadata:") == 5
    assert value("in at least one view:") == 5
    assert value("matched on both:") == 4
    assert value("dropped, no metadata:") == 1
    assert value("dropped, no view data:") == 1
    assert value("dropped, min_views=2:") == 1
    assert value("final:") == 3


def test_sample_axis_independent_of_input_order(tmp_path):
    """The axis must not depend on view order or on file order within a view."""
    samples = ["S02", "S00", "S03", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 4
    )
    a1 = _write_view_csv(tmp_path / "a1.csv", ["g0"], ["S00", "S01"])
    a2 = _write_view_csv(tmp_path / "a2.csv", ["g1"], ["S02", "S03"])
    b = _write_view_csv(tmp_path / "b.csv", ["h0"], ["S01", "S02"])

    orderings = [
        [("va", str(a1)), ("va", str(a2)), ("vb", str(b))],
        [("vb", str(b)), ("va", str(a2)), ("va", str(a1))],
        [("va", str(a2)), ("vb", str(b)), ("va", str(a1))],
    ]

    axes = []
    for i, specs in enumerate(orderings):
        out_path = tmp_path / f"out{i}.h5mu"
        csv_to_mudata(str(cond_path), specs, str(out_path))
        ds = load_mudata(str(out_path), ["va", "vb"])
        axes.append(list(ds.metadata.index))

    assert axes[0] == ["S00", "S01", "S02", "S03"]
    assert axes[1] == axes[0]
    assert axes[2] == axes[0]


def test_metadata_filter_restricts_sample_axis(tmp_path):
    """Filtering the metadata restricts which samples reach the sample axis."""
    samples = ["S00", "S01", "S02", "S03"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples,
        model_types=["Cell_Line", "Organoid", "Tumor", "Cell_Line"],
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], samples)

    out_path = tmp_path / "out.h5mu"
    csv_to_mudata(
        str(cond_path), [("view_a", str(view_path))], str(out_path),
        metadata_filters={"model_type": ["Cell_Line", "Tumor"]},
    )

    ds = load_mudata(str(out_path), ["view_a"])
    assert list(ds.metadata.index) == ["S00", "S02", "S03"]


def test_metadata_filter_unknown_column_raises(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=["S00"], model_types=["TypeA"]
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], ["S00"])

    with pytest.raises(ValueError, match="not a column"):
        csv_to_mudata(
            str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"),
            metadata_filters={"nope": ["x"]},
        )


def test_metadata_filter_matching_nothing_raises(tmp_path):
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=["S00"], model_types=["TypeA"]
    )
    view_path = _write_view_csv(tmp_path / "view_a.csv", ["g0"], ["S00"])

    with pytest.raises(ValueError, match="matched no samples"):
        csv_to_mudata(
            str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu"),
            metadata_filters={"model_type": ["Nonexistent"]},
        )


def test_shared_features_intersects_across_views(tmp_path):
    """shared_features reduces every view to the features they all share."""
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    # Gene-level views sharing g1/g2, each with one private feature.
    a_path = _write_view_csv(tmp_path / "a.csv", ["g0", "g1", "g2"], samples)
    b_path = _write_view_csv(tmp_path / "b.csv", ["g1", "g2", "g3"], samples)
    specs = [("va", str(a_path)), ("vb", str(b_path))]

    off = tmp_path / "off.h5mu"
    csv_to_mudata(str(cond_path), specs, str(off))
    ds_off = load_mudata(str(off), ["va", "vb"])
    assert ds_off.feature_names["va"] == ["g0", "g1", "g2"]
    assert ds_off.feature_names["vb"] == ["g1", "g2", "g3"]

    on = tmp_path / "on.h5mu"
    csv_to_mudata(str(cond_path), specs, str(on), shared_features=True)
    ds_on = load_mudata(str(on), ["va", "vb"])
    assert ds_on.feature_names["va"] == ["g1", "g2"]
    assert ds_on.feature_names["vb"] == ["g1", "g2"]


def test_shared_features_with_disjoint_namespaces_raises(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    a_path = _write_view_csv(tmp_path / "a.csv", ["gene1", "gene2"], samples)
    b_path = _write_view_csv(tmp_path / "b.csv", ["cg0001", "cg0002"], samples)

    with pytest.raises(ValueError, match="no feature names in common"):
        csv_to_mudata(
            str(cond_path),
            [("va", str(a_path)), ("vb", str(b_path))],
            str(tmp_path / "out.h5mu"),
            shared_features=True,
        )


def test_input_format_parity(tmp_path):
    """csv, tsv, gzipped csv and parquet inputs must produce identical output."""
    samples = [f"S{i:02d}" for i in range(5)]
    features = [f"g{i}" for i in range(4)]
    view = pd.DataFrame(
        np.random.RandomState(0).randn(len(features), len(samples)),
        index=features, columns=samples,
    )
    view.iloc[0, 1] = np.nan  # keep a missing value inside the comparison
    cond = pd.DataFrame({"model_id": samples, "model_type": ["TypeA", "TypeB"] * 2 + ["TypeA"]})

    writers = {
        "csv": (lambda p: view.to_csv(p), lambda p: cond.to_csv(p, index=False)),
        "csv.gz": (lambda p: view.to_csv(p), lambda p: cond.to_csv(p, index=False)),
        "tsv": (lambda p: view.to_csv(p, sep="\t"), lambda p: cond.to_csv(p, sep="\t", index=False)),
        "parquet": (lambda p: view.to_parquet(p), lambda p: cond.to_parquet(p, index=False)),
    }

    outputs = {}
    for fmt, (write_view, write_cond) in writers.items():
        view_path = tmp_path / f"view_a.{fmt}"
        cond_path = tmp_path / f"conditionals.{fmt}"
        write_view(view_path)
        write_cond(cond_path)

        out_path = tmp_path / f"out_{fmt.replace('.', '_')}.h5mu"
        csv_to_mudata(str(cond_path), [("view_a", str(view_path))], str(out_path))
        outputs[fmt] = load_mudata(str(out_path), ["view_a"])

    ref = outputs["csv"]
    for fmt, ds in outputs.items():
        assert list(ds.metadata.index) == list(ref.metadata.index), fmt
        assert ds.feature_names["view_a"] == ref.feature_names["view_a"], fmt
        assert np.allclose(ds.views["view_a"], ref.views["view_a"], equal_nan=True), fmt
        assert np.array_equal(ds.masks["view_a"], ref.masks["view_a"]), fmt


def test_unsupported_input_format_raises(tmp_path):
    samples = ["S00", "S01"]
    cond_path = _write_conditionals_csv(
        tmp_path / "conditionals.csv", model_ids=samples, model_types=["TypeA"] * 2
    )
    view_path = tmp_path / "view_a.xlsx"
    view_path.write_bytes(b"not really a spreadsheet")

    with pytest.raises(ValueError, match="Unsupported table format"):
        csv_to_mudata(
            str(cond_path), [("view_a", str(view_path))], str(tmp_path / "out.h5mu")
        )


# Group B: CSV-adapter specifics

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
