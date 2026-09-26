"""Training on a file `mosa convert` actually produced.

Every other training test starts from make_h5mu_file, a hand-built fixture:
dense views, no NaN, no mutations, and none of the `has_<view>` columns the
converter writes. So the two halves of the pipeline were each tested against
a different idea of what a MuData file contains, and nothing checked that
converter output is trainable at all.

The dataset here is deliberately awkward in the ways real conversions are:
one sample absent from a view, NaN cells inside another, and a mutations
table. Those are the cases where the mask layer has to carry the information
the values cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mosa.cli import main
from mosa.data.io import load_mudata, summarize_structure

VIEWS = {"view_a": 6, "view_b": 5}


@pytest.fixture
def converted(make_csv_dataset, tmp_path):
    """CSVs converted through the `mosa convert` CLI, with real-world gaps.

    S000 is absent from view_b entirely (a dropped column, not an empty one),
    and view_a has NaN cells. Both must survive as mask entries rather than
    as values.
    """
    csvs = make_csv_dataset(tmp_path, n_samples=12, view_specs=VIEWS, mutations=3)

    view_b = pd.read_csv(csvs["views"]["view_b"], index_col=0).drop(columns=["S000"])
    view_b.to_csv(csvs["views"]["view_b"])

    view_a = pd.read_csv(csvs["views"]["view_a"], index_col=0)
    view_a.iloc[0, 1] = np.nan
    view_a.iloc[2, 3] = np.nan
    view_a.to_csv(csvs["views"]["view_a"])

    out = tmp_path / "converted.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            "--view",
            f"view_a:{csvs['views']['view_a']}",
            "--view",
            f"view_b:{csvs['views']['view_b']}",
            "--mutations",
            str(csvs["mutations"]),
            "--output",
            str(out),
        ]
    )
    return out


def test_converted_file_carries_the_columns_the_fixtures_omit(converted):
    """Guards the premise of this file: if convert stopped writing has_<view>,
    the tests below would silently go back to testing fixture-shaped data."""
    obs_columns = summarize_structure(str(converted))["obs_columns"]

    assert "has_view_a" in obs_columns
    assert "has_view_b" in obs_columns
    assert [c for c in obs_columns if c.startswith("mutation_")]


def test_converted_file_records_absence_and_nan_in_the_mask(converted):
    dataset = load_mudata(str(converted), list(VIEWS), "mask")

    assert dataset.n_samples == 12
    row = dataset.sample_names.index("S000")
    assert not dataset.masks["view_b"][row].any(), (
        "a sample absent from a view should be entirely unobserved in it"
    )
    assert dataset.masks["view_a"][row].any(), "view_a is present for that sample"
    assert not dataset.masks["view_a"].all(), "NaN cells should be masked out"


def test_train_runs_on_converted_output(converted, make_config_file, tmp_path):
    output_dir = tmp_path / "run"
    config = make_config_file(tmp_path, converted, output_dir, VIEWS)

    main(["train", "--config", str(config)])

    latent = pd.read_parquet(output_dir / "full" / "latent.parquet")
    assert latent.shape == (12, 8)
    assert np.isfinite(latent.to_numpy()).all(), (
        "NaN in the source CSVs reached the latent space; the mask/impute path "
        "did not cover converter output"
    )

    for view, n_features in VIEWS.items():
        recon = pd.read_parquet(output_dir / "full" / f"recon_{view}.parquet")
        assert recon.shape == (12, n_features)
        assert np.isfinite(recon.to_numpy()).all()


def test_sample_absent_from_a_view_still_gets_a_latent_row(
    converted, make_config_file, tmp_path
):
    """The union sample axis is the point of convert: a sample with one omic
    is still a sample, and the model must place it in the latent space."""
    output_dir = tmp_path / "run"
    config = make_config_file(tmp_path, converted, output_dir, VIEWS)

    main(["train", "--config", str(config)])

    latent = pd.read_parquet(output_dir / "full" / "latent.parquet")
    assert "S000" in latent.index
    assert np.isfinite(latent.loc["S000"].to_numpy()).all()


def test_latent_rows_stay_on_the_converted_sample_axis(
    converted, make_config_file, tmp_path
):
    """Row i of the latent must be sample i of the file, in the same order."""
    output_dir = tmp_path / "run"
    config = make_config_file(tmp_path, converted, output_dir, VIEWS)

    main(["train", "--config", str(config)])

    dataset = load_mudata(str(converted), list(VIEWS), "mask")
    latent = pd.read_parquet(output_dir / "full" / "latent.parquet")
    assert list(latent.index) == dataset.sample_names


def test_datamodule_reads_mutations_and_ignores_presence_columns(converted):
    """`has_<view>` is bookkeeping, `mutation_*` is model input. Both are plain
    .obs columns, so only the prefix rule keeps them apart."""
    from mosa.config import DataConfig
    from mosa.models.mosa.config import MOSAConfig, OmicViewConfig
    from mosa.models.mosa.datamodule import MOSADataModule

    dataset = load_mudata(str(converted), list(VIEWS), "mask")
    data_cfg = DataConfig(path=str(converted), views=list(VIEWS))
    model_cfg = MOSAConfig(
        views={
            name: OmicViewConfig(name=name, hidden_layer_dims=[8]) for name in VIEWS
        },
        joint_latent_dim=4,
        num_epochs=1,
        batch_size=8,
    )

    dm = MOSADataModule(
        train_data=dataset, val_data=None, data_cfg=data_cfg, model_cfg=model_cfg
    )
    dm.setup()

    assert dm.mutation_columns == [
        "mutation_gene_0",
        "mutation_gene_1",
        "mutation_gene_2",
    ]
    assert not any(c.startswith("has_") for c in dm.mutation_columns)


def test_train_runs_on_converted_zarr_output(
    make_csv_dataset, make_config_file, tmp_path
):
    """The other output format convert offers, trained through the same path."""
    csvs = make_csv_dataset(tmp_path, n_samples=12, view_specs=VIEWS)
    store = tmp_path / "converted.zarr"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            "--view",
            f"view_a:{csvs['views']['view_a']}",
            "--view",
            f"view_b:{csvs['views']['view_b']}",
            "--output",
            str(store),
            "--format",
            "zarr",
        ]
    )

    output_dir = tmp_path / "run_zarr"
    config = make_config_file(tmp_path, store, output_dir, VIEWS)
    main(["train", "--config", str(config)])

    latent = pd.read_parquet(output_dir / "full" / "latent.parquet")
    assert latent.shape == (12, 8)
    assert np.isfinite(latent.to_numpy()).all()
