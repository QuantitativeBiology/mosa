"""The workflows a researcher actually runs, end to end through the CLI.

Every other test starts partway through: from a fixture-built MuData file, or
from a config that already matches it. These start from CSVs on disk and run
the commands in the order a user runs them, passing each stage's output to the
next. Nothing is imported except `main` — if a stage cannot consume what the
previous stage wrote, the chain breaks here rather than in someone's terminal.

Two workflows from the project's user stories:

  - the full pipeline: convert, validate, train, transform, cross-validate,
    optimize, plot;
  - adding an omic view, which the configurability requirement says must cost
    one config edit and no source edits.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mosa.cli import main

VIEWS = {"gexp": 8, "meth": 6}


@pytest.fixture
def csv_inputs(make_csv_dataset, tmp_path):
    return make_csv_dataset(
        tmp_path, n_samples=16, view_specs=VIEWS, n_types=2, mutations=3
    )


def test_full_pipeline_from_csvs_to_plots(
    csv_inputs, make_config_file, tmp_path, capsys
):
    stages: list[str] = []
    mudata_path = tmp_path / "study.h5mu"
    run_dir = tmp_path / "run"

    # 1. convert: CSVs to the MuData file every later stage reads.
    main(
        [
            "convert",
            "--conditionals",
            str(csv_inputs["conditionals"]),
            "--view",
            f"gexp:{csv_inputs['views']['gexp']}",
            "--view",
            f"meth:{csv_inputs['views']['meth']}",
            "--mutations",
            str(csv_inputs["mutations"]),
            "--output",
            str(mudata_path),
        ]
    )
    assert mudata_path.exists()
    stages.append("convert")

    # 2. inspect: what the file contains, before committing to a config.
    capsys.readouterr()
    main(["inspect", "--input", str(mudata_path)])
    inspected = capsys.readouterr().out
    assert "gexp" in inspected and "meth" in inspected
    stages.append("inspect")

    # 3. validate: the config against that file, without training.
    config = make_config_file(
        tmp_path, mudata_path, run_dir, VIEWS, evaluation={"test_size": 0.25}
    )
    main(["validate", "--config", str(config)])
    assert "Config OK" in capsys.readouterr().out
    assert not run_dir.exists(), "validate must not create the output directory"
    stages.append("validate")

    # 4. train: the outputs plot reads and the checkpoint transform reads.
    main(["train", "--config", str(config)])
    for split in ("train", "val", "full"):
        assert (run_dir / split / "latent.parquet").exists()
    checkpoint = run_dir / "last.ckpt"
    assert checkpoint.exists()
    stages.append("train")

    # 5. transform: project the same cohort through the saved checkpoint.
    projected = tmp_path / "projected"
    main(
        [
            "transform",
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(mudata_path),
            "--output",
            str(projected),
            "--reconstruct",
        ]
    )
    latent = pd.read_parquet(projected / "latent.parquet")
    assert latent.shape == (16, 8)
    for view in VIEWS:
        assert (projected / f"recon_{view}.parquet").exists()
    stages.append("transform")

    # 6. cross-validate: score the configuration on held-out folds.
    capsys.readouterr()
    main(["cross-validate", "--config", str(config), "--folds", "2"])
    assert "Aggregate NMSE" in capsys.readouterr().out
    stages.append("cross-validate")

    # 7. optimize: search over that configuration, scored by the same folds.
    search_space = tmp_path / "space.yaml"
    search_space.write_text(
        "learning_rate:\n  dist: loguniform\n  low: 1.0e-4\n  high: 1.0e-2\n"
    )
    capsys.readouterr()
    main(
        [
            "optimize",
            "--config",
            str(config),
            "--search-space",
            str(search_space),
            "--trials",
            "2",
            "--folds",
            "2",
        ]
    )
    assert "Best params:" in capsys.readouterr().out
    stages.append("optimize")

    # 8. plot: diagnostics from the training outputs.
    capsys.readouterr()
    main(["plot", "--config", str(config)])
    assert "Plots saved to" in capsys.readouterr().out
    assert any((run_dir / "plots").rglob("*.png"))
    stages.append("plot")

    expected = [
        "convert",
        "inspect",
        "validate",
        "train",
        "transform",
        "cross-validate",
        "optimize",
        "plot",
    ]
    assert stages == expected


def test_pipeline_survives_a_cohort_with_partial_omic_coverage(
    make_csv_dataset, make_config_file, tmp_path
):
    """The cohort Rita works with: not every sample has every omic.

    convert unions the sample axis and records coverage in the mask, so the
    run has to complete and place every sample in the latent space, not just
    the fully measured ones.
    """
    csvs = make_csv_dataset(tmp_path, n_samples=16, view_specs=VIEWS)
    partial = pd.read_csv(csvs["views"]["meth"], index_col=0)
    partial = partial.drop(columns=[f"S{i:03d}" for i in range(4)])
    partial.to_csv(csvs["views"]["meth"])

    mudata_path = tmp_path / "partial.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            "--view",
            f"gexp:{csvs['views']['gexp']}",
            "--view",
            f"meth:{csvs['views']['meth']}",
            "--output",
            str(mudata_path),
            "--min-views",
            "1",
        ]
    )

    run_dir = tmp_path / "run"
    config = make_config_file(tmp_path, mudata_path, run_dir, VIEWS)
    main(["train", "--config", str(config)])

    latent = pd.read_parquet(run_dir / "full" / "latent.parquet")
    assert len(latent) == 16
    assert latent.notna().all().all()
    assert {f"S{i:03d}" for i in range(4)} <= set(latent.index)


def test_adding_a_view_costs_one_config_edit_and_no_source_edits(
    make_csv_dataset, make_config_file, tmp_path
):
    """Requirement C1, as a test: the same converted file trains with two
    views or three, and the only difference between the runs is the YAML."""
    three_views = {"gexp": 8, "meth": 6, "cnv": 5}
    csvs = make_csv_dataset(tmp_path, n_samples=16, view_specs=three_views)

    mudata_path = tmp_path / "three_views.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *[
                arg
                for name, path in csvs["views"].items()
                for arg in ("--view", f"{name}:{path}")
            ],
            "--output",
            str(mudata_path),
        ]
    )

    two_dir = tmp_path / "two"
    two_config = make_config_file(
        tmp_path, mudata_path, two_dir, ["gexp", "meth"], name="two.yaml"
    )
    main(["train", "--config", str(two_config)])

    three_dir = tmp_path / "three"
    three_config = make_config_file(
        tmp_path, mudata_path, three_dir, list(three_views), name="three.yaml"
    )
    main(["train", "--config", str(three_config)])

    assert not (two_dir / "full" / "recon_cnv.parquet").exists()
    assert (three_dir / "full" / "recon_cnv.parquet").exists()

    # Same samples either way; only the omics being modelled changed.
    two_latent = pd.read_parquet(two_dir / "full" / "latent.parquet")
    three_latent = pd.read_parquet(three_dir / "full" / "latent.parquet")
    assert list(two_latent.index) == list(three_latent.index)
    assert two_latent.shape[1] == three_latent.shape[1] == 8


def test_dropping_a_view_needs_no_reconversion(csv_inputs, make_config_file, tmp_path):
    """The other half of C1: narrowing the config never touches the data file."""
    mudata_path = tmp_path / "study.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csv_inputs["conditionals"]),
            "--view",
            f"gexp:{csv_inputs['views']['gexp']}",
            "--view",
            f"meth:{csv_inputs['views']['meth']}",
            "--output",
            str(mudata_path),
        ]
    )
    written_at = mudata_path.stat().st_mtime_ns

    run_dir = tmp_path / "single"
    config = make_config_file(tmp_path, mudata_path, run_dir, ["gexp"])
    main(["train", "--config", str(config)])

    assert (run_dir / "full" / "recon_gexp.parquet").exists()
    assert not (run_dir / "full" / "recon_meth.parquet").exists()
    assert mudata_path.stat().st_mtime_ns == written_at, (
        "training rewrote the input data file"
    )
