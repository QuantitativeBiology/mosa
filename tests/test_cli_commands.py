"""Happy-path contract for every CLI subcommand.

test_cli_errors.py covers what the CLI does when the user is wrong. This
covers what it does when the user is right: the command completes, and the
artifact the next command consumes exists.

Two gaps this closes. First, only `train` had a success-path test, so the
other seven commands were verified solely through their error messages.
Second, `convert`'s flags were exercised against csv_to_mudata() directly
(test_convert.py), never through argument parsing — a wrong kwarg name in
cli._convert passed every test in the suite.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from mosa.cli import main

# Subcommands with a success-path test in this file. The completeness test at
# the bottom fails when a subcommand is added to the CLI and not to this set.
COVERED_COMMANDS = {
    "train",
    "transform",
    "cross-validate",
    "optimize",
    "plot",
    "convert",
    "inspect",
    "validate",
}

VIEWS = {"view_a": 6, "view_b": 5}


@pytest.fixture
def trained_run(make_h5mu_file, make_config_file, tmp_path):
    """A completed `mosa train` with a validation split, so a checkpoint exists.

    Checkpoints are only written when a val split exists (MOSAModel.fit adds
    ModelCheckpoint under `if has_val`), so downstream commands that need one
    have to train with test_size > 0.
    """
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    output_dir = tmp_path / "run"
    config = make_config_file(
        tmp_path, h5mu, output_dir, VIEWS, evaluation={"test_size": 0.25}
    )

    main(["train", "--config", str(config)])
    return {"config": config, "data": h5mu, "output_dir": output_dir}


# train


def test_train_writes_outputs_for_every_split(trained_run):
    output_dir = trained_run["output_dir"]
    for split in ("train", "val", "full"):
        assert (output_dir / split / "latent.parquet").exists()
        for view in VIEWS:
            assert (output_dir / split / f"recon_{view}.parquet").exists()


def test_train_writes_a_loadable_checkpoint(trained_run):
    assert (trained_run["output_dir"] / "last.ckpt").exists()


def test_train_resume_continues_from_a_checkpoint(
    trained_run, make_config_file, tmp_path
):
    """--resume must pick training up from the checkpoint's weights.

    Asserted as a difference against the same config trained from scratch:
    resuming a 1-epoch checkpoint for 3 more epochs cannot land on the same
    weights as 3 epochs from random init under the same seed. Anything less
    than a comparison passes even when --resume is ignored entirely.
    """
    checkpoint = trained_run["output_dir"] / "last.ckpt"

    def _run(name, *extra):
        out_dir = tmp_path / name
        config = make_config_file(
            tmp_path,
            trained_run["data"],
            out_dir,
            VIEWS,
            name=f"{name}.yaml",
            num_epochs=3,
            evaluation={"test_size": 0.25},
        )
        main(["train", "--config", str(config), *extra])
        return pd.read_parquet(out_dir / "full" / "latent.parquet")

    resumed = _run("resumed", "--resume", str(checkpoint))
    from_scratch = _run("scratch")

    assert resumed.shape == from_scratch.shape
    assert not resumed.equals(from_scratch), (
        "resuming produced the same latent as training from scratch; "
        "--resume was likely not passed through to fit()"
    )


# transform


def test_transform_writes_latent_parquet(trained_run, tmp_path):
    out = tmp_path / "projected"
    main(
        [
            "transform",
            "--checkpoint",
            str(trained_run["output_dir"] / "last.ckpt"),
            "--input",
            str(trained_run["data"]),
            "--output",
            str(out),
        ]
    )

    latent = pd.read_parquet(out / "latent.parquet")
    assert latent.shape == (16, 8)
    assert not latent.isna().any().any()


def test_transform_reconstruct_flag_adds_per_view_parquets(trained_run, tmp_path):
    out = tmp_path / "projected_recon"
    main(
        [
            "transform",
            "--checkpoint",
            str(trained_run["output_dir"] / "last.ckpt"),
            "--input",
            str(trained_run["data"]),
            "--output",
            str(out),
            "--reconstruct",
        ]
    )

    for view, n_features in VIEWS.items():
        recon = pd.read_parquet(out / f"recon_{view}.parquet")
        assert recon.shape == (16, n_features)


# cross-validate


def test_cross_validate_prints_a_score_table(
    make_h5mu_file, make_config_file, tmp_path, capsys
):
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    config = make_config_file(tmp_path, h5mu, tmp_path / "cv", VIEWS)

    main(["cross-validate", "--config", str(config), "--folds", "2"])

    out = capsys.readouterr().out
    assert "Aggregate NMSE" in out
    for view in VIEWS:
        assert view in out


def test_cross_validate_flags_override_the_config_block(
    make_h5mu_file, make_config_file, tmp_path, capsys
):
    """--folds/--strategy/--no-shuffle override evaluation:, which n_folds proves."""
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    config = make_config_file(
        tmp_path,
        h5mu,
        tmp_path / "cv",
        VIEWS,
        evaluation={"test_size": 0.0, "n_folds": 5},
    )

    main(
        [
            "cross-validate",
            "--config",
            str(config),
            "--folds",
            "3",
            "--strategy",
            "kfold",
            "--no-shuffle",
        ]
    )

    out = capsys.readouterr().out
    fold_rows = [line for line in out.splitlines() if re.match(r"^\d+\s", line)]
    assert len(fold_rows) == 3, f"expected 3 folds from --folds, got:\n{out}"


def test_cross_validate_strategy_flag_selects_the_splitter(
    make_h5mu_file, make_config_file, tmp_path, capsys
):
    """9 folds over 2 classes of 8 is legal for kfold and illegal for stratified,
    so the pair only behaves this way if --strategy reached the evaluation config."""
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    config = make_config_file(tmp_path, h5mu, tmp_path / "cv", VIEWS)

    main(
        [
            "cross-validate",
            "--config",
            str(config),
            "--folds",
            "9",
            "--strategy",
            "kfold",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "cross-validate",
                "--config",
                str(config),
                "--folds",
                "9",
                "--strategy",
                "stratified",
            ]
        )
    assert exc.value.code == 1
    assert "smallest model_type" in capsys.readouterr().err


# optimize


def test_optimize_runs_the_requested_number_of_trials(
    make_h5mu_file, make_config_file, tmp_path, capsys
):
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    config = make_config_file(tmp_path, h5mu, tmp_path / "hpo", VIEWS)
    search_space = tmp_path / "space.yaml"
    search_space.write_text(
        "learning_rate:\n  dist: loguniform\n  low: 1.0e-4\n  high: 1.0e-2\n"
    )

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

    out = capsys.readouterr().out
    # The count of trials the study actually ran, not the echoed flag value.
    assert "Trials: 2 (" in out, out
    assert "Best params:" in out
    assert "learning_rate" in out


# plot


def test_plot_generates_plots_from_a_completed_run(trained_run, capsys):
    main(["plot", "--config", str(trained_run["config"])])

    out = capsys.readouterr().out
    assert "Plots saved to" in out
    plots_dir = trained_run["output_dir"] / "plots"
    assert plots_dir.is_dir()
    assert any(plots_dir.rglob("*.png")), "plot reported success but wrote no figures"


def test_plot_output_dir_flag_targets_another_run(trained_run, capsys):
    main(
        [
            "plot",
            "--config",
            str(trained_run["config"]),
            "--output-dir",
            str(trained_run["output_dir"]),
        ]
    )
    assert "Plots saved to" in capsys.readouterr().out


# convert


def test_convert_writes_an_h5mu_the_loader_accepts(make_csv_dataset, tmp_path):
    csvs = make_csv_dataset(tmp_path)
    out = tmp_path / "converted.h5mu"

    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(out),
        ]
    )

    from mosa.data.io import load_mudata

    dataset = load_mudata(str(out), list(csvs["views"]), "mask")
    assert dataset.n_samples == 12
    assert set(dataset.view_names) == set(csvs["views"])


def test_convert_format_zarr_writes_a_directory(make_csv_dataset, tmp_path):
    csvs = make_csv_dataset(tmp_path)
    out = tmp_path / "converted.zarr"

    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(out),
            "--format",
            "zarr",
        ]
    )

    assert out.is_dir()


def test_convert_mutations_become_obs_columns(make_csv_dataset, tmp_path):
    csvs = make_csv_dataset(tmp_path, mutations=4)
    out = tmp_path / "with_mutations.h5mu"

    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--mutations",
            str(csvs["mutations"]),
            "--output",
            str(out),
        ]
    )

    from mosa.data.io import summarize_structure

    obs_columns = summarize_structure(str(out))["obs_columns"]
    assert [c for c in obs_columns if c.startswith("mutation_")]


def test_convert_filter_flag_restricts_the_sample_axis(make_csv_dataset, tmp_path):
    """--filter parsing lives in cli._convert, not in csv_to_mudata."""
    csvs = make_csv_dataset(tmp_path, n_samples=12, n_types=2)
    out = tmp_path / "filtered.h5mu"

    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(out),
            "--filter",
            "model_type=TypeA",
        ]
    )

    from mosa.data.io import load_mudata

    dataset = load_mudata(str(out), list(csvs["views"]), "mask")
    assert dataset.n_samples == 6
    assert set(dataset.metadata["model_type"]) == {"TypeA"}


def test_convert_min_views_flag_reaches_the_converter(make_csv_dataset, tmp_path):
    """One sample is absent from view_b, so --min-views 2 must drop exactly it."""
    csvs = make_csv_dataset(tmp_path, n_samples=8)
    partial = pd.read_csv(csvs["views"]["view_b"], index_col=0).drop(columns=["S000"])
    partial.to_csv(csvs["views"]["view_b"])

    from mosa.data.io import load_mudata

    kept = tmp_path / "min1.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(kept),
            "--min-views",
            "1",
        ]
    )
    assert load_mudata(str(kept), list(csvs["views"]), "mask").n_samples == 8

    dropped = tmp_path / "min2.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(dropped),
            "--min-views",
            "2",
        ]
    )
    dataset = load_mudata(str(dropped), list(csvs["views"]), "mask")
    assert dataset.n_samples == 7
    assert "S000" not in dataset.sample_names


def test_convert_shared_features_flag_reaches_the_converter(tmp_path):
    """--shared-features intersects features, so both views need one namespace."""
    import numpy as np

    samples = [f"S{i:03d}" for i in range(8)]
    pd.DataFrame(
        {"model_id": samples, "model_type": ["TypeA"] * 8, "tissue": ["t"] * 8}
    ).to_csv(tmp_path / "cond.csv", index=False)

    rng = np.random.RandomState(0)
    for name, features in (
        ("view_a", ["gene_0", "gene_1", "gene_2"]),
        ("view_b", ["gene_1", "gene_2", "gene_3"]),
    ):
        pd.DataFrame(
            rng.randn(len(features), len(samples)), index=features, columns=samples
        ).to_csv(tmp_path / f"{name}.csv")

    out = tmp_path / "shared.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(tmp_path / "cond.csv"),
            "--view",
            f"view_a:{tmp_path / 'view_a.csv'}",
            "--view",
            f"view_b:{tmp_path / 'view_b.csv'}",
            "--output",
            str(out),
            "--shared-features",
        ]
    )

    from mosa.data.io import load_mudata

    dataset = load_mudata(str(out), ["view_a", "view_b"], "mask")
    assert dataset.feature_names["view_a"] == ["gene_1", "gene_2"]


def test_convert_id_map_and_on_collision_reach_the_converter(
    make_csv_dataset, tmp_path
):
    """The view names samples X0..X3, the metadata names them S000..S003.

    Nothing aligns unless --id-map is applied, and the map sends two source
    columns to one model ID, so the run also fails unless --on-collision
    reaches the converter. Both flags are load-bearing for this to succeed.
    """
    csvs = make_csv_dataset(tmp_path, n_samples=4, view_specs={"view_a": 3})
    view = pd.read_csv(csvs["views"]["view_a"], index_col=0)
    view.columns = [f"X{i}" for i in range(len(view.columns))]
    view.to_csv(csvs["views"]["view_a"])

    id_map = tmp_path / "id_map.csv"
    pd.DataFrame(
        {
            "source_id": ["X0", "X1", "X2", "X3"],
            "model_id": ["S000", "S000", "S002", "S003"],
        }
    ).to_csv(id_map, index=False)

    out = tmp_path / "mapped.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(out),
            "--id-map",
            str(id_map),
            "--on-collision",
            "first",
        ]
    )

    from mosa.data.io import load_mudata

    dataset = load_mudata(str(out), ["view_a"], "mask")
    assert dataset.sample_names == ["S000", "S002", "S003"]


def _view_args(csvs: dict) -> list[str]:
    args = []
    for name, path in csvs["views"].items():
        args += ["--view", f"{name}:{path}"]
    return args


# inspect / validate


def test_inspect_summarizes_a_converted_file(make_csv_dataset, tmp_path, capsys):
    csvs = make_csv_dataset(tmp_path)
    out = tmp_path / "inspected.h5mu"
    main(
        [
            "convert",
            "--conditionals",
            str(csvs["conditionals"]),
            *_view_args(csvs),
            "--output",
            str(out),
        ]
    )
    capsys.readouterr()

    main(["inspect", "--input", str(out)])

    printed = capsys.readouterr().out
    for view in csvs["views"]:
        assert view in printed
    assert "model_type" in printed


def test_validate_reports_a_good_config(
    make_h5mu_file, make_config_file, tmp_path, capsys
):
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    config = make_config_file(tmp_path, h5mu, tmp_path / "out", VIEWS)

    main(["validate", "--config", str(config)])

    printed = capsys.readouterr().out
    assert "Config OK" in printed
    assert str(h5mu) in printed


def test_validate_does_not_train_or_write(make_h5mu_file, make_config_file, tmp_path):
    """The point of validate: it inspects, it does not run anything."""
    h5mu = make_h5mu_file(tmp_path, n_samples=16, view_specs=VIEWS)
    output_dir = tmp_path / "untouched"
    config = make_config_file(tmp_path, h5mu, output_dir, VIEWS)

    main(["validate", "--config", str(config)])

    assert not output_dir.exists()


# Completeness


def test_every_subcommand_has_a_success_path_test(capsys):
    """Fails when a subcommand is added to the CLI without a test here.

    Reads the parser's own subcommand list rather than a copy of it, so the
    check cannot drift from the CLI it guards.
    """
    with pytest.raises(SystemExit):
        main(["--help"])

    usage = capsys.readouterr().out
    listed = re.search(r"\{([a-z,\-]+)\}", usage)
    assert listed, f"could not read subcommands from help output:\n{usage}"
    commands = set(listed.group(1).split(","))

    assert commands == COVERED_COMMANDS, (
        f"CLI subcommands and covered commands differ: "
        f"untested={sorted(commands - COVERED_COMMANDS)}, "
        f"stale={sorted(COVERED_COMMANDS - commands)}"
    )
