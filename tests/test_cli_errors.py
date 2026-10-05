"""CLI error-presentation contract.

Every user-fixable failure must reach the terminal as one actionable line on
stderr with exit code 1, and never as a traceback. The "no Traceback" assertion
is the one that catches regressions: before the CLI boundary existed, every
case here still failed, just unreadably.
"""

from __future__ import annotations

import sys

import pytest

from mosa.cli import main

MOSA_VAE_MODEL = (
    "model:\n"
    "  type: mosa_vae\n"
    "  num_epochs: 1\n"
    "  views:\n"
    "    view_a:\n"
    "      hidden_layer_dims: [16, 8]\n"
    "    view_b:\n"
    "      hidden_layer_dims: [16, 8]\n"
)


def _write(path, text):
    path.write_text(text)
    return str(path)


def _config(tmp_path, data_path, name="c.yaml", model=MOSA_VAE_MODEL, evaluation=""):
    return _write(
        tmp_path / name,
        f"data:\n  path: {data_path}\n  views: [view_a, view_b]\n{model}{evaluation}",
    )


def run_cli(argv, capsys):
    """Invoke the CLI and return (exit_code, stdout, stderr)."""
    with pytest.raises(SystemExit) as exc:
        main(argv)
    captured = capsys.readouterr()
    return exc.value.code, captured.out, captured.err


def assert_guided_failure(capsys, argv, expected):
    """Assert argv fails with exit 1 and `expected` on stderr, with no traceback."""
    code, out, err = run_cli(argv, capsys)

    assert code == 1, f"expected exit 1, got {code}. stderr:\n{err}"
    assert expected in err, f"expected {expected!r} in stderr, got:\n{err}"
    assert err.startswith("Error: "), f"stderr should lead with 'Error: ', got:\n{err}"
    assert "Traceback" not in err + out, f"traceback leaked to the user:\n{err}{out}"


# --- config-level failures -------------------------------------------------


def test_missing_data_path(tmp_path, capsys):
    cfg = _config(tmp_path, "does/not/exist.h5mu")
    assert_guided_failure(capsys, ["train", "--config", cfg], "data.path not found")


def test_malformed_yaml(tmp_path, capsys):
    cfg = _write(tmp_path / "bad.yaml", "data:\n  path: x.h5mu\n   views: [view_a]\n")
    assert_guided_failure(capsys, ["validate", "--config", cfg], "Invalid YAML in")


def test_missing_config_file(tmp_path, capsys):
    assert_guided_failure(
        capsys,
        ["validate", "--config", str(tmp_path / "nope.yaml")],
        "File not found",
    )


def test_typo_in_model_key_suggests_correction(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path)
    cfg = _config(tmp_path, h5mu, model=MOSA_VAE_MODEL + "  laten_dim: 32\n")
    assert_guided_failure(
        capsys, ["validate", "--config", cfg], "Did you mean 'joint_latent_dim'?"
    )


def test_typo_in_per_view_key_names_the_view(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path)
    model = MOSA_VAE_MODEL.replace(
        "    view_b:\n      hidden_layer_dims: [16, 8]\n",
        "    view_b:\n      hidden_layer_dims: [16, 8]\n      dropout: 0.5\n",
    )
    cfg = _config(tmp_path, h5mu, model=model)
    assert_guided_failure(capsys, ["validate", "--config", cfg], "model.views.view_b")


def test_view_name_key_is_rejected_not_crashed(tmp_path, capsys, make_h5mu_file):
    """'name' is injected per view, so setting it in YAML must be a guided error
    rather than a TypeError from the OmicViewConfig constructor."""
    h5mu = make_h5mu_file(tmp_path)
    model = MOSA_VAE_MODEL.replace(
        "    view_a:\n      hidden_layer_dims: [16, 8]\n",
        "    view_a:\n      name: something\n      hidden_layer_dims: [16, 8]\n",
    )
    cfg = _config(tmp_path, h5mu, model=model)
    assert_guided_failure(
        capsys,
        ["validate", "--config", cfg],
        "Unknown key 'name' in model.views.view_a",
    )


def test_unknown_model_type(tmp_path, capsys):
    cfg = _config(tmp_path, "x.h5mu", model="model:\n  type: not_a_model\n")
    assert_guided_failure(
        capsys, ["validate", "--config", cfg], "unknown model.type 'not_a_model'"
    )


def test_view_absent_from_data(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path, view_specs={"view_a": 10})
    cfg = _config(tmp_path, h5mu)
    assert_guided_failure(
        capsys, ["validate", "--config", cfg], "View 'view_b' not in MuData"
    )


def test_bad_n_folds(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path)
    cfg = _config(tmp_path, h5mu, evaluation="evaluation:\n  n_folds: 1\n")
    assert_guided_failure(
        capsys, ["cross-validate", "--config", cfg], "n_folds must be at least 2"
    )


def test_n_folds_exceeds_smallest_class(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path, n_samples=6)
    cfg = _config(tmp_path, h5mu)
    assert_guided_failure(
        capsys,
        ["cross-validate", "--config", cfg, "--folds", "5"],
        "exceeds the size of the smallest model_type",
    )


# --- command-level failures ------------------------------------------------


def test_transform_missing_checkpoint(tmp_path, capsys):
    assert_guided_failure(
        capsys,
        [
            "transform",
            "--checkpoint",
            str(tmp_path / "nope.ckpt"),
            "--input",
            str(tmp_path / "in.h5mu"),
            "--output",
            str(tmp_path / "out"),
        ],
        "Checkpoint not found",
    )


def test_inspect_missing_input(tmp_path, capsys):
    assert_guided_failure(
        capsys, ["inspect", "--input", str(tmp_path / "nope.h5mu")], "File not found"
    )


def test_convert_bad_view_spec(tmp_path, capsys):
    assert_guided_failure(
        capsys,
        [
            "convert",
            "--conditionals",
            str(tmp_path / "c.csv"),
            "--view",
            "no_colon_here",
            "--output",
            str(tmp_path / "o.h5mu"),
        ],
        "Invalid --view format",
    )


def test_convert_bad_filter_spec(tmp_path, capsys):
    assert_guided_failure(
        capsys,
        [
            "convert",
            "--conditionals",
            str(tmp_path / "c.csv"),
            "--view",
            "view_a:a.csv",
            "--filter",
            "no_equals_sign",
            "--output",
            str(tmp_path / "o.h5mu"),
        ],
        "Invalid --filter format",
    )


def test_optimize_bad_search_space(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path)
    cfg = _config(tmp_path, h5mu)
    space = _write(
        tmp_path / "space.yaml",
        "learning_rate:\n  dist: gaussian\n  low: 1\n  high: 2\n",
    )
    assert_guided_failure(
        capsys,
        ["optimize", "--config", cfg, "--search-space", space],
        "dist must be one of",
    )


def test_plot_without_training_artifacts(tmp_path, capsys, make_h5mu_file):
    h5mu = make_h5mu_file(tmp_path)
    out_dir = tmp_path / "empty_run"
    out_dir.mkdir()
    cfg = _config(tmp_path, h5mu)
    assert_guided_failure(
        capsys,
        ["plot", "--config", cfg, "--output-dir", str(out_dir)],
        "No latent representations found",
    )


def test_mofa_missing_extra_names_the_install(
    tmp_path, capsys, monkeypatch, make_h5mu_file
):
    """Simulated rather than ambient, so this holds whether or not .[mofa] is installed."""
    # A None entry in sys.modules makes find_spec report the module as absent.
    monkeypatch.setitem(sys.modules, "mofapy2", None)

    h5mu = make_h5mu_file(tmp_path)
    cfg = _config(tmp_path, h5mu, model="model:\n  type: mofa\n")
    assert_guided_failure(capsys, ["train", "--config", cfg], "pip install '.[mofa]'")


def test_cross_validate_rejects_model_without_out_of_sample(
    tmp_path, capsys, make_h5mu_file
):
    pytest.importorskip("mofapy2")
    h5mu = make_h5mu_file(tmp_path)
    cfg = _config(tmp_path, h5mu, model="model:\n  type: mofa\n")
    assert_guided_failure(
        capsys,
        ["cross-validate", "--config", cfg],
        "Cross-validation is not supported",
    )


def test_optimize_rejects_model_without_out_of_sample(
    tmp_path, capsys, make_h5mu_file, monkeypatch
):
    pytest.importorskip("mofapy2")
    pytest.importorskip("optuna")
    import optuna

    def no_study(*args, **kwargs):
        raise AssertionError("optimize created a study before rejecting the model")

    monkeypatch.setattr(optuna, "create_study", no_study)
    h5mu = make_h5mu_file(tmp_path)
    cfg = _config(tmp_path, h5mu, model="model:\n  type: mofa\n")
    space = _write(
        tmp_path / "space.yaml",
        "n_factors:\n  dist: int\n  low: 2\n  high: 5\n",
    )
    assert_guided_failure(
        capsys,
        ["optimize", "--config", cfg, "--search-space", space],
        "Hyperparameter search is not supported",
    )


# --- escape hatch ----------------------------------------------------------


def test_debug_flag_restores_traceback(tmp_path, capsys):
    cfg = _config(tmp_path, "does/not/exist.h5mu")
    code, _, err = run_cli(["train", "--config", cfg, "--debug"], capsys)

    assert code == 1
    assert "Error: data.path not found" in err
    assert "Traceback" in err


def test_keyboard_interrupt_exits_130(tmp_path, capsys, monkeypatch):
    """Ctrl-C outside Lightning. Inside fit() Lightning traps SIGINT and raises
    SystemExit(1) itself, so that path never reaches this handler."""
    import mosa.cli

    def interrupt(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(mosa.cli, "_inspect", interrupt)

    code, _, err = run_cli(["inspect", "--input", str(tmp_path / "x.h5mu")], capsys)
    assert code == 130
    assert "Interrupted" in err


def test_bug_keeps_its_traceback(tmp_path, capsys, monkeypatch):
    """A non-MosaError must not be swallowed: it is a bug, not user input."""
    import mosa.cli

    def boom(_args):
        raise AttributeError("'NoneType' object has no attribute 'shape'")

    monkeypatch.setattr(mosa.cli, "_inspect", boom)

    with pytest.raises(AttributeError):
        main(["inspect", "--input", str(tmp_path / "x.h5mu")])
