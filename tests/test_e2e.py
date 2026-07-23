from __future__ import annotations

"""End-to-end tests for the MOSA CLI.

These tests exercise the full pipeline a user runs: a config YAML file + a
data file as inputs, the `mosa train` CLI as the entry point, and output
parquet files as the observable result. No internal imports — the CLI is
invoked via subprocess exactly as a user would call it.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# Helpers


def _write_config(path: Path, data_path: Path, output_dir: Path, view_specs: dict[str, int]) -> Path:
    config = {
        "data": {
            "path": str(data_path),
            "views": list(view_specs),
        },
        "model": {
            "type": "mosa_vae",
            "output_dir": str(output_dir),
            "test_size": 0.0,
            "joint_latent_dim": 8,
            "num_epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "weighted_random_sampler": False,
            "views": {
                name: {"hidden_layer_dims": [16, 8]}
                for name in view_specs
            },
            "accelerator": "cpu",
            "devices": 1,
            "precision": "32",
        },
    }
    config_path = path / "config.yaml"
    config_path.write_text(yaml.dump(config))
    return config_path


# Tests


def test_cli_train_creates_output_files(make_h5mu_file, tmp_path):
    """mosa train writes latent.parquet and recon_*.parquet to output_dir/train/."""
    view_specs = {"view_a": 10, "view_b": 8}
    data_path = make_h5mu_file(tmp_path, n_samples=20, view_specs=view_specs)
    output_dir = tmp_path / "outputs"
    config_path = _write_config(tmp_path, data_path, output_dir, view_specs)

    result = subprocess.run(
        [sys.executable, "-m", "mosa.cli", "--help"],
        capture_output=True,
        text=True,
    )
    # Use the installed entry point
    result = subprocess.run(
        [sys.executable, "-c",
         f"from mosa.cli import main; import sys; sys.argv = ['mosa', 'train', '--config', '{config_path}']; main()"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"CLI exited with code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    assert (output_dir / "train" / "latent.parquet").exists()
    for name in view_specs:
        assert (output_dir / "train" / f"recon_{name}.parquet").exists()


def test_cli_train_output_shapes(make_h5mu_file, tmp_path):
    """Latent matrix has shape (n_samples, joint_latent_dim); recon matches input dims."""
    view_specs = {"view_a": 10, "view_b": 8}
    n_samples = 20
    joint_latent_dim = 8
    data_path = make_h5mu_file(tmp_path, n_samples=n_samples, view_specs=view_specs)
    output_dir = tmp_path / "outputs"
    config_path = _write_config(tmp_path, data_path, output_dir, view_specs)

    result = subprocess.run(
        [sys.executable, "-c",
         f"from mosa.cli import main; import sys; sys.argv = ['mosa', 'train', '--config', '{config_path}']; main()"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    latent = pd.read_parquet(output_dir / "train" / "latent.parquet")
    assert latent.shape == (n_samples, joint_latent_dim)
    assert not latent.isna().any().any()

    for name, n_features in view_specs.items():
        recon = pd.read_parquet(output_dir / "train" / f"recon_{name}.parquet")
        assert recon.shape == (n_samples, n_features)
        assert not recon.isna().any().any()
