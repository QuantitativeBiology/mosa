from __future__ import annotations

import pytest

from mosa.models.optimize import optimize

optuna = pytest.importorskip("optuna")


def test_optimize_returns_best_params_and_value(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {
        "learning_rate": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
        "joint_latent_dim": {"dist": "categorical", "choices": [8, 16]},
    }

    results = optimize(dataset, data_cfg, model_cfg, search_space, n_trials=3, n_folds=2)

    assert "best_params" in results and "best_value" in results and "study" in results
    assert isinstance(results["best_value"], float)
    assert set(results["best_params"].keys()) <= set(search_space.keys())


def test_optimize_prunes_invalid_configs_instead_of_crashing(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=1, output_dir=str(tmp_path), fusion_method="poe",
    )

    # joint_latent_dim <= 0 is rejected by MOSAConfig.__post_init__, so any
    # trial sampling the low end of this range must be pruned, not crash
    # the whole study.
    search_space = {
        "joint_latent_dim": {"dist": "int", "low": -5, "high": 5},
    }

    results = optimize(dataset, data_cfg, model_cfg, search_space, n_trials=5, n_folds=2)

    study = results["study"]
    assert len(study.trials) == 5
    states = {t.state for t in study.trials}
    assert optuna.trial.TrialState.PRUNED in states or optuna.trial.TrialState.COMPLETE in states
    # No trial should have failed outright.
    assert optuna.trial.TrialState.FAIL not in states
