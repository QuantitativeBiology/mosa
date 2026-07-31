from __future__ import annotations

import pytest

import mosa.models.optimize as optimize_module
from mosa.config import EvaluationConfig
from mosa.models.optimize import optimize

optuna = pytest.importorskip("optuna")


def test_optimize_returns_best_params_and_value(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {
        "learning_rate": {"dist": "loguniform", "low": 1e-4, "high": 1e-2},
        "joint_latent_dim": {"dist": "categorical", "choices": [8, 16]},
    }

    results = optimize(dataset, data_cfg, model_cfg, search_space, n_trials=3, eval_cfg=EvaluationConfig(n_folds=2))

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

    results = optimize(dataset, data_cfg, model_cfg, search_space, n_trials=5, eval_cfg=EvaluationConfig(n_folds=2))

    study = results["study"]
    assert len(study.trials) == 5
    states = {t.state for t in study.trials}
    assert optuna.trial.TrialState.PRUNED in states or optuna.trial.TrialState.COMPLETE in states
    # No trial should have failed outright.
    assert optuna.trial.TrialState.FAIL not in states


def test_optimize_rejects_unknown_search_space_field(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {"learnign_rate": {"dist": "uniform", "low": 1e-4, "high": 1e-2}}

    with pytest.raises(ValueError, match="not a field of MOSAConfig"):
        optimize(dataset, data_cfg, model_cfg, search_space, n_trials=3)


def test_optimize_rejects_evaluation_fields_in_search_space(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    """The protocol scores the study, so tuning it would optimize the measurement."""
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {"n_folds": {"dist": "int", "low": 2, "high": 10}}

    with pytest.raises(ValueError, match="evaluation block"):
        optimize(dataset, data_cfg, model_cfg, search_space, n_trials=3)


def test_optimize_prunes_wrongly_typed_sampled_values(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    """A real field with the wrong value type raises TypeError in __post_init__."""
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {"joint_latent_dim": {"dist": "categorical", "choices": ["small", "large"]}}

    with pytest.raises(RuntimeError, match="No trial completed"):
        optimize(
            dataset, data_cfg, model_cfg, search_space, n_trials=3,
            eval_cfg=EvaluationConfig(n_folds=2),
        )


def test_optimize_prunes_trials_that_fail_during_training(
    make_multi_omic_dataset, make_mosa_config, tmp_path, monkeypatch
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {
        "joint_latent_dim": {"dist": "categorical", "choices": [8, 16]},
    }

    def flaky_cross_validate(dataset, data_cfg, trial_model_cfg, eval_cfg):
        if trial_model_cfg.joint_latent_dim == 8:
            raise RuntimeError("simulated training divergence")
        return {"mean": 0.5}

    monkeypatch.setattr(optimize_module, "cross_validate", flaky_cross_validate)

    results = optimize(dataset, data_cfg, model_cfg, search_space, n_trials=5, eval_cfg=EvaluationConfig(n_folds=2))

    study = results["study"]
    assert len(study.trials) == 5
    states = {t.state for t in study.trials}
    assert optuna.trial.TrialState.COMPLETE in states
    assert optuna.trial.TrialState.PRUNED in states
    assert optuna.trial.TrialState.FAIL not in states
    assert results["best_value"] == 0.5


def test_optimize_raises_when_every_trial_fails(
    make_multi_omic_dataset, make_mosa_config, tmp_path, monkeypatch
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, num_epochs=1, output_dir=str(tmp_path))

    search_space = {
        "joint_latent_dim": {"dist": "categorical", "choices": [8, 16]},
    }

    def always_fails(dataset, data_cfg, trial_model_cfg, eval_cfg):
        raise RuntimeError("simulated training divergence")

    monkeypatch.setattr(optimize_module, "cross_validate", always_fails)

    with pytest.raises(RuntimeError, match="No trial completed") as exc:
        optimize(dataset, data_cfg, model_cfg, search_space, n_trials=3, eval_cfg=EvaluationConfig(n_folds=2))

    # The per-trial cause is otherwise only in the logs, which the CLI hides
    # unless --debug is passed.
    assert "simulated training divergence" in str(exc.value)
