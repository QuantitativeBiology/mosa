from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from mosa.config import DataConfig, ModelConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.evaluation import cross_validate
from mosa.utils import read_yaml

logger = logging.getLogger(__name__)

_SUPPORTED_DISTS = ("loguniform", "uniform", "int", "categorical")


def load_search_space(path: str | Path) -> dict:
    """Load and validate a search-space YAML mapping top-level ModelConfig fields to distributions."""
    return parse_search_space(read_yaml(path))


def _suggest(trial, name: str, spec: dict):
    """Turn one search-space entry into a trial.suggest_* call.

    Scoped to top-level ModelConfig fields only; nested per-view params
    (e.g. views.<name>.hidden_layer_dims) are out of scope.
    """
    dist = spec.get("dist")
    if dist == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if dist == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if dist == "int":
        return trial.suggest_int(name, spec["low"], spec["high"])
    if dist == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(
        f"Unknown dist '{dist}' for search-space entry '{name}'; "
        f"must be one of {_SUPPORTED_DISTS}"
    )


def parse_search_space(raw: dict) -> dict:
    """Validate a raw search-space mapping of names to {dist, ...} specs.

    Only checks structure (each entry has a known 'dist' and its required
    keys); actual sampling happens per-trial in `_suggest`.
    """
    for name, spec in raw.items():
        if not isinstance(spec, dict) or "dist" not in spec:
            raise ValueError(f"Search-space entry '{name}' must be a mapping with a 'dist' key")
        dist = spec["dist"]
        if dist not in _SUPPORTED_DISTS:
            raise ValueError(
                f"Search-space entry '{name}': dist must be one of {_SUPPORTED_DISTS}, got '{dist}'"
            )
        required = {"categorical": ("choices",)}.get(dist, ("low", "high"))
        missing = [k for k in required if k not in spec]
        if missing:
            raise ValueError(f"Search-space entry '{name}' missing key(s): {missing}")
    return raw


def optimize(
    dataset: MultiOmicDataset,
    data_cfg: DataConfig,
    base_model_cfg: ModelConfig,
    search_space: dict,
    n_trials: int,
    n_folds: int = 3,
):
    """Optuna hyperparameter search over top-level ModelConfig fields, scored via cross_validate.

    Each trial samples values per `search_space`, builds a mutated config with
    `dataclasses.replace(base_model_cfg, **sampled)`, and scores it with
    `cross_validate(..., n_folds)` (lower is better). If the sampled combo is
    invalid, the config's own __post_init__ raises ValueError; this is caught
    and turned into optuna.TrialPruned so the trial is pruned instead of
    crashing the whole study. Any exception raised during cross_validate
    itself is caught the same way, with the full traceback logged, so one
    failing trial does not abort the remaining trials.

    Requires optuna (`pip install '.[hpo]'`); imported lazily so importing
    mosa never requires it.

    Returns {"best_params": dict, "best_value": float, "study": optuna.Study}.
    """
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "optuna is required for optimize(); install it with pip install '.[hpo]'"
        ) from e

    search_space = parse_search_space(search_space)

    def objective(trial: "optuna.Trial") -> float:
        sampled = {name: _suggest(trial, name, spec) for name, spec in search_space.items()}
        try:
            trial_model_cfg = dataclasses.replace(base_model_cfg, **sampled)
        except ValueError as e:
            logger.debug("Trial %d pruned: invalid config (%s)", trial.number, e)
            raise optuna.TrialPruned(str(e)) from e

        try:
            result = cross_validate(dataset, data_cfg, trial_model_cfg, n_folds=n_folds)
        except Exception as e:
            logger.exception("Trial %d failed", trial.number)
            raise optuna.TrialPruned(f"trial {trial.number} failed: {e}") from e
        return result["mean"]

    # Seed the sampler from the config so a search is reproducible run to run.
    seed = getattr(base_model_cfg, "random_seed", 42)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials)

    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        raise RuntimeError(f"No trial completed out of {n_trials}; see logged tracebacks")

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
    }
