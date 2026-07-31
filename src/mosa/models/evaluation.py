from __future__ import annotations

import dataclasses
import logging
import tempfile

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from mosa.config import DataConfig, EvaluationConfig, ModelConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.registry import build_model, model_class_for

logger = logging.getLogger(__name__)


def _score_fold(model, val: MultiOmicDataset) -> dict:
    """Score a fitted model on a held-out fold via masked, variance-normalized MSE.

    Uses only the ABC's reconstruct(), so this works for any model implementing
    the interface. Per view: MSE over observed entries (val.masks), normalized
    by the variance of the observed targets (fraction of variance unexplained)
    so views on different scales are comparable. Aggregate is the mean of the
    per-view normalized errors across views with nonzero variance.
    """
    recon = model.reconstruct(val)

    per_view = {}
    for view in val.view_names:
        mask = val.masks[view]
        target = val.views[view]
        pred = recon[view]

        if mask.sum() == 0:
            per_view[view] = {"mse": float("nan"), "nmse": float("nan")}
            continue

        observed_target = target[mask]
        observed_pred = pred[mask]
        mse = float(np.mean((observed_pred - observed_target) ** 2))
        var = float(np.var(observed_target))
        nmse = mse / var if var > 0 else float("nan")
        per_view[view] = {"mse": mse, "nmse": nmse}

    valid_nmse = [v["nmse"] for v in per_view.values() if not np.isnan(v["nmse"])]
    aggregate = float(np.mean(valid_nmse)) if valid_nmse else float("nan")
    return {"per_view": per_view, "aggregate": aggregate}


def _check_folds_fit_data(
    dataset: MultiOmicDataset, labels: np.ndarray, eval_cfg: EvaluationConfig
) -> None:
    """Reject fold counts the data cannot support, per strategy."""
    n_folds = eval_cfg.n_folds
    if eval_cfg.strategy == "stratified":
        classes, counts = np.unique(labels, return_counts=True)
        if counts.min() < n_folds:
            smallest = classes[np.argmin(counts)]
            raise ValueError(
                f"n_folds={n_folds} exceeds the size of the smallest model_type "
                f"class ('{smallest}', {counts.min()} samples); reduce n_folds, "
                f"add more samples for that class, or use strategy='kfold'."
            )
    elif dataset.n_samples < n_folds:
        raise ValueError(
            f"n_folds={n_folds} exceeds the number of samples "
            f"({dataset.n_samples}); reduce n_folds."
        )


def _build_splitter(eval_cfg: EvaluationConfig, seed: int):
    """Construct the sklearn splitter for the configured strategy."""
    cls = StratifiedKFold if eval_cfg.strategy == "stratified" else KFold
    # sklearn rejects random_state outright when shuffle is False.
    return cls(
        n_splits=eval_cfg.n_folds,
        shuffle=eval_cfg.shuffle,
        random_state=seed if eval_cfg.shuffle else None,
    )


def cross_validate(
    dataset: MultiOmicDataset,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    eval_cfg: EvaluationConfig | None = None,
) -> dict:
    """K-fold cross-validation, scored by masked variance-normalized MSE.

    `eval_cfg.strategy` selects the splitter: "stratified" balances
    dataset.metadata["model_type"] across folds, "kfold" ignores it. With
    `eval_cfg.shuffle` False the folds are contiguous blocks of the dataset's
    sample order (sorted by sample ID, per align_views) and model_cfg's seed
    no longer affects fold composition.

    Each fold trains a fresh model from scratch (build_model + fit) and scores
    it on the held-out fold with reconstruct(). No artifacts are written:
    each fold's model_cfg is a copy with checkpoint_top_k forced to 0 (where
    the field exists) and output_dir pointed at a unique, auto-cleaned temp
    directory, so folds never write to the caller's output_dir and never
    clobber each other.
    """
    eval_cfg = eval_cfg or EvaluationConfig()

    model_cls = model_class_for(model_cfg)
    if not model_cls.supports_out_of_sample:
        raise RuntimeError(
            f"Cross-validation is not supported for the "
            f"'{getattr(model_cls, 'registered_name', model_cls.__name__)}' "
            f"model: it has no out-of-sample projection."
        )

    labels = dataset.metadata["model_type"].to_numpy()
    _check_folds_fit_data(dataset, labels, eval_cfg)

    splitter = _build_splitter(eval_cfg, getattr(model_cfg, "random_seed", 42))

    per_fold = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(np.arange(dataset.n_samples), labels)
    ):
        train = dataset.subset(train_idx)
        val = dataset.subset(val_idx)

        with tempfile.TemporaryDirectory(prefix=f"mosa_cv_fold{fold_idx}_") as tmp_dir:
            fold_overrides = {"output_dir": tmp_dir}
            if hasattr(model_cfg, "checkpoint_top_k"):
                fold_overrides["checkpoint_top_k"] = 0
            fold_model_cfg = dataclasses.replace(model_cfg, **fold_overrides)

            model = build_model(data_cfg, fold_model_cfg)
            model.fit(train, val)
            score = _score_fold(model, val)

        logger.debug("Fold %d: aggregate=%.4f", fold_idx, score["aggregate"])
        per_fold.append(score)

    aggregates = np.array([f["aggregate"] for f in per_fold])
    return {
        "per_fold": per_fold,
        "mean": float(np.mean(aggregates)),
        "std": float(np.std(aggregates)),
    }
