from __future__ import annotations

import dataclasses
import logging
import tempfile

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mosa.config import DataConfig, ModelConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.registry import build_model

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


def cross_validate(
    dataset: MultiOmicDataset,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    n_folds: int = 5,
) -> dict:
    """Stratified k-fold cross-validation, scored by masked variance-normalized MSE.

    Folds are stratified on dataset.metadata["model_type"] via StratifiedKFold.
    Each fold trains a fresh model from scratch (build_model + fit) and scores
    it on the held-out fold with reconstruct(). No artifacts are written:
    each fold's model_cfg is a copy with checkpoint_top_k forced to 0 (where
    the field exists) and output_dir pointed at a unique, auto-cleaned temp
    directory, so folds never write to the caller's output_dir and never
    clobber each other.

    Models with no out-of-sample projection raise NotImplementedError from
    reconstruct(); this is caught and re-raised as a clear error on the first
    fold, before the remaining folds are trained.
    """
    labels = dataset.metadata["model_type"].to_numpy()
    classes, counts = np.unique(labels, return_counts=True)
    if counts.min() < n_folds:
        smallest = classes[np.argmin(counts)]
        raise ValueError(
            f"n_folds={n_folds} exceeds the size of the smallest model_type "
            f"class ('{smallest}', {counts.min()} samples); reduce n_folds or "
            f"add more samples for that class."
        )

    seed = getattr(model_cfg, "random_seed", 42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    per_fold = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.arange(dataset.n_samples), labels)
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

            try:
                score = _score_fold(model, val)
            except (NotImplementedError, RuntimeError) as e:
                # Transductive models reject held-out scoring either via
                # NotImplementedError (no out-of-sample projection) or
                # RuntimeError (requires save_outputs() first, which
                # cross_validate never calls). Only fold 0 is wrapped this
                # broadly; later folds' RuntimeErrors propagate as real errors.
                if fold_idx != 0:
                    raise
                name = getattr(model, "registered_name", type(model).__name__)
                raise RuntimeError(
                    f"Cross-validation is not supported for the '{name}' "
                    f"model — it has no out-of-sample projection."
                ) from e

        logger.debug("Fold %d: aggregate=%.4f", fold_idx, score["aggregate"])
        per_fold.append(score)

    aggregates = np.array([f["aggregate"] for f in per_fold])
    return {
        "per_fold": per_fold,
        "mean": float(np.mean(aggregates)),
        "std": float(np.std(aggregates)),
    }
