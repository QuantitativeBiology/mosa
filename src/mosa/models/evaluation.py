from __future__ import annotations

import dataclasses
import logging
import tempfile
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from mosa.config import DataConfig, EvaluationConfig, ModelConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.errors import DataError, UnsupportedError
from mosa.models.registry import build_model, model_class_for

logger = logging.getLogger(__name__)


def _score_fold(recon: dict[str, np.ndarray], val: MultiOmicDataset) -> dict:
    """Squared-error total and observed-entry count per view on a held-out fold.

    Takes the fold's reconstruction (already computed once by the caller, so
    it can also be reused to assemble out-of-sample reconstructions) rather
    than calling model.reconstruct() itself. Returns raw sums rather than a
    finished error: normalizing inside the fold would make each fold's score
    depend on which views that fold happened to observe. cross_validate pools
    these across folds first.
    """
    per_view = {}
    for view in val.view_names:
        mask = val.masks[view]
        n_obs = int(mask.sum())
        if n_obs == 0:
            per_view[view] = {"sse": 0.0, "n_obs": 0}
            continue

        error = recon[view][mask] - val.views[view][mask]
        per_view[view] = {"sse": float(np.sum(error**2)), "n_obs": n_obs}

    return per_view


def _observed_variance(dataset: MultiOmicDataset, view: str) -> float:
    """Variance of a view's observed entries over the whole dataset.

    One normalizer per view for every fold, so per-fold errors are comparable
    to each other and to the pooled total.
    """
    mask = dataset.masks[view]
    if mask.sum() == 0:
        return 0.0
    return float(np.var(dataset.views[view][mask]))


def _check_folds_fit_data(
    dataset: MultiOmicDataset, labels: np.ndarray, eval_cfg: EvaluationConfig
) -> None:
    """Reject fold counts the data cannot support, per strategy."""
    n_folds = eval_cfg.n_folds
    if eval_cfg.strategy == "stratified":
        classes, counts = np.unique(labels, return_counts=True)
        if counts.min() < n_folds:
            smallest = classes[np.argmin(counts)]
            raise DataError(
                f"n_folds={n_folds} exceeds the size of the smallest model_type "
                f"class ('{smallest}', {counts.min()} samples); reduce n_folds, "
                f"add more samples for that class, or use strategy='kfold'."
            )
    elif dataset.n_samples < n_folds:
        raise DataError(
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


def require_out_of_sample(model_cfg: ModelConfig, action: str) -> None:
    """Raise before any training when model_cfg's model cannot score held-out samples."""
    model_cls = model_class_for(model_cfg)
    if not model_cls.supports_out_of_sample:
        raise UnsupportedError(
            f"{action} is not supported for the "
            f"'{getattr(model_cls, 'registered_name', model_cls.__name__)}' "
            f"model: it has no out-of-sample projection."
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

    "mean" is the pooled estimate: squared errors are summed across folds per
    view, divided by the total observed entries for that view, normalized by
    that view's whole-dataset variance, then averaged over views. Because the
    folds partition the data, this scores every observed entry exactly once,
    by a model that never trained on it. Per-fold aggregates are reported for
    diagnostics only; a fold that observes no entries for a view covers fewer
    views than its neighbours, so "std" is a spread indicator rather than an
    uncertainty on "mean".

    Each per_fold entry also carries "epoch_history": one dict per training
    epoch with "epoch", "train_loss", and "val_loss" (val/loss is the same
    metric EarlyStopping monitors for that fold), so callers can average
    curves across folds and plot train vs. val loss per epoch. History is
    empty for a fold whose model has no per-epoch training loop.

    The top-level "reconstructions" dict holds one out-of-sample
    reconstruction array per view, aligned to `dataset`'s original sample
    order: since folds partition the data, concatenating each fold's
    held-out reconstruction covers every sample exactly once, ready to
    compare directly against `dataset.views[view]` (e.g. MSE, Pearson r)
    without any extra bookkeeping. Entries for samples missing from a view
    (per dataset.masks) reconstruct to whatever the model predicts there and
    should be filtered with dataset.masks[view] before comparing.
    """
    eval_cfg = eval_cfg or EvaluationConfig()
    require_out_of_sample(model_cfg, "Cross-validation")

    labels = dataset.metadata["model_type"].to_numpy()
    _check_folds_fit_data(dataset, labels, eval_cfg)

    splitter = _build_splitter(eval_cfg, getattr(model_cfg, "random_seed", 42))

    # Out-of-sample reconstruction per view, assembled fold by fold. Folds
    # partition the dataset, so once every fold has run this covers every
    # sample exactly once, each reconstructed by a model that never trained
    # on it -- ready to compare against dataset.views directly (MSE, Pearson).
    reconstructions = {
        v: np.full_like(dataset.views[v], np.nan) for v in dataset.view_names
    }

    fold_totals = []
    fold_histories = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(np.arange(dataset.n_samples), labels)
    ):
        train = dataset.subset(train_idx)
        val = dataset.subset(val_idx)

        with tempfile.TemporaryDirectory(prefix=f"mosa_cv_fold{fold_idx}_") as tmp_dir:
            fold_overrides: dict[str, Any] = {"output_dir": tmp_dir}
            if hasattr(model_cfg, "checkpoint_top_k"):
                fold_overrides["checkpoint_top_k"] = 0
            fold_model_cfg = dataclasses.replace(model_cfg, **fold_overrides)

            model = build_model(data_cfg, fold_model_cfg)
            model.fit(train, val)
            recon = model.reconstruct(val)
            totals = _score_fold(recon, val)
            fold_histories.append(list(model.epoch_history))

        for view, values in recon.items():
            reconstructions[view][val_idx] = values

        unobserved = [v for v in dataset.view_names if totals[v]["n_obs"] == 0]
        if unobserved:
            logger.warning(
                "Fold %d has no observed entries for view(s) %s; this split "
                "cannot assess %s. Consider strategy='stratified' or shuffle=True.",
                fold_idx,
                unobserved,
                "them" if len(unobserved) > 1 else "it",
            )
        fold_totals.append(totals)

    variances = {v: _observed_variance(dataset, v) for v in dataset.view_names}

    pooled = {}
    for view in dataset.view_names:
        n_obs = sum(f[view]["n_obs"] for f in fold_totals)
        if n_obs == 0:
            raise DataError(
                f"View '{view}' has no observed entries in any fold, so it "
                f"cannot be scored. Drop it from data.views or check its mask layer."
            )
        if variances[view] == 0:
            raise DataError(
                f"View '{view}' has zero variance across its observed entries, "
                f"so variance-normalized error is undefined. Drop it from data.views."
            )
        mse = sum(f[view]["sse"] for f in fold_totals) / n_obs
        pooled[view] = {"mse": mse, "nmse": mse / variances[view], "n_obs": n_obs}

    aggregate = float(np.mean([pooled[v]["nmse"] for v in dataset.view_names]))

    per_fold = []
    for fold_idx, totals in enumerate(fold_totals):
        per_view = {}
        for view in dataset.view_names:
            n_obs = totals[view]["n_obs"]
            if n_obs == 0:
                per_view[view] = {"mse": float("nan"), "nmse": float("nan"), "n_obs": 0}
                continue
            mse = totals[view]["sse"] / n_obs
            per_view[view] = {
                "mse": mse,
                "nmse": mse / variances[view],
                "n_obs": n_obs,
            }
        covered = [v for v in dataset.view_names if per_view[v]["n_obs"] > 0]
        per_fold.append(
            {
                "per_view": per_view,
                "aggregate": float(np.mean([per_view[v]["nmse"] for v in covered])),
                "n_views": len(covered),
                "epoch_history": fold_histories[fold_idx],
            }
        )
        logger.debug(
            "Fold %d: aggregate=%.4f over %d/%d views",
            fold_idx,
            per_fold[-1]["aggregate"],
            len(covered),
            len(dataset.view_names),
        )

    return {
        "per_fold": per_fold,
        "per_view": pooled,
        "mean": aggregate,
        "std": float(np.std([f["aggregate"] for f in per_fold])),
        "reconstructions": reconstructions,
    }
