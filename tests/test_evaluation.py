from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mosa.config import DataConfig, EvaluationConfig, ModelConfig
from mosa.models.api import MultiOmicModel
from mosa.models.evaluation import cross_validate
from mosa.models.registry import register_model


def _iter_files(root):
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def test_cross_validate_returns_per_fold_and_summary(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=30, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=1, output_dir=str(tmp_path)
    )

    results = cross_validate(dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=3))

    assert len(results["per_fold"]) == 3
    for fold in results["per_fold"]:
        assert "per_view" in fold and "aggregate" in fold
        for view in dataset.view_names:
            assert view in fold["per_view"]
            assert "mse" in fold["per_view"][view]
            assert "nmse" in fold["per_view"][view]
    assert isinstance(results["mean"], float)
    assert isinstance(results["std"], float)

    # No artifacts written into the caller's output_dir (or anywhere else the
    # config points at) — each fold must use its own throwaway temp dir.
    written = list(_iter_files(tmp_path))
    assert not written, f"cross_validate wrote unexpected files: {written}"


def test_cross_validate_reports_epoch_history_and_reconstructions(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=30, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=2, output_dir=str(tmp_path)
    )

    results = cross_validate(dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=3))

    for fold in results["per_fold"]:
        history = fold["epoch_history"]
        assert len(history) > 0
        for entry in history:
            assert "epoch" in entry and "train_loss" in entry and "val_loss" in entry

    assert set(results["reconstructions"]) == set(dataset.view_names)
    for view, recon in results["reconstructions"].items():
        assert recon.shape == dataset.views[view].shape
        # Every fold contributes disjoint rows, so the whole dataset ends up
        # covered (no leftover NaN placeholders from the preallocation).
        assert not np.isnan(recon).any()


def test_cross_validate_too_many_folds_raises(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=10, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="exceeds the size of the smallest"):
        cross_validate(dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=10))


def test_kfold_guard_bounded_by_sample_count_not_class_size(make_multi_omic_dataset):
    """A class smaller than n_folds blocks stratified splitting but not plain kfold."""
    from mosa.models.evaluation import _check_folds_fit_data

    dataset = make_multi_omic_dataset(n_samples=12, n_groups=2)
    labels = np.array(["rare"] + ["common"] * 11)

    with pytest.raises(ValueError, match="exceeds the size of the smallest"):
        _check_folds_fit_data(dataset, labels, EvaluationConfig(n_folds=3))

    _check_folds_fit_data(
        dataset, labels, EvaluationConfig(n_folds=3, strategy="kfold")
    )

    with pytest.raises(ValueError, match="exceeds the number of samples"):
        _check_folds_fit_data(
            dataset, labels, EvaluationConfig(n_folds=20, strategy="kfold")
        )


def test_cross_validate_kfold_strategy_runs(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=30, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=1, output_dir=str(tmp_path)
    )

    results = cross_validate(
        dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=3, strategy="kfold")
    )
    assert len(results["per_fold"]) == 3


@pytest.mark.parametrize("strategy", ["stratified", "kfold"])
def test_unshuffled_folds_are_contiguous_blocks(make_multi_omic_dataset, strategy):
    """shuffle=False assigns folds by sample order, so it is seed-independent."""
    from mosa.models.evaluation import _build_splitter

    dataset = make_multi_omic_dataset(n_samples=12, n_groups=2)
    labels = dataset.metadata["model_type"].to_numpy()
    indices = np.arange(dataset.n_samples)

    eval_cfg = EvaluationConfig(n_folds=3, strategy=strategy, shuffle=False)
    a = [
        v.tolist() for _, v in _build_splitter(eval_cfg, seed=1).split(indices, labels)
    ]
    b = [
        v.tolist()
        for _, v in _build_splitter(eval_cfg, seed=999).split(indices, labels)
    ]
    assert a == b

    shuffled = EvaluationConfig(n_folds=3, strategy=strategy, shuffle=True)
    c = [
        v.tolist() for _, v in _build_splitter(shuffled, seed=1).split(indices, labels)
    ]
    assert c != a


def test_aggregate_covers_every_view_when_a_fold_observes_none(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    """A view unobserved in some folds must not shrink the aggregate's denominator.

    Scoring each fold on its own would average 2 views in the starved folds and
    3 in the rest, so the reported mean would mix incompatible quantities.
    Pooling squared errors across folds before normalizing keeps one view set.
    """
    dataset = make_multi_omic_dataset(
        n_samples=30, n_groups=2, view_dims={"dense_a": 12, "dense_b": 10, "sparse": 8}
    )
    # Observed only in the first third, which unshuffled folds put entirely in
    # fold 0; folds 1 and 2 then hold no sparse entries at all.
    dataset.masks["sparse"][10:] = False

    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=1, output_dir=str(tmp_path)
    )
    results = cross_validate(
        dataset,
        data_cfg,
        model_cfg,
        EvaluationConfig(n_folds=3, strategy="kfold", shuffle=False),
    )

    assert set(results["per_view"]) == {"dense_a", "dense_b", "sparse"}
    for view, scores in results["per_view"].items():
        assert not np.isnan(scores["nmse"]), f"{view} pooled NMSE is NaN"
    assert results["per_view"]["sparse"]["n_obs"] == 10 * 8

    # The headline number averages all three views, not a per-fold subset.
    expected = np.mean([results["per_view"][v]["nmse"] for v in dataset.view_names])
    assert results["mean"] == pytest.approx(expected)
    assert not np.isnan(results["mean"])

    # The starved folds are still visible as such in the diagnostic table.
    coverage = [f["n_views"] for f in results["per_fold"]]
    assert coverage == [3, 2, 2]
    for fold in results["per_fold"][1:]:
        assert np.isnan(fold["per_view"]["sparse"]["nmse"])
        assert fold["per_view"]["sparse"]["n_obs"] == 0


def test_unscoreable_view_raises_instead_of_reporting_nan(
    make_multi_omic_dataset, make_mosa_config, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    dataset.masks["view_b"][:] = False

    data_cfg, model_cfg = make_mosa_config(
        dataset, num_epochs=1, output_dir=str(tmp_path)
    )
    with pytest.raises(ValueError, match="no observed entries in any fold"):
        cross_validate(dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=2))


# Transductive (MOFA-like) models: no out-of-sample projection.


@dataclass
class _FakeTransductiveConfig(ModelConfig):
    pass


@register_model("fake_transductive", _FakeTransductiveConfig)
class _FakeTransductiveModel(MultiOmicModel):
    """Stand-in for a transductive model (e.g. MOFA), which cannot be cross-validated."""

    supports_out_of_sample = False

    def __init__(self, data_cfg, model_cfg):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg

    def fit(self, train, val=None, resume_from=None):
        raise AssertionError("cross_validate must reject this model before fitting")

    def transform(self, data):
        raise NotImplementedError("no out-of-sample projection")

    def reconstruct(self, data):
        raise NotImplementedError("no out-of-sample projection")

    def save_outputs(self, output_dir=None):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path, **kwargs):
        raise NotImplementedError


def test_cross_validate_transductive_model_raises_clear_error(
    make_multi_omic_dataset, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg = DataConfig(path="unused", views=list(dataset.view_names))
    model_cfg = _FakeTransductiveConfig(output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="Cross-validation is not supported"):
        cross_validate(dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=3))


def test_mofa_declares_no_out_of_sample_support():
    pytest.importorskip("mofapy2")
    from mosa.models.mofa.model import MOFAModel

    assert MOFAModel.supports_out_of_sample is False
