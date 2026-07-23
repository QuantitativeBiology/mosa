from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from mosa.config import DataConfig, ModelConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.evaluation import cross_validate
from mosa.models.registry import register_model
from mosa.models.api import MultiOmicModel


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

    results = cross_validate(dataset, data_cfg, model_cfg, n_folds=3)

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


def test_cross_validate_too_many_folds_raises(make_multi_omic_dataset, make_mosa_config, tmp_path):
    dataset = make_multi_omic_dataset(n_samples=10, n_groups=2)
    data_cfg, model_cfg = make_mosa_config(dataset, output_dir=str(tmp_path))

    with pytest.raises(ValueError, match="exceeds the size of the smallest"):
        cross_validate(dataset, data_cfg, model_cfg, n_folds=10)


# Transductive (MOFA-like) models: no out-of-sample projection.


def _make_fake_transductive(name, raised_exc_cls, message):
    """Build a MultiOmicModel stand-in for a transductive model (e.g. MOFA).

    `raised_exc_cls` lets tests cover both failure shapes real MOFA can
    produce depending on how far it gets: NotImplementedError from
    transform()/reconstruct() directly (no out-of-sample projection), or
    RuntimeError if reconstruct() requires an explicit save_outputs() call
    that cross_validate never makes (see evaluation.py's fold-0 catch).
    """

    @dataclass
    class _FakeConfig(ModelConfig):
        pass

    @register_model(name, _FakeConfig)
    class _FakeModel(MultiOmicModel):
        def __init__(self, data_cfg, model_cfg):
            self.data_cfg = data_cfg
            self.model_cfg = model_cfg

        def fit(self, train, val=None, resume_from=None):
            pass

        def transform(self, data):
            raise raised_exc_cls(message)

        def reconstruct(self, data):
            raise raised_exc_cls(message)

        def save_outputs(self, output_dir=None):
            pass

        def save(self, path):
            pass

        @classmethod
        def load(cls, path, **kwargs):
            raise NotImplementedError

    return _FakeConfig


_FakeNotImplementedConfig = _make_fake_transductive(
    "fake_transductive_not_implemented", NotImplementedError, "no out-of-sample projection"
)
_FakeRuntimeErrorConfig = _make_fake_transductive(
    "fake_transductive_runtime_error", RuntimeError, "Model must be fit before calling reconstruct()"
)


@pytest.mark.parametrize(
    "config_cls", [_FakeNotImplementedConfig, _FakeRuntimeErrorConfig]
)
def test_cross_validate_transductive_model_raises_clear_error(
    config_cls, make_multi_omic_dataset, tmp_path
):
    dataset = make_multi_omic_dataset(n_samples=20, n_groups=2)
    data_cfg = DataConfig(path="unused", views=list(dataset.view_names))
    model_cfg = config_cls(output_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="Cross-validation is not supported"):
        cross_validate(dataset, data_cfg, model_cfg, n_folds=3)
