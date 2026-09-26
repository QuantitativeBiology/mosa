"""Registry and MultiOmicModel contract conformance.

Two halves:

  1. Conformance — every *registered* model is checked against the ABC and
     against the assumptions the layers above it make (build_model,
     cross_validate, optimize, load_model). Parametrized over the registry, so
     a model added later is checked without editing this file. That is the
     point: the previous interface test named MOSAModel and MOFAModel
     literally, so a third model could break every contract silently.

  2. Extensibility — a dummy model, registered only for the duration of a
     test, is driven through the full external workflow (build, fit,
     transform, reconstruct, save, load, cross-validate) using nothing but the
     public API. If it passes, "wrap an external model for comparison" works
     without touching src/.
"""

from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from mosa.config import DataConfig, EvaluationConfig, ModelConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.errors import UnsupportedError
from mosa.models.api import MultiOmicModel
from mosa.models.evaluation import cross_validate
from mosa.models.registry import (
    _REGISTRY,
    build_model,
    load_model,
    model_class_for,
    model_config_classes,
    register_model,
)

# The registry is private because model modules are its only writers. Tests
# read it to enumerate what is registered; there is no public equivalent that
# yields model classes without an instantiated config.
REGISTERED = sorted(_REGISTRY.items())
REGISTERED_NAMES = [name for name, _ in REGISTERED]

ABC_METHODS = ("fit", "transform", "reconstruct", "save_outputs", "save", "load")


@pytest.fixture(params=REGISTERED_NAMES)
def registration(request):
    return request.param, _REGISTRY[request.param]


# Conformance: every registered model


def test_registry_is_not_empty():
    """Guards the parametrized tests below: an empty registry would pass them all."""
    assert REGISTERED_NAMES, "no models registered; mosa.models imports nothing"


def test_model_class_subclasses_the_abc(registration):
    name, reg = registration
    assert issubclass(reg.model_cls, MultiOmicModel), (
        f"'{name}' is registered but is not a MultiOmicModel"
    )


def test_model_class_is_concrete(registration):
    """No abstract methods left unimplemented, so the class can be instantiated."""
    name, reg = registration
    missing = sorted(getattr(reg.model_cls, "__abstractmethods__", set()))
    assert not missing, f"'{name}' leaves {missing} unimplemented"


@pytest.mark.parametrize("method_name", ABC_METHODS)
def test_method_signature_matches_the_abc(registration, method_name):
    """Callers hold a MultiOmicModel; a renamed parameter breaks them at runtime."""
    name, reg = registration
    expected = _positional_params(getattr(MultiOmicModel, method_name))
    actual = _positional_params(getattr(reg.model_cls, method_name))

    # Extra trailing parameters are fine only if they are optional: the ABC's
    # callers never pass them.
    assert actual[: len(expected)] == expected, (
        f"'{name}'.{method_name}{tuple(actual)} does not match "
        f"MultiOmicModel.{method_name}{tuple(expected)}"
    )


def _positional_params(func) -> list[str]:
    """Parameter names of func, minus self/cls and minus *args/**kwargs."""
    params = list(inspect.signature(func).parameters.values())
    return [
        p.name
        for p in params
        if p.name not in ("self", "cls")
        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]


def test_model_class_is_constructible_from_build_model_signature(registration):
    """build_model calls model_cls(data_cfg, model_cfg); the class must accept that."""
    name, reg = registration
    params = _positional_params(reg.model_cls.__init__)
    assert len(params) >= 2, (
        f"'{name}'.__init__({', '.join(params)}) cannot be called as "
        f"build_model does: model_cls(data_cfg, model_cfg)"
    )


def test_registered_name_is_stamped_on_the_class(registration):
    """save() embeds this so load_model can dispatch back to the right class."""
    name, reg = registration
    assert getattr(reg.model_cls, "registered_name", None) == name


def test_config_class_subclasses_model_config(registration):
    name, reg = registration
    assert issubclass(reg.config_cls, ModelConfig), (
        f"'{name}' config {reg.config_cls.__name__} is not a ModelConfig"
    )


def test_config_class_keeps_the_base_orchestration_fields(registration):
    """cross_validate rebuilds the config with dataclasses.replace(output_dir=...),
    and the CLI reads random_seed off it before every run. A config that drops
    or shadows either field breaks those callers, not the model."""
    name, reg = registration
    assert dataclasses.is_dataclass(reg.config_cls), (
        f"'{name}' config {reg.config_cls.__name__} is not a dataclass, so "
        f"cross_validate's dataclasses.replace() will fail on it"
    )
    field_names = {f.name for f in dataclasses.fields(reg.config_cls)}
    assert {"output_dir", "random_seed"} <= field_names, (
        f"'{name}' config is missing {sorted({'output_dir', 'random_seed'} - field_names)}"
    )


def test_supports_out_of_sample_is_a_bool(registration):
    """cross_validate branches on this; a non-bool would silently pass the check."""
    name, reg = registration
    assert isinstance(reg.model_cls.supports_out_of_sample, bool), (
        f"'{name}'.supports_out_of_sample is not a bool"
    )


def test_config_classes_are_not_shared_between_models():
    """model_class_for matches on config type, so two models sharing one config
    class would make dispatch depend on registration order."""
    seen: dict[type, str] = {}
    for name, reg in REGISTERED:
        clash = seen.get(reg.config_cls)
        assert clash is None, (
            f"'{name}' and '{clash}' both register {reg.config_cls.__name__}"
        )
        seen[reg.config_cls] = name


def test_model_config_classes_matches_the_registry():
    """The public accessor load_config uses must not drift from the registry."""
    assert model_config_classes() == {n: r.config_cls for n, r in REGISTERED}


def test_model_class_for_resolves_every_registered_config(registration):
    name, reg = registration
    assert model_class_for(reg.config_cls.__new__(reg.config_cls)) is reg.model_cls


# Extensibility: a model registered from outside src/


@dataclass
class DummyConfig(ModelConfig):
    """Minimal ModelConfig subclass: only the base orchestration fields."""

    n_components: int = 3


class DummyModel(MultiOmicModel):
    """Predicts each view's per-feature training mean. No framework, no GPU.

    Deliberately as simple as a MultiOmicModel can be: whatever it needs in
    order to work is, by definition, the real cost of wrapping an external
    model behind this API.
    """

    def __init__(self, data_cfg: DataConfig, model_cfg: DummyConfig):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg
        self._means: dict[str, np.ndarray] | None = None

    def fit(self, train, val=None, resume_from=None) -> None:
        self._means = {}
        for view in train.view_names:
            values, mask = train.views[view], train.masks[view]
            total = np.where(mask, values, 0.0).sum(axis=0)
            count = np.maximum(mask.sum(axis=0), 1)
            self._means[view] = (total / count).astype(np.float32)

    def _check_fitted(self) -> None:
        if self._means is None:
            raise RuntimeError("DummyModel must be fit before use")

    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        self._check_fitted()
        assert self._means is not None
        rows = [
            np.concatenate([data.views[v][i] for v in sorted(self._means)])
            for i in range(data.n_samples)
        ]
        return np.stack(rows)[:, : self.model_cfg.n_components].astype(np.float32)

    def reconstruct(self, data: MultiOmicDataset) -> dict[str, np.ndarray]:
        self._check_fitted()
        assert self._means is not None
        return {
            view: np.tile(self._means[view], (data.n_samples, 1))
            for view in data.view_names
        }

    def save_outputs(self, output_dir=None) -> None:
        self._check_fitted()
        out = Path(output_dir or self.model_cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "latent.txt").write_text("written")

    def save(self, path) -> None:
        self._check_fitted()
        import torch

        # The dispatch contract from registry.load_model: a checkpoint carries
        # the registered name it was written by.
        torch.save(
            {
                "hyper_parameters": {"model_type_name": self.registered_name},
                "means": self._means,
                "model_cfg": self.model_cfg,
                "data_cfg": self.data_cfg,
            },
            str(path),
        )

    @classmethod
    def load(cls, path, **kwargs) -> DummyModel:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        model = cls(state["data_cfg"], state["model_cfg"])
        model._means = state["means"]
        return model


class TransductiveDummyModel(DummyModel):
    supports_out_of_sample = False


@dataclass
class TransductiveDummyConfig(DummyConfig):
    pass


@pytest.fixture
def dummy_registered():
    """Register DummyModel for one test, then restore the registry.

    Registration is global state. Leaving it mutated would leak into every
    later test that enumerates models or reports known model types.
    """
    before = dict(_REGISTRY)
    register_model("dummy", DummyConfig)(DummyModel)
    register_model("dummy_transductive", TransductiveDummyConfig)(
        TransductiveDummyModel
    )
    yield
    _REGISTRY.clear()
    _REGISTRY.update(before)


@pytest.fixture
def dummy_cfgs(sample_dataset, tmp_path):
    data_cfg = DataConfig(path="unused", views=list(sample_dataset.view_names))
    return data_cfg, DummyConfig(output_dir=str(tmp_path))


def test_registering_a_model_needs_no_source_changes(dummy_registered):
    assert "dummy" in model_config_classes()
    assert DummyModel.registered_name == "dummy"


def test_build_model_dispatches_to_the_new_model(dummy_registered, dummy_cfgs):
    data_cfg, model_cfg = dummy_cfgs
    assert isinstance(build_model(data_cfg, model_cfg), DummyModel)


def test_new_model_completes_the_full_api_round_trip(
    dummy_registered, dummy_cfgs, sample_dataset, tmp_path
):
    """fit, transform, reconstruct, save_outputs, save, load — public API only."""
    data_cfg, model_cfg = dummy_cfgs
    model = build_model(data_cfg, model_cfg)
    model.fit(sample_dataset)

    latent = model.transform(sample_dataset)
    assert latent.shape == (sample_dataset.n_samples, model_cfg.n_components)

    recon = model.reconstruct(sample_dataset)
    assert set(recon) == set(sample_dataset.view_names)
    for view in sample_dataset.view_names:
        assert recon[view].shape == sample_dataset.views[view].shape

    model.save_outputs(tmp_path / "outputs")
    assert (tmp_path / "outputs" / "latent.txt").exists()

    ckpt = tmp_path / "dummy.ckpt"
    model.save(ckpt)
    reloaded = load_model(ckpt)
    assert isinstance(reloaded, DummyModel)
    np.testing.assert_array_equal(
        reloaded.transform(sample_dataset), model.transform(sample_dataset)
    )


def test_new_model_is_cross_validated_through_the_same_scorer(
    dummy_registered, dummy_cfgs, sample_dataset
):
    """The benchmark workflow: a wrapped model is scored by the shared scorer."""
    data_cfg, model_cfg = dummy_cfgs
    results = cross_validate(
        sample_dataset,
        data_cfg,
        model_cfg,
        EvaluationConfig(n_folds=3, strategy="kfold"),
    )

    assert len(results["per_fold"]) == 3
    assert set(results["per_view"]) == set(sample_dataset.view_names)
    assert np.isfinite(results["mean"])


def test_cross_validate_rejects_a_new_transductive_model(
    dummy_registered, sample_dataset, tmp_path
):
    """supports_out_of_sample is honoured for models the scorer has never seen."""
    data_cfg = DataConfig(path="unused", views=list(sample_dataset.view_names))
    model_cfg = TransductiveDummyConfig(output_dir=str(tmp_path))

    with pytest.raises(UnsupportedError, match="dummy_transductive"):
        cross_validate(sample_dataset, data_cfg, model_cfg, EvaluationConfig(n_folds=3))


def test_cross_validate_writes_nothing_to_the_configured_output_dir(
    dummy_registered, dummy_cfgs, sample_dataset, tmp_path
):
    """Folds are scored in temp dirs; the caller's output_dir stays untouched."""
    data_cfg, model_cfg = dummy_cfgs
    out = Path(model_cfg.output_dir)
    before = sorted(p.name for p in out.iterdir())

    cross_validate(
        sample_dataset,
        data_cfg,
        model_cfg,
        EvaluationConfig(n_folds=3, strategy="kfold"),
    )

    assert sorted(p.name for p in out.iterdir()) == before


def test_unregistered_config_subclass_is_rejected_with_its_type_name(dummy_registered):
    """Dispatch is by exact type: a config subclass is a new model, not a variant.

    Documents the constraint a developer hits when deriving a config for a
    model variation — register the variation, do not rely on inheritance.
    """

    @dataclass
    class UnregisteredVariantConfig(DummyConfig):
        pass

    with pytest.raises(TypeError, match="UnregisteredVariantConfig"):
        model_class_for(UnregisteredVariantConfig())
