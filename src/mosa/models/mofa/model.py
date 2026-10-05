from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.errors import DataError, MissingDependencyError, UnsupportedError
from mosa.models.api import MultiOmicModel
from mosa.models.mofa.config import MOFAConfig
from mosa.models.registry import register_model
from mosa.utils import ensure_dir

logger = logging.getLogger(__name__)

# Top-level groups mofapy2's saveModel writes even with save_data=False.
_MOFAPY2_GROUPS = ("expectations", "features", "samples", "model_options")

# mofapy2 reads this value as missing: R's integer NA, as passed by reticulate.
_R_NA_INTEGER = -2147483648


def _require_mofa() -> None:
    """Fail early with the install command when the mofa extra is absent."""
    missing = [m for m in ("mofapy2", "mofax") if importlib.util.find_spec(m) is None]
    if missing:
        raise MissingDependencyError(
            f"the mofa model requires {' and '.join(missing)}: pip install '.[mofa]'"
        )


@register_model("mofa", MOFAConfig)
class MOFAModel(MultiOmicModel):
    """MultiOmicModel implementation using MOFA+ (mofapy2 / mofax).

    Training uses Automatic Relevance Determination (ARD) for regularization;
    validation data is therefore not used. transform() only looks up the
    factors learned for training samples, so it rejects unseen samples and
    training samples whose values changed.
    """

    supports_out_of_sample = False
    checkpoint_suffixes = (".hdf5",)

    @classmethod
    def owns_checkpoint(cls, path: str | Path) -> bool:
        """An .hdf5 file laid out the way mofapy2 saves a trained model.

        Checks content, not just the suffix, so another model writing .hdf5
        can be registered without changing this class.
        """
        if not super().owns_checkpoint(path):
            return False
        import h5py

        try:
            with h5py.File(path, "r") as f:
                return all(k in f for k in _MOFAPY2_GROUPS)
        except OSError:
            return False

    def __init__(
        self,
        data_cfg: DataConfig | None = None,
        model_cfg: MOFAConfig | None = None,
    ):
        self.data_cfg = data_cfg
        self.model_cfg = model_cfg or MOFAConfig()
        self.save_path: str | None = None
        self._model: Any = None
        self._ent: Any = None
        self._train_samples: list[str] | None = None
        self._train_features: dict[str, list[str]] = {}
        self._train_fingerprints: dict[str, np.ndarray] = {}
        self._scales: dict[str, tuple[float, dict[str, float]]] = {}

    @staticmethod
    def _require_ascii_names(data: MultiOmicDataset) -> None:
        """Fail before training on names mofax cannot read back.

        mofax decodes every stored name as ASCII, so a non-ASCII name trains
        and saves but the model file cannot be opened afterwards.
        """
        named: dict[str, Any] = {
            "sample": data.sample_names,
            "view": data.view_names,
            "model_type": data.metadata["model_type"].unique(),
            **{f"'{v}' feature": data.feature_names[v] for v in data.view_names},
        }
        for kind, names in named.items():
            bad = [str(n) for n in names if not str(n).isascii()]
            if bad:
                raise DataError(
                    f"MOFA only supports ASCII names, but {len(bad)} {kind} "
                    f"name(s) are not: {bad[:3]}. Rename them before training."
                )

    @staticmethod
    def _group_labels(data: MultiOmicDataset) -> np.ndarray:
        """model_type as strings; mofapy2 cannot save non-string group names."""
        raw = data.metadata["model_type"]
        labels = raw.astype(str)
        if labels.nunique() != raw.nunique():
            raise DataError(
                f"model_type values {sorted(map(repr, raw.unique()))} name the "
                f"same group once written as text (e.g. 1 and '1'). Give each "
                f"group a distinct label."
            )
        return labels.to_numpy()

    @staticmethod
    def _observed(data: MultiOmicDataset, view_name: str) -> np.ndarray:
        """View as float64 with NaN wherever mofapy2 treats a value as missing."""
        x = np.where(data.masks[view_name], data.views[view_name], np.nan).astype(float)
        x[x == _R_NA_INTEGER] = np.nan
        return x

    @staticmethod
    def _group_matrices(
        data: MultiOmicDataset, views: list[str]
    ) -> tuple[list[list[np.ndarray]], list[str], list[list[str]]]:
        """Per-view, per-group (samples x features) matrices with NaN where masked.

        Groups are sorted and samples keep their input order within a group,
        the layout set_data_df produced.
        """
        labels = MOFAModel._group_labels(data)
        groups = np.unique(labels).tolist()
        rows = [np.flatnonzero(labels == g) for g in groups]
        matrices = []
        for view_name in views:
            x = MOFAModel._observed(data, view_name)
            matrices.append([x[r] for r in rows])
        samples = [[data.sample_names[i] for i in r] for r in rows]
        return matrices, groups, samples

    def _scale_factors(
        self, train: MultiOmicDataset
    ) -> dict[str, tuple[float, dict[str, float]]]:
        """Per-view and per-(view, group) std that mofapy2 divides by, or 1.0 when off.

        Mirrors mofapy2's process_data: center each feature within its group,
        divide the view by its nanstd, then divide each group block by its own.
        """
        mc = self.model_cfg
        groups = self._group_labels(train)
        names = np.unique(groups)
        factors: dict[str, tuple[float, dict[str, float]]] = {}
        for view_name in train.view_names:
            x = self._observed(train, view_name)
            for g in names:
                x[groups == g] -= np.nanmean(x[groups == g], axis=0)
            view_sd = float(np.nanstd(x)) if mc.scale_views else 1.0
            x /= view_sd
            group_sd = {
                str(g): float(np.nanstd(x[groups == g])) if mc.scale_groups else 1.0
                for g in names
            }
            factors[view_name] = (view_sd, group_sd)
        return factors

    def fit(
        self,
        train: MultiOmicDataset,
        val: MultiOmicDataset | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        """Train the MOFA model. val and resume_from are ignored."""
        _require_mofa()
        self._require_ascii_names(train)
        from mofapy2.run.entry_point import entry_point

        mc = self.model_cfg
        ent = entry_point()
        ent.set_data_options(
            scale_views=mc.scale_views,
            scale_groups=mc.scale_groups,
        )
        views = sorted(train.view_names)
        matrices, groups, samples = self._group_matrices(train, views)
        features = [list(train.feature_names[v]) for v in views]
        # set_data_matrix rejects a feature name repeated across views, which
        # the model itself handles. Pass placeholders unique by construction
        # (view index, feature index) and restore the real names afterwards.
        ent.set_data_matrix(
            matrices,
            views_names=views,
            groups_names=groups,
            samples_names=samples,
            features_names=[
                [f"{m}:{j}" for j in range(len(names))]
                for m, names in enumerate(features)
            ],
        )
        ent.data_opts["features_names"] = features
        ent.set_model_options(factors=mc.n_factors, ard_factors=mc.ard_factors)
        ent.set_train_options(
            dropR2=mc.drop_r2,
            # mofapy2 runs updates 1..iter-1, so iter=n would give n-1 updates.
            iter=mc.iterations + 1,
            seed=mc.random_seed,
            convergence_mode=mc.convergence_mode,
        )
        ent.build()
        try:
            ent.run()
        except SystemExit as e:
            # mofapy2 calls exit() instead of raising: when ARD drops every
            # factor, and when its run() wrapper catches Ctrl-C or a TypeError.
            # Left alone, the process would end with status 0.
            if isinstance(e.__context__, KeyboardInterrupt):
                raise KeyboardInterrupt from None
            if isinstance(e.__context__, TypeError):
                raise e.__context__ from None
            if ent.model.dim["K"] == 0:
                raise DataError(
                    "MOFA dropped every factor during training: none explained "
                    "more than drop_r2 of the variance in any view and group. "
                    "Check the input, or set model.drop_r2: null to keep every "
                    "factor."
                ) from None
            raise RuntimeError("mofapy2 exited during training") from e
        # mofapy2 catches Ctrl-C, marks the model untrained and returns.
        if not ent.model.trained:
            raise KeyboardInterrupt
        # The ELBO trace is truncated on convergence, so a full-length trace
        # (iter + 1 entries) means training stopped at the cap.
        if len(ent.model.train_stats["elbo"]) > mc.iterations + 1:
            logger.warning(
                "MOFA stopped at the iteration cap (%d) without converging; "
                "raise model.iterations.",
                mc.iterations,
            )

        # Snapshot what save_outputs needs: train is the caller's object and
        # may change before then.
        self._ent = ent
        self._train_samples = list(train.sample_names)
        self._train_features = {v: list(train.feature_names[v]) for v in views}
        self._train_fingerprints = self._fingerprints(train)
        self._scales = self._scale_factors(train)
        self._model = None

    @staticmethod
    def _fingerprints(data: MultiOmicDataset) -> dict[str, np.ndarray]:
        """One 64-bit hash per sample and view of its values, NaN where masked."""
        out: dict[str, np.ndarray] = {}
        for view_name in data.view_names:
            x = np.where(data.masks[view_name], data.views[view_name], np.nan)
            out[view_name] = np.array(
                [
                    int.from_bytes(
                        hashlib.blake2b(row.tobytes(), digest_size=8).digest(), "little"
                    )
                    for row in x.astype(np.float32)
                ],
                dtype=np.uint64,
            )
        return out

    def _check_features(self, data: MultiOmicDataset) -> None:
        """Raise when a view is unknown or lists other features than training, in order."""
        trained = self._model.model["features"]
        missing = [v for v in trained if v not in data.view_names]
        if missing:
            raise DataError(
                f"Input is missing view(s) {missing} the model was trained on; "
                f"the training values of those views cannot be checked."
            )
        for view_name in data.view_names:
            if view_name not in trained:
                raise DataError(
                    f"View '{view_name}' was not part of training; the model "
                    f"was trained on {list(trained)}."
                )
            names = [n.decode() for n in trained[view_name][:]]
            if names != list(data.feature_names[view_name]):
                raise DataError(
                    f"View '{view_name}' features do not match the ones the "
                    f"model was trained on, in names or in order."
                )

    def _check_training_values(self, data: MultiOmicDataset) -> None:
        """Raise when data's values differ from the ones these samples were trained on."""
        f = self._model.model
        # Files written before MOSA stored fingerprints cannot be checked.
        if "mosa/fingerprint" not in f:
            return
        stored = f["mosa/fingerprint"]
        index = {s: i for i, s in enumerate(stored["samples"].asstr()[:])}
        rows = [index[s] for s in data.sample_names]
        names = np.array(data.sample_names)
        changed: set[str] = set()
        for view_name, current in self._fingerprints(data).items():
            if view_name in stored["views"]:
                trained = stored["views"][view_name][:][rows]
                changed.update(names[trained != current])
        if changed:
            shown = sorted(str(s) for s in changed)
            more = f" and {len(shown) - 5} more" if len(shown) > 5 else ""
            raise DataError(
                f"MOFA cannot project new data: the values of {len(shown)} "
                f"sample(s) differ from training, in values or feature order: "
                f"{shown[:5]}{more}. It only returns the factors it learned for "
                f"its training samples."
            )

    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        """Latent factors for training samples. Raises for unseen samples or changed values."""
        if self._model is None:
            raise RuntimeError(
                "Model must be fit and save_outputs() called before calling transform()"
            )

        factors_df = self._model.get_factors(df=True)
        missing = [s for s in data.sample_names if s not in factors_df.index]
        if missing:
            raise UnsupportedError(
                f"MOFA does not support out-of-sample projection. "
                f"Unseen samples: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        self._check_features(data)
        self._check_training_values(data)

        return factors_df.loc[data.sample_names].values

    def _likelihoods(self) -> dict[str, str]:
        """Likelihood mofapy2 fit each view with, keyed by view name."""
        stored = self._model.model["model_options"]["likelihoods"][:]
        return {v: lk.decode() for v, lk in zip(self._model.views, stored, strict=True)}

    def _reconstruct_view(
        self,
        view_name: str,
        Z: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """Undo mofapy2's preprocessing of Z @ W.T for one view."""
        likelihood = self._likelihoods()[view_name]
        if likelihood != "gaussian":
            # mofapy2 neither centers nor scales these views, and Z @ W.T is
            # on the link scale, not the data scale.
            raise UnsupportedError(
                f"Cannot reconstruct view '{view_name}': MOFA fit it with a "
                f"{likelihood} likelihood (guessed from its values), whose "
                f"reconstructions are not on the data scale."
            )

        W_df = self._model.get_weights(views=view_name, df=True)
        scales = self._model.model["mosa"]
        group_sd = np.array([scales["group_scale"][view_name][g][()] for g in groups])
        recon = Z @ W_df.values.T
        recon = recon * group_sd[:, None] * scales["view_scale"][view_name][()]
        f = self._model.model
        intercepts = {g: f["intercepts"][view_name][g][:] for g in np.unique(groups)}
        return recon + np.stack([intercepts[g] for g in groups])

    def reconstruct(self, data: MultiOmicDataset) -> dict[str, np.ndarray]:
        """Reconstruct omic views on the original scale."""
        if self._model is None:
            raise RuntimeError("Model must be fit before calling reconstruct()")
        # Without the mosa group the file's centering and scaling are unknown:
        # written by mofapy2 directly, or by MOSA before it recorded them.
        if "mosa" not in self._model.model:
            raise UnsupportedError(
                f"Cannot reconstruct with '{self.save_path}': the file does not "
                f"record how MOFA preprocessed the data. Retrain the model with "
                f"this version of MOSA. transform() still works."
            )

        Z = self.transform(data)
        # Groups come from the model, not data.metadata: relabeling model_type
        # in the input must not shift the offsets.
        groups = self._model.samples_metadata["group"].loc[data.sample_names].values
        return {v: self._reconstruct_view(v, Z, groups) for v in data.view_names}

    def save_outputs(self, output_dir: str | Path | None = None) -> None:
        """Write the model file and the latent and reconstructions of the training samples.

        Writes mofa_model.hdf5 and full/latent.parquet, full/recon_<view>.parquet
        to output_dir, or model_cfg.output_dir when output_dir is None. Also
        constructs the mofax reader used by transform()/reconstruct(); those
        methods raise until this has been called at least once after fit().
        """
        if self._ent is None or self._train_samples is None:
            raise RuntimeError("Model must be fit before calling save_outputs()")

        out_dir = ensure_dir(output_dir or self.model_cfg.output_dir)
        save_path = str(out_dir / "mofa_model.hdf5")
        logger.info("Saving outputs to %s", out_dir)

        # h5py cannot truncate a file it still holds open.
        if self._model is not None:
            self._model.close()
            self._model = None

        self._ent.save(save_path, save_data=True)
        self.save_path = save_path

        import h5py

        # mofapy2 does not save its scale factors; a separate top-level group
        # keeps mofax's reader unaffected.
        with h5py.File(save_path, "a") as f:
            mosa = f.create_group("mosa")
            if self.data_cfg is not None:
                cfg = dataclasses.asdict(self.data_cfg)
                cfg["discrete_views"] = sorted(cfg["discrete_views"])
                mosa.attrs["data_cfg"] = json.dumps(cfg)
            for view_name, (view_sd, group_sd) in self._scales.items():
                mosa[f"view_scale/{view_name}"] = view_sd
                for g, sd in group_sd.items():
                    mosa[f"group_scale/{view_name}/{g}"] = sd
            fingerprint = mosa.create_group("fingerprint")
            fingerprint.create_dataset(
                "samples",
                data=self._train_samples,
                dtype=h5py.string_dtype(),
            )
            for view_name, hashes in self._train_fingerprints.items():
                fingerprint[f"views/{view_name}"] = hashes

        import mofax as mfx

        self._model = mfx.mofa_model(save_path)

        samples = self._train_samples
        split_dir = ensure_dir(out_dir / "full")
        # A previous run into this directory may have written views this one
        # skips or no longer has.
        for stale in split_dir.glob("recon_*.parquet"):
            stale.unlink()
        Z = self._model.get_factors(df=True).loc[samples].values
        pd.DataFrame(Z, index=samples).to_parquet(split_dir / "latent.parquet")

        groups = self._model.samples_metadata["group"].loc[samples].values
        likelihoods = self._likelihoods()
        for view_name, features in self._train_features.items():
            if likelihoods[view_name] != "gaussian":
                logger.warning(
                    "Skipping recon_%s.parquet: MOFA fit this view with a %s "
                    "likelihood, whose reconstructions are not on the data scale.",
                    view_name,
                    likelihoods[view_name],
                )
                continue
            recon = self._reconstruct_view(view_name, Z, groups)
            pd.DataFrame(recon, index=samples, columns=features).to_parquet(
                split_dir / f"recon_{view_name}.parquet"
            )

    def save(self, path: str | Path) -> None:
        """Copy the HDF5 model file to path."""
        if self._model is None or self.save_path is None:
            raise RuntimeError("Model must be fit before saving")
        self.require_checkpoint_suffix(path)

        src = Path(self.save_path)
        dst = Path(path)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> MOFAModel:
        """Load a trained MOFA model from an HDF5 file.

        Restores the DataConfig saved with the model. Files without one (plain
        mofapy2 output, or MOSA before it saved it) get the model's view names
        and defaults for every other field.
        """
        _require_mofa()
        import mofax as mfx

        instance = cls()
        instance.save_path = str(path)
        instance._model = mfx.mofa_model(str(path))
        saved = instance._model.model.get("mosa")
        if saved is not None and "data_cfg" in saved.attrs:
            cfg = json.loads(saved.attrs["data_cfg"])
            cfg["discrete_views"] = set(cfg["discrete_views"])
            instance.data_cfg = DataConfig(**cfg)
        else:
            instance.data_cfg = DataConfig(
                path=str(path), views=list(instance._model.views)
            )
        return instance
