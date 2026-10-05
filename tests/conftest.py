from __future__ import annotations

from pathlib import Path

import mudata
import numpy as np
import pandas as pd
import pytest

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig

# Tests that build, write or read MuData directly wrap those calls in
# mudata.set_options(pull_on_update=False), as production code does. The
# option is not set globally: pyproject turns mudata's FutureWarning into an
# error, so a src/ call missing the wrapper still fails the suite.


@pytest.fixture
def make_multi_omic_dataset():
    def _make(
        n_samples: int = 20,
        view_dims: dict[str, int] | None = None,
        n_groups: int = 2,
        missing_frac: float = 0.0,
        seed: int = 42,
    ) -> MultiOmicDataset:
        if view_dims is None:
            view_dims = {"view_a": 50, "view_b": 30}

        rng = np.random.RandomState(seed)

        views = {
            name: rng.randn(n_samples, dim).astype(np.float32)
            for name, dim in view_dims.items()
        }
        masks: dict[str, np.ndarray] = {
            name: np.ones((n_samples, dim), dtype=bool)
            for name, dim in view_dims.items()
        }

        if missing_frac > 0.0:
            for name, dim in view_dims.items():
                flat_mask = masks[name].ravel()
                n_missing = int(len(flat_mask) * missing_frac)
                missing_idx = rng.choice(len(flat_mask), size=n_missing, replace=False)
                flat_mask[missing_idx] = False
                masks[name] = flat_mask.reshape(n_samples, dim)
                views[name][~masks[name]] = 0.0

        group_labels = [
            f"Type{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}" for i in range(n_groups)
        ]
        index = [f"sample_{i:03d}" for i in range(n_samples)]
        metadata = pd.DataFrame(
            {
                "model_type": [group_labels[i % n_groups] for i in range(n_samples)],
                "tissue": [["tissue_0", "tissue_1"][i % 2] for i in range(n_samples)],
            },
            index=index,
        )

        feature_names = {
            name: [f"{name}_feat_{j}" for j in range(dim)]
            for name, dim in view_dims.items()
        }

        return MultiOmicDataset(views, masks, metadata, feature_names)

    return _make


@pytest.fixture
def sample_dataset(make_multi_omic_dataset):
    return make_multi_omic_dataset()


@pytest.fixture
def make_mosa_config():
    """Build a (DataConfig, MOSAConfig) pair sized to a test dataset."""

    def _make(
        dataset: MultiOmicDataset,
        joint_latent_dim: int = 16,
        fusion_method: str = "concat",
        num_epochs: int = 2,
        batch_size: int = 8,
        discrete_views: set[str] | None = None,
        data_path: str = "unused",
        **overrides,
    ) -> tuple[DataConfig, MOSAConfig]:
        data_cfg = DataConfig(
            path=data_path,
            views=list(dataset.view_names),
            discrete_views=discrete_views or set(),
        )

        view_configs = {
            name: OmicViewConfig(name=name, hidden_layer_dims=[32, 16])
            for name in dataset.view_names
        }

        output_dir = overrides.pop("output_dir", "/tmp/mosa_test")
        # Default to CPU/single-device: accelerator="auto" + devices="auto" makes
        # Lightning DDP-spawn across every visible GPU, which is both wasteful and,
        # on a shared multi-GPU host, slow enough to look like a hang.
        overrides.setdefault("accelerator", "cpu")
        overrides.setdefault("devices", 1)

        model_cfg = MOSAConfig(
            views=view_configs,
            joint_latent_dim=joint_latent_dim,
            fusion_method=fusion_method,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            output_dir=output_dir,
            weighted_random_sampler=False,
            **overrides,
        )

        return data_cfg, model_cfg

    return _make


@pytest.fixture
def sample_config(sample_dataset, make_mosa_config, tmp_path):
    return make_mosa_config(sample_dataset, output_dir=str(tmp_path))


@pytest.fixture
def make_h5mu_file():
    """Fixture factory: write a tiny .h5mu file and return its path."""

    def _make(tmp_path, n_samples: int = 20, view_specs: dict[str, int] | None = None):
        import anndata

        from mosa.data.io import _dearrow_mudata

        if view_specs is None:
            view_specs = {"view_a": 10, "view_b": 8}

        anndata.settings.allow_write_nullable_strings = True
        rng = np.random.RandomState(42)

        adatas = {}
        for view_name, n_features in view_specs.items():
            X = rng.randn(n_samples, n_features).astype(np.float32)
            mask: np.ndarray = np.ones((n_samples, n_features), dtype=bool)
            obs = pd.DataFrame(index=[f"sample_{i:03d}" for i in range(n_samples)])
            var = pd.DataFrame(
                index=[f"{view_name}_feat_{j}" for j in range(n_features)]
            )
            adata = anndata.AnnData(X=X, obs=obs, var=var)
            adata.layers["mask"] = mask
            adatas[view_name] = adata

        index = [f"sample_{i:03d}" for i in range(n_samples)]
        obs_df = pd.DataFrame(
            {
                "model_type": [
                    "TypeA" if i % 2 == 0 else "TypeB" for i in range(n_samples)
                ],
                "tissue": [
                    "tissue_0" if i % 2 == 0 else "tissue_1" for i in range(n_samples)
                ],
            },
            index=index,
        )
        obs_df.index = obs_df.index.astype(object)

        path = tmp_path / "test.h5mu"
        with mudata.set_options(pull_on_update=False):
            mdata = mudata.MuData(adatas)
            mdata.obs = obs_df.copy()
            _dearrow_mudata(mdata)
            mdata.write(str(path))
        return path

    return _make


@pytest.fixture
def make_csv_dataset():
    """Fixture factory: write the CSV set `mosa convert` consumes.

    Returns a dict with 'conditionals', 'views' ({name: path}) and, when
    requested, 'mutations'. Views are written features x samples, the
    orientation the converter expects.
    """

    def _make(
        tmp_path,
        n_samples: int = 12,
        view_specs: dict[str, int] | None = None,
        n_types: int = 2,
        mutations: int = 0,
        prefix: str = "",
    ) -> dict:
        if view_specs is None:
            view_specs = {"view_a": 6, "view_b": 5}

        rng = np.random.RandomState(7)
        samples = [f"S{i:03d}" for i in range(n_samples)]
        types = [f"Type{'ABCDE'[i % n_types]}" for i in range(n_samples)]

        cond_path = tmp_path / f"{prefix}conditionals.csv"
        pd.DataFrame(
            {
                "model_id": samples,
                "model_type": types,
                "tissue": [f"tissue_{i % 2}" for i in range(n_samples)],
            }
        ).to_csv(cond_path, index=False)

        view_paths = {}
        for name, n_features in view_specs.items():
            features = [f"{name}_feat_{j}" for j in range(n_features)]
            frame = pd.DataFrame(
                rng.randn(n_features, n_samples), index=features, columns=samples
            )
            view_paths[name] = tmp_path / f"{prefix}{name}.csv"
            frame.to_csv(view_paths[name])

        paths = {"conditionals": cond_path, "views": view_paths}

        if mutations:
            features = [f"gene_{j}" for j in range(mutations)]
            frame = pd.DataFrame(
                rng.randint(0, 2, size=(mutations, n_samples)),
                index=features,
                columns=samples,
            )
            paths["mutations"] = tmp_path / f"{prefix}mutations.csv"
            frame.to_csv(paths["mutations"])

        return paths

    return _make


@pytest.fixture
def make_config_file():
    """Fixture factory: write a CLI config YAML and return its path.

    Sized for tests (one epoch, CPU, tiny layers) so a command that trains
    stays fast. `model_overrides` and `evaluation` patch the corresponding
    YAML blocks.
    """

    def _make(
        path,
        data_path,
        output_dir,
        views,
        *,
        model_type: str = "mosa_vae",
        evaluation: dict | None = None,
        data: dict | None = None,
        name: str = "config.yaml",
        **model_overrides,
    ):
        import yaml

        model: dict = {
            "type": model_type,
            "output_dir": str(output_dir),
            "random_seed": 42,
        }
        if model_type == "mosa_vae":
            model.update(
                joint_latent_dim=8,
                num_epochs=1,
                batch_size=8,
                learning_rate=1e-3,
                weighted_random_sampler=False,
                accelerator="cpu",
                devices=1,
                precision="32",
                views={name_: {"hidden_layer_dims": [16, 8]} for name_ in views},
            )
        model.update(model_overrides)

        config = {
            "data": {"path": str(data_path), "views": list(views), **(data or {})},
            "model": model,
            "evaluation": evaluation if evaluation is not None else {"test_size": 0.0},
        }

        config_path = Path(path) / name
        config_path.write_text(yaml.dump(config))
        return config_path

    return _make
