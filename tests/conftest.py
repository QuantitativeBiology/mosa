from __future__ import annotations

import mudata
import numpy as np
import pandas as pd
import pytest

from mosa.config import DataConfig
from mosa.data.dataset import MultiOmicDataset
from mosa.models.mosa.config import MOSAConfig, OmicViewConfig

# Test fixtures build MuData objects directly and never rely on the
# pull-obs/var-on-update behavior; adopt the post-0.4 default to match
# production code (see mosa/data/io.py) instead of leaving it a FutureWarning.
mudata.set_options(pull_on_update=False)


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
        masks = {
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

        group_labels = [f"Type{'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i]}" for i in range(n_groups)]
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

        model_cfg = MOSAConfig(
            views=view_configs,
            joint_latent_dim=joint_latent_dim,
            fusion_method=fusion_method,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=1e-3,
            test_size=0.0,
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
        import mudata
        from mosa.data.io import _dearrow_mudata

        if view_specs is None:
            view_specs = {"view_a": 10, "view_b": 8}

        anndata.settings.allow_write_nullable_strings = True
        rng = np.random.RandomState(42)

        adatas = {}
        for view_name, n_features in view_specs.items():
            X = rng.randn(n_samples, n_features).astype(np.float32)
            mask = np.ones((n_samples, n_features), dtype=bool)
            obs = pd.DataFrame(index=[f"sample_{i:03d}" for i in range(n_samples)])
            var = pd.DataFrame(index=[f"{view_name}_feat_{j}" for j in range(n_features)])
            adata = anndata.AnnData(X=X, obs=obs, var=var)
            adata.layers["mask"] = mask
            adatas[view_name] = adata

        mdata = mudata.MuData(adatas)
        index = [f"sample_{i:03d}" for i in range(n_samples)]
        obs_df = pd.DataFrame(
            {
                "model_type": ["TypeA" if i % 2 == 0 else "TypeB" for i in range(n_samples)],
                "tissue": ["tissue_0" if i % 2 == 0 else "tissue_1" for i in range(n_samples)],
            },
            index=index,
        )
        obs_df.index = obs_df.index.astype(object)
        mdata.obs = obs_df.copy()
        _dearrow_mudata(mdata)

        path = tmp_path / "test.h5mu"
        mdata.write(str(path))
        return path

    return _make
