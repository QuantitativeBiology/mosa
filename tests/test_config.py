from __future__ import annotations

import pytest
import yaml

from mosa.config import Config, DataConfig
from mosa.models.mofa.config import MOFAConfig
from mosa.models.mosa.config import MOSAVAEConfig, MOSAVAEViewConfig
from mosa.utils import load_config


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


def test_data_config_basic():
    cfg = DataConfig(path="x.h5mu", views=["rna", "meth"])
    assert cfg.views == ["rna", "meth"]
    assert cfg.discrete_views == set()


def test_data_config_discrete_views_list_coerced_to_set():
    cfg = DataConfig(path="x.h5mu", views=["rna"], discrete_views=["rna"])
    assert cfg.discrete_views == {"rna"}


def test_data_config_no_views_raises():
    with pytest.raises(ValueError, match="data.views"):
        DataConfig(path="x.h5mu", views=[])


def test_data_config_discrete_not_in_views():
    with pytest.raises(ValueError, match="discrete_views"):
        DataConfig(path="x.h5mu", views=["rna"], discrete_views={"meth"})


def test_data_config_validate_paths_missing():
    cfg = DataConfig(path="nonexistent.h5mu", views=["rna"])
    with pytest.raises(FileNotFoundError, match="data.path"):
        cfg.validate_paths()


def test_data_config_validate_paths_empty():
    cfg = DataConfig.__new__(DataConfig)
    cfg.path = ""
    cfg.views = ["rna"]
    cfg.mask_layer_name = "mask"
    cfg.discrete_views = set()
    cfg.use_tissue = True
    cfg.use_mutations = True
    with pytest.raises(FileNotFoundError, match="required"):
        cfg.validate_paths()


# ---------------------------------------------------------------------------
# MOSAVAEViewConfig
# ---------------------------------------------------------------------------


def test_view_config_empty_hidden_dims():
    with pytest.raises(ValueError, match="hidden_layer_dims must not be empty"):
        MOSAVAEViewConfig(name="x", hidden_layer_dims=[])


def test_view_config_negative_hidden_dim():
    with pytest.raises(ValueError, match="all hidden_layer_dims must be positive"):
        MOSAVAEViewConfig(name="x", hidden_layer_dims=[256, -1])


def test_view_config_invalid_loss_type():
    with pytest.raises(ValueError, match="loss_type must be one of"):
        MOSAVAEViewConfig(name="x", loss_type="invalid")


def test_view_config_invalid_dropout():
    with pytest.raises(ValueError, match="dropout_p must be in"):
        MOSAVAEViewConfig(name="x", dropout_p=1.5)


# ---------------------------------------------------------------------------
# MOSAVAEConfig — defaults and numeric ranges
# ---------------------------------------------------------------------------


def test_mosa_default():
    cfg = MOSAVAEConfig()
    assert cfg.joint_latent_dim == 64
    assert cfg.output_dir == "outputs"


def test_mosa_string_coercion():
    cfg = MOSAVAEConfig(learning_rate="1e-5")  # type: ignore[arg-type]
    assert isinstance(cfg.learning_rate, float)
    assert cfg.learning_rate == pytest.approx(1e-5)


def test_invalid_fusion_method():
    with pytest.raises(ValueError, match="fusion_method must be one of"):
        MOSAVAEConfig(fusion_method="invalid")


@pytest.mark.parametrize("dim", [0, -1])
def test_invalid_joint_latent_dim(dim):
    with pytest.raises(ValueError, match="joint_latent_dim must be positive"):
        MOSAVAEConfig(joint_latent_dim=dim)


def test_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size must be positive"):
        MOSAVAEConfig(batch_size=0)


def test_invalid_num_epochs():
    with pytest.raises(ValueError, match="num_epochs must be positive"):
        MOSAVAEConfig(num_epochs=0)


@pytest.mark.parametrize("ts", [-0.1, 1.0])
def test_invalid_test_size(ts):
    with pytest.raises(ValueError, match="test_size must be in"):
        MOSAVAEConfig(test_size=ts)


@pytest.mark.parametrize("lr", [0, -1e-3])
def test_invalid_learning_rate(lr):
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        MOSAVAEConfig(learning_rate=lr)


@pytest.mark.parametrize("prob", [1.0, -0.1])
def test_invalid_view_dropout(prob):
    with pytest.raises(ValueError, match="view_dropout_prob must be in"):
        MOSAVAEConfig(view_dropout_prob=prob)


@pytest.mark.parametrize("frac", [0.0, 1.5])
def test_invalid_scaler_sample_frac(frac):
    with pytest.raises(ValueError, match="scaler_sample_frac must be in"):
        MOSAVAEConfig(scaler_sample_frac=frac)


def test_adv_lr_required_when_adv_weight():
    with pytest.raises(ValueError, match="adv_learning_rate must be positive"):
        MOSAVAEConfig(adv_weight=1.0, adv_learning_rate=0)


def test_kl_warmup_warning():
    with pytest.warns(UserWarning, match="kl_warmup_epochs"):
        MOSAVAEConfig(use_kl_scheduler=True, kl_warmup_epochs=500, num_epochs=100)


def test_invalid_precision():
    with pytest.raises(ValueError, match="precision must be one of"):
        MOSAVAEConfig(precision="64")


def test_invalid_accelerator():
    with pytest.raises(ValueError, match="accelerator must be one of"):
        MOSAVAEConfig(accelerator="tpu")


def test_invalid_devices():
    with pytest.raises(ValueError, match="devices must be"):
        MOSAVAEConfig(devices=0)


def test_poe_mismatched_hidden_dims():
    views = {
        "rna": MOSAVAEViewConfig(name="rna", hidden_layer_dims=[256, 128]),
        "meth": MOSAVAEViewConfig(name="meth", hidden_layer_dims=[256, 64]),
    }
    with pytest.raises(ValueError, match="PoE fusion requires"):
        MOSAVAEConfig(views=views, fusion_method="poe")


def test_poe_matched_hidden_dims():
    views = {
        "rna": MOSAVAEViewConfig(name="rna", hidden_layer_dims=[256, 128]),
        "meth": MOSAVAEViewConfig(name="meth", hidden_layer_dims=[512, 128]),
    }
    cfg = MOSAVAEConfig(views=views, fusion_method="poe")
    assert cfg.fusion_method == "poe"


# ---------------------------------------------------------------------------
# MOFAConfig
# ---------------------------------------------------------------------------


def test_mofa_default():
    cfg = MOFAConfig()
    assert cfg.n_factors == 50
    assert cfg.convergence_mode == "fast"


def test_mofa_invalid_n_factors():
    with pytest.raises(ValueError, match="n_factors"):
        MOFAConfig(n_factors=0)


def test_mofa_invalid_convergence_mode():
    with pytest.raises(ValueError, match="convergence_mode"):
        MOFAConfig(convergence_mode="invalid")


# ---------------------------------------------------------------------------
# load_config — YAML parsing and dispatch
# ---------------------------------------------------------------------------


def _write_yaml(path, payload):
    path.write_text(yaml.dump(payload))
    return path


def _mosa_payload(path="data/dataset.h5mu"):
    return {
        "data": {
            "path": path,
            "views": ["rna"],
        },
        "model": {
            "type": "mosa_vae",
            "joint_latent_dim": 32,
            "num_epochs": 10,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "test_size": 0.0,
            "views": {
                "rna": {"hidden_layer_dims": [256, 128], "loss_type": "mean", "dropout_p": 0.1},
            },
        },
    }


def test_load_config_mosa(tmp_path):
    yaml_path = _write_yaml(tmp_path / "c.yaml", _mosa_payload())
    cfg = load_config(yaml_path)
    assert isinstance(cfg, Config)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, MOSAVAEConfig)
    assert cfg.data.views == ["rna"]
    assert cfg.model.joint_latent_dim == 32
    assert cfg.model.views["rna"].hidden_layer_dims == [256, 128]


def test_load_config_mofa(tmp_path):
    payload = {
        "data": {"path": "x.h5mu", "views": ["rna", "meth"]},
        "model": {"type": "mofa", "n_factors": 10, "test_size": 0.0},
    }
    yaml_path = _write_yaml(tmp_path / "c.yaml", payload)
    cfg = load_config(yaml_path)
    assert isinstance(cfg.model, MOFAConfig)
    assert cfg.model.n_factors == 10


def test_load_config_missing_data_block(tmp_path):
    yaml_path = _write_yaml(tmp_path / "c.yaml", {"model": {"type": "mofa"}})
    with pytest.raises(ValueError, match="data.*model"):
        load_config(yaml_path)


def test_load_config_missing_model_type(tmp_path):
    yaml_path = _write_yaml(tmp_path / "c.yaml", {
        "data": {"path": "x.h5mu", "views": ["rna"]},
        "model": {"n_factors": 10},
    })
    with pytest.raises(ValueError, match="model.type"):
        load_config(yaml_path)


def test_load_config_unknown_model_type(tmp_path):
    yaml_path = _write_yaml(tmp_path / "c.yaml", {
        "data": {"path": "x.h5mu", "views": ["rna"]},
        "model": {"type": "bogus"},
    })
    with pytest.raises(ValueError, match="unknown model.type"):
        load_config(yaml_path)


def test_load_config_views_mismatch(tmp_path):
    payload = _mosa_payload()
    payload["data"]["views"] = ["rna", "meth"]
    # model.views still has only "rna"
    yaml_path = _write_yaml(tmp_path / "c.yaml", payload)
    with pytest.raises(ValueError, match="model.views must match data.views"):
        load_config(yaml_path)
