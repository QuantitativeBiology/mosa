from __future__ import annotations

import warnings
from dataclasses import dataclass, field

from mosa.config import ModelConfig, check_unknown_keys
from mosa.errors import ConfigError, DataError

_VALID_FUSION_METHODS = ("concat", "poe")
_VALID_LOSS_TYPES = ("mean", "macro")
_VALID_LR_SCHEDULERS = ("none", "step")
_VALID_PRECISIONS = ("32", "16-mixed", "bf16-mixed")
_VALID_ACCELERATORS = ("auto", "cpu", "gpu", "mps")
_VALID_PREPROCESSING_MODES = ("standardize", "center", "none")


@dataclass
class OmicViewConfig:
    """Per-view encoder/decoder architecture for the MOSA VAE."""

    name: str
    hidden_layer_dims: list[int] = field(default_factory=lambda: [512, 256])
    loss_type: str = "mean"
    dropout_p: float = 0.1
    recon_weight: float = 1.0

    def __post_init__(self):
        if not self.hidden_layer_dims:
            raise ConfigError(f"View '{self.name}': hidden_layer_dims must not be empty")
        if any(d <= 0 for d in self.hidden_layer_dims):
            raise ConfigError(f"View '{self.name}': all hidden_layer_dims must be positive")
        if self.loss_type not in _VALID_LOSS_TYPES:
            raise ConfigError(
                f"View '{self.name}': loss_type must be one of {_VALID_LOSS_TYPES}, "
                f"got '{self.loss_type}'"
            )
        if not 0.0 <= self.dropout_p < 1.0:
            raise ConfigError(f"View '{self.name}': dropout_p must be in [0, 1), got {self.dropout_p}")
        if self.recon_weight <= 0:
            raise ConfigError(f"View '{self.name}': recon_weight must be positive, got {self.recon_weight}")


@dataclass
class MOSAConfig(ModelConfig):
    """All MOSA-VAE hyperparameters: architecture, losses, optimiser, training loop, Lightning."""

    # Architecture
    views: dict[str, OmicViewConfig] = field(default_factory=dict)
    joint_latent_dim: int = 64
    fusion_method: str = "concat"
    shared_hidden_layer_dims: list[int] = field(default_factory=list)
    poe_use_shared_head: bool = True  # PoE: True = shared head, False = direct mu/logvar per view
    view_dropout_prob: float = 0.2

    # Losses
    # kl_divergence sums over joint_latent_dim before averaging over the batch
    # (textbook VAE KL), so its magnitude scales with joint_latent_dim; these
    # defaults are 0.01 / 64 (the default joint_latent_dim), matching the old
    # per-element-mean formula's effective weighting.
    kl_weight: float = 1.5625e-4
    kl_weight_final: float = 1.5625e-4
    kl_warmup_epochs: int = 0
    use_kl_scheduler: bool = False
    contrastive_weight: float = 0.0
    adv_weight: float = 0.0
    adv_focal_gamma: float = 0.0  # focal loss gamma for adversarial CE; 0 disables (plain CE)

    # Optimiser
    learning_rate: float = 1e-3
    adv_learning_rate: float = 1e-3
    lr_scheduler: str = "none"
    lr_step_size: int = 100
    lr_gamma: float = 0.5

    # Training loop
    num_epochs: int = 200
    batch_size: int = 64
    scaler_sample_frac: float = 1.0
    weighted_random_sampler: bool = True
    use_adv_class_weights: bool = True  # use class weights in adversarial cross-entropy loss
    inference: bool = False
    target_batch: str = ""
    preprocessing_mode: str = "standardize"  # "standardize", "center", or "none" (group-mean imputation)

    # Lightning trainer
    accelerator: str = "auto"
    devices: int | str = "auto"
    precision: str = "32"
    gradient_clip_val: float = 0.0
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 50
    early_stopping_patience: int = 20
    checkpoint_top_k: int = 3
    num_workers: int = 0

    @classmethod
    def from_yaml_dict(cls, raw: dict) -> "MOSAConfig":
        """Parse the per-view mapping into OmicViewConfig objects before construction."""
        raw = dict(raw)
        check_unknown_keys(cls, raw, "model:")
        if "views" in raw:
            # 'name' is injected below, so it is never a valid key in the YAML.
            for view_name, vcfg in raw["views"].items():
                check_unknown_keys(
                    OmicViewConfig, vcfg, f"model.views.{view_name}", ignore={"name"}
                )
            raw["views"] = {
                name: OmicViewConfig(name=name, **vcfg)
                for name, vcfg in raw["views"].items()
            }
        return cls(**raw)

    def __post_init__(self):
        super().__post_init__()

        # YAML may parse floats as strings (e.g. "1e-5")
        for name in (
            "kl_weight", "kl_weight_final", "contrastive_weight", "adv_weight",
            "learning_rate", "adv_learning_rate", "lr_gamma", "view_dropout_prob",
            "scaler_sample_frac", "adv_focal_gamma",
        ):
            val = getattr(self, name)
            if not isinstance(val, float):
                object.__setattr__(self, name, float(val))

        # Structural validation
        if self.fusion_method not in _VALID_FUSION_METHODS:
            raise ConfigError(f"fusion_method must be one of {_VALID_FUSION_METHODS}, got '{self.fusion_method}'")
        if self.lr_scheduler not in _VALID_LR_SCHEDULERS:
            raise ConfigError(f"lr_scheduler must be one of {_VALID_LR_SCHEDULERS}, got '{self.lr_scheduler}'")
        if self.joint_latent_dim <= 0:
            raise ConfigError(f"joint_latent_dim must be positive, got {self.joint_latent_dim}")
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ConfigError(f"num_epochs must be positive, got {self.num_epochs}")
        if not 0.0 <= self.view_dropout_prob < 1.0:
            raise ConfigError(f"view_dropout_prob must be in [0, 1), got {self.view_dropout_prob}")
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.adv_weight > 0 and self.adv_learning_rate <= 0:
            raise ConfigError(
                f"adv_learning_rate must be positive when adv_weight > 0, got {self.adv_learning_rate}"
            )
        if not 0.0 < self.scaler_sample_frac <= 1.0:
            raise ConfigError(f"scaler_sample_frac must be in (0.0, 1.0], got {self.scaler_sample_frac}")
        if self.adv_focal_gamma < 0:
            raise ConfigError(f"adv_focal_gamma must be >= 0, got {self.adv_focal_gamma}")
        if self.preprocessing_mode not in _VALID_PREPROCESSING_MODES:
            raise ConfigError(
                f"preprocessing_mode must be one of {_VALID_PREPROCESSING_MODES}, "
                f"got '{self.preprocessing_mode}'"
            )

        # Lightning
        if self.precision not in _VALID_PRECISIONS:
            raise ConfigError(f"precision must be one of {_VALID_PRECISIONS}, got '{self.precision}'")
        if self.accelerator not in _VALID_ACCELERATORS:
            raise ConfigError(f"accelerator must be one of {_VALID_ACCELERATORS}, got '{self.accelerator}'")
        if isinstance(self.devices, int) and self.devices < 1:
            raise ConfigError(f"devices must be >= 1, got {self.devices}")
        if self.accumulate_grad_batches < 1:
            raise ConfigError(f"accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}")
        if self.gradient_clip_val < 0:
            raise ConfigError(f"gradient_clip_val must be >= 0, got {self.gradient_clip_val}")

        # Cross-field: PoE with a shared head requires equal last hidden dims across views
        if self.fusion_method == "poe" and self.views and self.poe_use_shared_head:
            last = {n: v.hidden_layer_dims[-1] for n, v in self.views.items()}
            if len(set(last.values())) > 1:
                raise ConfigError(
                    f"PoE fusion requires all views to have the same last hidden dim, got {last}"
                )

        if self.use_kl_scheduler and self.kl_warmup_epochs > self.num_epochs:
            warnings.warn(
                f"kl_warmup_epochs ({self.kl_warmup_epochs}) > num_epochs ({self.num_epochs}); "
                f"KL weight will never reach kl_weight_final",
                stacklevel=2,
            )

    def validate_against_data(self, data_cfg, eval_cfg, summary: dict) -> list[str]:
        """Check MOSA-specific value-level requirements against a data summary.

        Hard invariants raise DataError; soft ones (tissue/mutations/adversarial
        batch count) degrade silently at fit time, so they are returned as warnings.
        """
        obs_columns = summary.get("obs_columns", [])
        categories = summary.get("model_type_categories") or []

        if self.inference and self.target_batch:
            target = self.target_batch.strip()
            if target not in categories:
                raise DataError(
                    f"target_batch '{target}' not in model_type categories: {categories}"
                )

        result: list[str] = []
        if (data_cfg.use_tissue or self.contrastive_weight > 0) and "tissue" not in obs_columns:
            result.append(
                "No 'tissue' column in data; tissue conditioning will be disabled and "
                "the contrastive loss will degrade to zero."
            )
        if data_cfg.use_mutations and not any(c.startswith("mutation_") for c in obs_columns):
            result.append(
                "No 'mutation_*' columns in data; mutation conditioning will be disabled."
            )
        if self.adv_weight > 0 and len(categories) < 2:
            result.append(
                f"adv_weight > 0 but only {len(categories)} model_type categor"
                f"{'y' if len(categories) == 1 else 'ies'} in data; adversarial batch "
                f"correction will have no effect."
            )
        return result
