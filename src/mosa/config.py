from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_VALID_FUSION_METHODS = ("concat", "poe")
_VALID_LOSS_TYPES = ("mean", "macro")
_VALID_LR_SCHEDULERS = ("none", "step")
_VALID_PRECISIONS = ("32", "16-mixed", "bf16-mixed")
_VALID_ACCELERATORS = ("auto", "cpu", "gpu", "mps")


@dataclass
class ViewConfig:
    name: str
    hidden_layer_dims: list[int] = field(default_factory=lambda: [512, 256])
    loss_type: str = "mean"
    dropout_p: float = 0.1
    discrete: bool = False

    def __post_init__(self):
        if not self.hidden_layer_dims:
            raise ValueError(f"View '{self.name}': hidden_layer_dims must not be empty")
        if any(d <= 0 for d in self.hidden_layer_dims):
            raise ValueError(f"View '{self.name}': all hidden_layer_dims must be positive")
        if self.loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(
                f"View '{self.name}': loss_type must be one of {_VALID_LOSS_TYPES}, "
                f"got '{self.loss_type}'"
            )
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError(f"View '{self.name}': dropout_p must be in [0, 1), got {self.dropout_p}")


@dataclass
class TrainerConfig:
    # Hardware
    accelerator: str = "auto"   # "auto" | "cpu" | "gpu" | "mps"
    devices: int | str = "auto" # "auto" | int (number of devices)
    precision: str = "32"       # "32" | "16-mixed" | "bf16-mixed"

    # Optimisation
    gradient_clip_val: float = 0.0      # 0 disables gradient clipping
    accumulate_grad_batches: int = 1    # steps before each optimiser update

    # Logging & checkpointing
    log_every_n_steps: int = 50
    early_stopping_patience: int = 20   # epochs with no val/loss improvement before stopping
    checkpoint_top_k: int = 3           # number of best checkpoints to keep

    # DataLoader
    num_workers: int = 0                # worker processes for data loading; 0 = main process

    def __post_init__(self):
        if self.precision not in _VALID_PRECISIONS:
            raise ValueError(
                f"trainer.precision must be one of {_VALID_PRECISIONS}, got '{self.precision}'"
            )
        if self.accelerator not in _VALID_ACCELERATORS:
            raise ValueError(
                f"trainer.accelerator must be one of {_VALID_ACCELERATORS}, got '{self.accelerator}'"
            )
        if isinstance(self.devices, int) and self.devices < 1:
            raise ValueError(f"trainer.devices must be >= 1, got {self.devices}")
        if self.accumulate_grad_batches < 1:
            raise ValueError(
                f"trainer.accumulate_grad_batches must be >= 1, got {self.accumulate_grad_batches}"
            )
        if self.gradient_clip_val < 0:
            raise ValueError(
                f"trainer.gradient_clip_val must be >= 0, got {self.gradient_clip_val}"
            )


@dataclass
class MOSAConfig:
    views: dict[str, ViewConfig] = field(default_factory=dict)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    fusion_method: str = "concat"  # "concat" or "poe"
    joint_latent_dim: int = 64
    shared_hidden_layer_dims: list[int] = field(default_factory=list)  # intermediate dims for PoE shared head
    poe_use_shared_head: bool = True  # PoE: True = shared head, False = direct mu/logvar per view
    view_dropout_prob: float = 0.2
    kl_weight: float = 0.01
    kl_weight_final: float = 0.01
    kl_warmup_epochs: int = 0
    use_kl_scheduler: bool = False
    contrastive_weight: float = 0.0
    adv_weight: float = 0.0
    learning_rate: float = 1e-3
    adv_learning_rate: float = 1e-3
    lr_scheduler: str = "none"
    lr_step_size: int = 100
    lr_gamma: float = 0.5
    num_epochs: int = 200
    batch_size: int = 64
    n_folds: int = 5
    test_size: float = 0.1
    random_seed: int = 42
    data_path: str = ""                     # path to .h5mu/.zarr file
    mask_layer_name: str = "mask"           # layer name for per-feature masks
    scaler_sample_frac: float = 1.0         # fraction of training data for fitting StandardScaler
    use_tissue: bool = True                 # include tissue as conditional (model_type always included)
    use_mutations: bool = True              # include mutation_* columns as conditionals
    weighted_random_sampler: bool = True    # use WeightedRandomSampler to balance model_type in training
    use_adv_class_weights: bool = True      # use class weights in adversarial cross-entropy loss
    output_dir: str = "outputs"
    inference: bool = False
    target_batch: str = ""

    def __post_init__(self):
        # Coerce fields that YAML may parse as strings (e.g. "1e-5")
        for name in (
            "kl_weight", "kl_weight_final", "contrastive_weight", "adv_weight",
            "learning_rate", "adv_learning_rate", "lr_gamma", "view_dropout_prob",
            "test_size", "scaler_sample_frac",
        ):
            val = getattr(self, name)
            if not isinstance(val, float):
                object.__setattr__(self, name, float(val))

        # --- Structural validation ---
        if self.fusion_method not in _VALID_FUSION_METHODS:
            raise ValueError(
                f"fusion_method must be one of {_VALID_FUSION_METHODS}, "
                f"got '{self.fusion_method}'"
            )
        if self.lr_scheduler not in _VALID_LR_SCHEDULERS:
            raise ValueError(
                f"lr_scheduler must be one of {_VALID_LR_SCHEDULERS}, "
                f"got '{self.lr_scheduler}'"
            )
        if self.joint_latent_dim <= 0:
            raise ValueError(f"joint_latent_dim must be positive, got {self.joint_latent_dim}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if not 0.0 <= self.test_size < 1.0:
            raise ValueError(f"test_size must be in [0, 1), got {self.test_size}")
        if not 0.0 <= self.view_dropout_prob < 1.0:
            raise ValueError(f"view_dropout_prob must be in [0, 1), got {self.view_dropout_prob}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.adv_weight > 0 and self.adv_learning_rate <= 0:
            raise ValueError(
                f"adv_learning_rate must be positive when adv_weight > 0, "
                f"got {self.adv_learning_rate}"
            )
        if not 0.0 < self.scaler_sample_frac <= 1.0:
            raise ValueError(
                f"scaler_sample_frac must be in (0.0, 1.0], got {self.scaler_sample_frac}"
            )

        # --- Cross-field validation ---
        if self.fusion_method == "poe" and self.views and self.poe_use_shared_head:
            last_dims = {name: vc.hidden_layer_dims[-1] for name, vc in self.views.items()}
            if len(set(last_dims.values())) > 1:
                raise ValueError(
                    f"PoE fusion requires all views to have the same last hidden dim, "
                    f"got {last_dims}"
                )

        if self.views and not self.data_path:
            raise ValueError(
                "data_path is required when views are configured"
            )

        # --- Warnings for suspicious but valid configs ---
        if self.use_kl_scheduler and self.kl_warmup_epochs > self.num_epochs:
            warnings.warn(
                f"kl_warmup_epochs ({self.kl_warmup_epochs}) > num_epochs ({self.num_epochs}); "
                f"KL weight will never reach kl_weight_final",
                stacklevel=2,
            )

    def validate_paths(self) -> None:
        """Check that all referenced files exist. Call before training."""
        errors = []

        # Check MuData file exists
        if not self.data_path:
            errors.append("data_path is required but not set")
        elif not (Path(self.data_path).is_file() or Path(self.data_path).is_dir()):
            errors.append(f"data_path not found: {self.data_path}")

        if not self.views:
            errors.append("At least one view must be configured")

        if errors:
            raise FileNotFoundError(
                "Config validation failed:\n  " + "\n  ".join(errors)
            )
