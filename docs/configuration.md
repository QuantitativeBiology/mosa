# Configuration Reference

All experiment parameters are defined in a single YAML config file. This page documents every option, its default value, and what it controls.

## Minimal config

The shortest valid config defines one view and points to a MuData file:

```yaml
data_path: data/data.h5mu

views:
  gexp:
    hidden_layer_dims: [512, 256]
```

Everything else uses sensible defaults. The sections below document what those defaults are and when you might want to change them.

## Views

Each entry under `views:` defines one omic modality. The key (e.g., `gexp`) must match a modality name in the MuData file.

```yaml
views:
  gexp:
    hidden_layer_dims: [512, 256]     # encoder/decoder MLP layer sizes
    loss_type: mean                   # "mean" or "macro"
    dropout_p: 0.1                    # dropout rate in encoder/decoder
    discrete: false                   # set true for count data (e.g., copy number)
```

| Option | Default | Description |
|--------|---------|-------------|
| `hidden_layer_dims` | `[512, 256]` | Sizes of hidden layers in the encoder and decoder MLPs. The **last value** also determines the per-view embedding dimension before the joint latent bottleneck. |
| `loss_type` | `"mean"` | `"mean"`: standard MSE averaged over present features. `"macro"`: class-balanced MSE where each model_type contributes equally regardless of sample count. |
| `dropout_p` | `0.1` | Dropout probability applied in encoder and decoder layers. Must be in [0, 1). |
| `discrete` | `false` | If `true`, skips z-score normalization for this view (useful for integer/count data). |

**Auto-set fields** (leave at 0):
- `input_dim`: Set automatically from the number of features in the MuData modality.
- `output_dim`: Set automatically to match `input_dim`. Only override if you want the decoder to output a different dimensionality.

### Adding multiple views

```yaml
views:
  gexp:
    hidden_layer_dims: [256, 128]
    loss_type: macro

  meth:
    hidden_layer_dims: [256, 128]
    loss_type: macro

  proteomics:
    hidden_layer_dims: [512, 256]
```

View names must match modality names in your MuData file. MOSA builds a separate encoder and decoder for each.

## Fusion method

Controls how per-view embeddings are combined into a single latent representation.

```yaml
fusion_method: concat       # "concat" or "poe"
joint_latent_dim: 64        # dimensionality of the shared latent space
```

| Option | Default | Description |
|--------|---------|-------------|
| `fusion_method` | `"concat"` | `"concat"`: concatenate all view embeddings, then project to mu/logvar. `"poe"`: Product of Experts, where each view produces its own mu/logvar and they are fused via precision-weighted averaging with an N(0, I) prior. |
| `joint_latent_dim` | `64` | Size of the shared bottleneck. This is the dimensionality of the latent vector `z` used for downstream analysis. |

**Important**: When using `poe`, all views must have the **same last value** in `hidden_layer_dims` (e.g., all ending in 128). This is validated at config load time.

## Conditionals

Metadata from the MuData `.obs` is automatically converted to conditional vectors:
- `model_type` — always one-hot encoded
- `tissue` — one-hot encoded if column exists
- `mutation_*` columns — included if present

Configure which to use:

```yaml
# tissue conditional is on by default; set false to exclude it
# mutations are included if mutation_* columns exist in .obs
```

The `conditional_dim` is computed automatically from the MuData file.

## Loss weights

```yaml
kl_weight: 0.01              # weight on KL divergence term
contrastive_weight: 0.0      # weight on tissue-supervised contrastive loss
adv_weight: 0.0              # weight on adversarial batch-correction loss
```

| Option | Default | Description |
|--------|---------|-------------|
| `kl_weight` | `0.01` | Scaling factor for the KL divergence term. Higher values push the latent space closer to N(0, I) but may reduce reconstruction quality. |
| `contrastive_weight` | `0.0` | Weight on the contrastive loss applied to latent means, supervised by tissue labels. Set to 0 to disable. |
| `adv_weight` | `0.0` | Weight on the adversarial loss for batch correction. Setting this to a value > 0 enables the discriminator network. |

**Total VAE loss** = reconstruction + `kl_weight` * KL + `contrastive_weight` * contrastive - `adv_weight` * adversarial

The adversarial term is subtracted because the VAE tries to *fool* the discriminator (make batches indistinguishable).

## KL warmup schedule

Gradually increase the KL weight during training to avoid posterior collapse.

```yaml
use_kl_scheduler: true
kl_weight: 0.00001          # starting KL weight
kl_weight_final: 0.001      # KL weight after warmup
kl_warmup_epochs: 100       # epochs over which to linearly increase
```

| Option | Default | Description |
|--------|---------|-------------|
| `use_kl_scheduler` | `false` | Enable linear KL warmup. |
| `kl_weight` | `0.01` | Starting KL weight (at epoch 0). |
| `kl_weight_final` | `0.01` | Final KL weight (reached at `kl_warmup_epochs`). |
| `kl_warmup_epochs` | `0` | Number of epochs over which to linearly interpolate from `kl_weight` to `kl_weight_final`. |

If `kl_warmup_epochs` exceeds `num_epochs`, MOSA prints a warning: the KL weight will never reach its final value.

## Optimization

```yaml
learning_rate: 0.001
adv_learning_rate: 0.001     # only used when adv_weight > 0
lr_scheduler: none           # "none" or "step"
lr_step_size: 100            # epochs between LR decay steps
lr_gamma: 0.5                # multiplicative decay factor
```

| Option | Default | Description |
|--------|---------|-------------|
| `learning_rate` | `0.001` | Learning rate for the VAE optimizer (Adam). |
| `adv_learning_rate` | `0.001` | Learning rate for the discriminator optimizer. Only used when `adv_weight > 0`. |
| `lr_scheduler` | `"none"` | `"none"`: constant learning rate. `"step"`: multiply LR by `lr_gamma` every `lr_step_size` epochs. |
| `lr_step_size` | `100` | Epochs between StepLR decay steps. |
| `lr_gamma` | `0.5` | Factor to multiply the learning rate at each step. |

## Data & Training

```yaml
data_path: data/data.h5mu        # path to MuData file (.h5mu or .zarr)
mask_layer_name: mask            # layer name in .layers for per-feature masks
scaler_sample_frac: 1.0          # fraction of training data to fit scaler on

num_epochs: 200
batch_size: 64
test_size: 0.1                   # validation split fraction
view_dropout_prob: 0.2           # probability of dropping an entire view during training
random_seed: 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `data_path` | `""` | Path to MuData file (.h5mu for eager loading or .zarr for lazy loading). |
| `mask_layer_name` | `"mask"` | Layer name in MuData `.layers` containing per-feature boolean masks. |
| `scaler_sample_frac` | `1.0` | Fraction of training data to fit StandardScaler on (e.g., 0.5 for large datasets to speed up scaler fitting). |
| `num_epochs` | `200` | Maximum number of training epochs. |
| `batch_size` | `64` | Number of samples per training batch. |
| `test_size` | `0.1` | Fraction of data held out for validation. Set to `0` to use all data for training (disables early stopping and validation). |
| `view_dropout_prob` | `0.2` | During training, each view is randomly zeroed with this probability. This forces the model to learn from incomplete view combinations. Set to `0.0` to disable. |
| `random_seed` | `42` | Seed for all random number generators (Python, NumPy, PyTorch, Lightning). |

## Trainer (PyTorch Lightning)

Hardware and Lightning-specific settings. Nested under `trainer:`.

```yaml
trainer:
  accelerator: auto           # auto | cpu | gpu | mps
  devices: auto               # auto | integer (for multi-GPU)
  precision: "32"             # "32" | "16-mixed" | "bf16-mixed"
  gradient_clip_val: 0.0      # max gradient norm (0 = disabled)
  accumulate_grad_batches: 1  # gradient accumulation steps
  log_every_n_steps: 50
  early_stopping_patience: 20
  checkpoint_top_k: 3
  num_workers: 0              # DataLoader worker processes
```

| Option | Default | Description |
|--------|---------|-------------|
| `accelerator` | `"auto"` | `"auto"` picks GPU/MPS/CPU automatically. Use `"gpu"` to force GPU, `"mps"` for Apple Silicon. |
| `devices` | `"auto"` | `"auto"` uses one device. Set to `2`, `4`, etc. for multi-GPU Distributed Data Parallel (DDP). |
| `precision` | `"32"` | `"32"` for float32. `"16-mixed"` or `"bf16-mixed"` for faster mixed-precision training. |
| `gradient_clip_val` | `0.0` | Max gradient norm. Set > 0 to prevent gradient explosion. |
| `accumulate_grad_batches` | `1` | Simulate larger batches by accumulating gradients over N batches before updating. |
| `log_every_n_steps` | `50` | How often (in training steps) to log metrics. |
| `early_stopping_patience` | `20` | Stop after this many epochs without `val/loss` improvement. Only when `test_size > 0`. |
| `checkpoint_top_k` | `3` | Keep the N best model checkpoints. |
| `num_workers` | `0` | Parallel data-loading workers. `0` = main process. Set to 4-8 for large datasets or multi-GPU. |

**Multi-GPU note**: Set `devices` to an integer (e.g., 4) to enable Distributed Data Parallel. For `.zarr` format with `num_workers > 0`, each worker reads independently (recommended for large data).

## Output

```yaml
output_dir: outputs/my_experiment
inference: false
target_batch: ""
```

| Option | Default | Description |
|--------|---------|-------------|
| `inference` | `false` | If `true`, runs corrected inference by forcing all samples to one target `model_type` conditional and saves results under `inference/`. |
| `target_batch` | `""` | Target `model_type` for corrected inference. If empty, uses the first available `model_type` category. |

After training, this directory contains:

```
outputs/my_experiment/
  lightning_logs/version_0/
    metrics.csv                       # all logged metrics per epoch
  train/
    latent.csv                        # latent z for training samples
    recon_<view>.csv                  # reconstructed features (training)
  val/
    latent.csv                        # latent z for validation samples
    recon_<view>.csv                  # reconstructed features (validation)
  full/
    latent.csv                        # latent z for all samples (original conditionals)
    recon_<view>.csv                  # reconstructed features for all samples (original conditionals)
  inference/
    latent.csv                        # corrected latent z for all samples (forced target model_type)
    recon_<view>.csv                  # corrected reconstructed features (forced target model_type)
  mosa-<epoch>-<val_loss>.ckpt       # model checkpoints (if val enabled)
```

After running `mosa plot`:

```
outputs/my_experiment/
  plots/
    umap_z.png                        # latent space UMAP
    umap_recon_<view>.png             # per-view reconstruction UMAP
    umap_recon_corrected_<view>.png   # per-view corrected reconstruction UMAP
    loss_total.png                    # composite loss curve
    loss_kl.png                       # KL divergence curve
    loss_adv.png                      # adversarial loss curve (if enabled)
    loss_disc.png                     # discriminator loss curve (if enabled)
    mse_<view>.png                    # per-view MSE with model_type breakdown
    input_recon_sample_<view>_*.png   # sample-level reconstruction scatter
    input_recon_feature_<view>_*.png  # feature-level reconstruction scatter
  metrics/
    clustering_metrics.csv            # Calinski-Harabasz and Davies-Bouldin scores
```

## Complete example

See `configs/example.yaml` for a fully commented template, or `configs/depmap_example.yaml` for a real experiment config with PoE fusion, KL warmup, and adversarial training.
