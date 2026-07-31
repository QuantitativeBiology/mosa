# Configuration Reference

All experiment parameters are defined in a single YAML config file.

## Minimal config

The shortest valid config defines one view and points to a MuData file:

```yaml
data_path: data/data.h5mu

views:
  gexp:
    hidden_layer_dims: [512, 256]
```

Everything else uses sensible defaults. The sections below document what those defaults are.

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
| `hidden_layer_dims` | `[512, 256]` | Hidden layer sizes for encoder and decoder MLPs. The last value determines the per-view embedding dimension before the joint latent bottleneck. |
| `loss_type` | `"mean"` | `"mean"`: standard MSE. `"macro"`: class-balanced MSE where each model_type contributes equally. |
| `dropout_p` | `0.1` | Dropout probability in encoder and decoder layers. |
| `discrete` | `false` | If true, skips z-score normalization for this view (for integer/count data). |

### Multiple views

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

View names must match modality names in your MuData file.

## Fusion method

Controls how per-view embeddings are combined into a single latent representation.

```yaml
fusion_method: concat       # "concat" or "poe"
joint_latent_dim: 64        # dimensionality of the shared latent space
poe_use_shared_head: true   # only relevant for poe: true = shared head, false = direct mu/logvar output
```

| Option | Default | Description |
|--------|---------|-------------|
| `fusion_method` | `"concat"` | `"concat"`: concatenate all view embeddings, then project to mu/logvar. `"poe"`: Product of Experts via precision-weighted averaging with an N(0, I) prior. |
| `joint_latent_dim` | `64` | Size of the latent vector `z`. |
| `poe_use_shared_head` | `true` | Only used when `fusion_method: poe`. `true`: each view encoder outputs an embedding and a shared projection head produces `mu/logvar`. `false`: each view encoder ends with a linear layer that outputs `mu/logvar` directly. |

When using `poe` with `poe_use_shared_head: true`, all views must have the same last value in `hidden_layer_dims`. Validated at config load time. The shared head then maps that embedding to `2 * joint_latent_dim` with a linear output layer. With `poe_use_shared_head: false`, each encoder must output a vector of size `2 * joint_latent_dim` directly, also with a linear output layer.

## Conditionals

Metadata from MuData `.obs` is automatically converted to conditional vectors: `model_type` (always one-hot encoded), `tissue` (one-hot if column exists), and `mutation_*` columns (if present). No config needed — these dimensions are derived from the data at training time.

## Loss weights

```yaml
kl_weight: 0.01              # weight on KL divergence term
contrastive_weight: 0.0      # weight on tissue-supervised contrastive loss
adv_weight: 0.0              # weight on adversarial batch-correction loss
```

| Option | Default | Description |
|--------|---------|-------------|
| `kl_weight` | `0.01` | Scaling factor for KL divergence. Higher values push the latent space closer to N(0, I). |
| `contrastive_weight` | `0.0` | Weight on contrastive loss (cosine similarity, supervised by tissue). 0 to disable. |
| `adv_weight` | `0.0` | Weight on adversarial loss for batch correction. > 0 enables the discriminator. |

Total VAE loss = reconstruction + `kl_weight` * KL + `contrastive_weight` * contrastive - `adv_weight` * adversarial

The adversarial term is subtracted because the VAE tries to fool the discriminator.

## KL warmup schedule

Gradually increases the KL weight during training to avoid posterior collapse.

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
| `kl_warmup_epochs` | `0` | Epochs over which to interpolate from `kl_weight` to `kl_weight_final`. |

If `kl_warmup_epochs` exceeds `num_epochs`, MOSA warns that the KL weight will never reach its final value.

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
| `lr_scheduler` | `"none"` | `"none"`: constant LR. `"step"`: multiply LR by `lr_gamma` every `lr_step_size` epochs. |
| `lr_step_size` | `100` | Epochs between StepLR decay steps. |
| `lr_gamma` | `0.5` | Multiplicative LR decay factor. |

## Data & training

```yaml
data_path: data/data.h5mu        # path to MuData file (.h5mu or .zarr)
mask_layer_name: mask            # layer name in .layers for per-feature masks
scaler_sample_frac: 1.0          # zarr only: fraction of training samples for fitting StandardScaler

num_epochs: 200
batch_size: 64
view_dropout_prob: 0.2           # probability of dropping an entire view during training
random_seed: 42
```

| Option | Default | Description |
|--------|---------|-------------|
| `data_path` | `""` | Path to MuData file. See [Data Pipeline](data-pipeline.md#converting-csvs-to-mudata). |
| `mask_layer_name` | `"mask"` | Layer name in `.layers` for per-feature boolean masks. |
| `scaler_sample_frac` | `1.0` | Fraction of training samples used to fit the StandardScaler. Only relevant for zarr datasets — reduces memory and I/O when fitting on millions of samples is expensive (e.g. `0.1`). Has no effect for h5mu, which is already in memory. |
| `num_epochs` | `200` | Maximum training epochs. |
| `batch_size` | `64` | Samples per batch. |
| `view_dropout_prob` | `0.2` | Probability of zeroing an entire view during training. 0 to disable. |
| `random_seed` | `42` | Seed for all RNGs (Python, NumPy, PyTorch, Lightning). |

## Evaluation

How data is held out for assessment. Optional block; omitting it takes these
defaults. Nested under `evaluation:`, not under `model:`.

```yaml
evaluation:
  test_size: 0.1
  n_folds: 5
  strategy: stratified
  shuffle: true
```

| Option | Default | Description |
|--------|---------|-------------|
| `test_size` | `0.1` | Validation split fraction for `mosa train`. 0 disables validation and early stopping. |
| `n_folds` | `5` | Folds for `mosa cross-validate` and each `mosa optimize` trial. Minimum 2. |
| `strategy` | `"stratified"` | `stratified` balances `model_type` across folds; `kfold` ignores it. |
| `shuffle` | `true` | `false` assigns folds as contiguous blocks of sample order, making fold composition seed-independent. |

`mosa cross-validate` and `mosa optimize` accept `--folds`, `--strategy`, and
`--no-shuffle` to override the block for one run.

## Trainer (PyTorch Lightning)

Hardware and Lightning-specific settings, nested under `trainer:`.

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
| `accelerator` | `"auto"` | Auto-detects GPU/MPS/CPU. Force with `"gpu"` or `"mps"`. |
| `devices` | `"auto"` | `"auto"` uses one device. Set to 2, 4, etc. for multi-GPU DDP. |
| `precision` | `"32"` | `"16-mixed"` or `"bf16-mixed"` for faster mixed-precision training. |
| `gradient_clip_val` | `0.0` | Max gradient norm. > 0 to prevent gradient explosion. |
| `accumulate_grad_batches` | `1` | Simulate larger batches by accumulating gradients over N steps. |
| `log_every_n_steps` | `50` | How often (in steps) to log metrics. |
| `early_stopping_patience` | `20` | Epochs without `val/loss` improvement before stopping. Only when `test_size > 0`. |
| `checkpoint_top_k` | `3` | Number of best checkpoints to keep. |
| `num_workers` | `0` | Parallel data-loading workers. 0 = main process only. Set to 4-8 for large datasets or multi-GPU. |

For multi-GPU with `.zarr` and `num_workers > 0`, each worker reads independently (recommended for large data).

## Output

```yaml
output_dir: outputs/my_experiment
inference: false
target_batch: ""
```

| Option | Default | Description |
|--------|---------|-------------|
| `inference` | `false` | If true, runs corrected inference by forcing all samples to one target `model_type` conditional. |
| `target_batch` | `""` | Target `model_type` for corrected inference. Empty uses the first available category. |

After training:

```
outputs/my_experiment/
  lightning_logs/version_0/
    metrics.csv
  train/
    latent.parquet
    recon_<view>.parquet
  val/
    latent.parquet
    recon_<view>.parquet
  full/
    latent.parquet
    recon_<view>.parquet
  inference/
    latent.parquet
    recon_<view>.parquet
  mosa-<epoch>-<val_loss>.ckpt
```

After `mosa plot`:

```
outputs/my_experiment/
  plots/
    umap_z.png
    umap_recon_<view>.png
    umap_recon_corrected_<view>.png
    loss_total.png
    loss_kl.png
    loss_adv.png
    loss_disc.png
    mse_<view>.png
    input_recon_sample_<view>_*.png
    input_recon_feature_<view>_*.png
  metrics/
    clustering_metrics.csv
```

## Complete example

See `configs/example.yaml` for a fully commented template.
