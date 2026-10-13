# MOSA VAE configuration

[Documentation](../../index.md)

These fields belong to `model:` when `model.type: mosa_vae`. Shared `data`, `evaluation`, `output_dir`, and `random_seed` settings are documented in [Library configuration](shared.md). For the architecture and training objectives, see [The MOSA VAE](../../models/mosa-vae.md).

## Architecture

| Key | Type | Default | Description |
|---|---|---|---|
| `views` | `dict[str, view config]` | `{}` | Per-view architecture, keyed by view name. Keys must match `data.views` exactly. |
| `joint_latent_dim` | `int` | `64` | Dimensionality of the shared latent space. Must be positive. |
| `fusion_method` | `str` | `concat` | `concat` concatenates embeddings; `poe` combines Gaussian experts. |
| `shared_hidden_layer_dims` | `list[int]` | `[]` | Hidden layers in the shared PoE posterior head. Not used by concatenation or direct per-view PoE heads. An empty list means none. |
| `poe_use_shared_head` | `bool` | `true` | PoE only: share posterior heads (`true`) or emit parameters per encoder (`false`). Shared heads require matching final encoder widths. |
| `view_dropout_prob` | `float` | `0.2` | Probability of dropping an entire view during training. Provides missing-view inputs during training. Must be in `[0, 1)`. |

Each `views` entry accepts:

| Key | Type | Default | Description |
|---|---|---|---|
| `hidden_layer_dims` | `list[int]` | `[512, 256]` | Encoder hidden layer sizes. The decoder mirrors them. Must be non-empty and all positive. |
| `loss_type` | `str` | `mean` | `mean` averages per-sample observed-feature MSE; `macro` averages those sample losses within source groups, then gives each group equal weight. |
| `dropout_p` | `float` | `0.1` | Dropout probability inside this specific view's encoder and decoder. Must be in `[0, 1)`. |
| `recon_weight` | `float` | `1.0` | Multiplier on this view's reconstruction loss. Must be positive. |

```yaml
model:
  views:
    gexp:
      hidden_layer_dims: [1024, 512]
      loss_type: macro
      dropout_p: 0.1
    meth:
      hidden_layer_dims: [1024, 512]
```

## Losses

| Key | Type | Default | Description |
|---|---|---|---|
| `kl_weight` | `float` | `0.00015625` | KL weight, defaulting to `0.01 / 64`. KL sums over latent dimensions before averaging samples; assess this weight together with latent width. |
| `kl_weight_final` | `float` | `0.00015625` | KL weight reached at the end of the warmup schedule. Only read when `use_kl_scheduler` is true. |
| `kl_warmup_epochs` | `int` | `0` | Epochs over which the KL weight transitions from `kl_weight` to `kl_weight_final`. Only read when `use_kl_scheduler` is true. |
| `use_kl_scheduler` | `bool` | `false` | Ramps the KL weight over `kl_warmup_epochs` instead of holding it fixed. |
| `contrastive_weight` | `float` | `0.0` | Tissue contrastive weight; zero disables it. For tissue-based supervision, provide tissue labels and enable `data.use_tissue`. |
| `adv_weight` | `float` | `0.0` | Adversarial source-classification weight; zero disables the discriminator. At least two source categories are needed for useful discrimination. |
| `adv_focal_gamma` | `float` | `0.0` | Focal-loss gamma for the adversarial cross-entropy. A value of `0` applies standard cross-entropy. Must be >= 0. |

## Optimiser

| Key | Type | Default | Description |
|---|---|---|---|
| `learning_rate` | `float` | `0.001` | Learning rate for the VAE optimiser. Must be positive. |
| `adv_learning_rate` | `float` | `0.001` | Learning rate for the discriminator optimiser. Must be positive when `adv_weight > 0`. |
| `lr_scheduler` | `str` | `none` | `none` keeps a fixed rate; `step` multiplies it by `lr_gamma` every `lr_step_size` epochs. |
| `lr_step_size` | `int` | `100` | Epochs between learning-rate decays. Only read when `lr_scheduler` is `step`. |
| `lr_gamma` | `float` | `0.5` | Multiplier applied at each decay step. Only read when `lr_scheduler` is `step`. |

## Training loop

| Key | Type | Default | Description |
|---|---|---|---|
| `num_epochs` | `int` | `200` | Maximum training epochs. Must be positive. The early stopping callback may end training sooner. |
| `batch_size` | `int` | `64` | Samples per mini-batch. Must be positive. |
| `scaler_sample_frac` | `float` | `1.0` | Reserved for the internal lazy `.zarr` path, which the current CLI does not construct; has no effect in CLI runs. Must be in `(0, 1]`. |
| `weighted_random_sampler` | `bool` | `true` | Inverse-frequency sampling to balance source groups in expectation. |
| `use_adv_class_weights` | `bool` | `true` | Weights the adversarial cross-entropy by inverse class frequency. |
| `inference` | `bool` | `false` | Export conditional reconstructions in `inference/` using the selected decoder source condition. |
| `target_batch` | `str` | `""` | Decoder source label when `inference` is true; must be a fitted category. Empty selects the first fitted category. |
| `preprocessing_mode` | `str` | `standardize` | `standardize`: per-feature z-score; `center`: subtract source-group means; `none`: impute missing values with group means. Statistics are fitted on training samples. |

## Lightning trainer

Write string precision values in quotes in [YAML](https://yaml.org/), for example `precision: "32"`. These settings apply to the VAE trainer, not to external models.

| Key | Type | Default | Description |
|---|---|---|---|
| `accelerator` | `str` | `auto` | [Lightning](https://lightning.ai/docs/pytorch/stable/) accelerator. Options are `auto`, `cpu`, `gpu`, or `mps`. |
| `devices` | `int \| str` | `auto` | Number of devices, or `auto`. Must be >= 1 when provided as an integer. |
| `precision` | `str` | `32` | Numeric precision. Options are `32`, `16-mixed`, or `bf16-mixed`. |
| `gradient_clip_val` | `float` | `0.0` | Gradient-norm clipping threshold. Setting it to `0` disables clipping. Must be >= 0. |
| `accumulate_grad_batches` | `int` | `1` | Number of batches to accumulate before taking an optimiser step. Must be >= 1. |
| `early_stopping_patience` | `int` | `20` | Epochs without validation improvement before training stops. This requires a validation split. |
| `checkpoint_top_k` | `int` | `3` | Number of top checkpoints to retain. Setting it to `0` disables monitored checkpoints. |
| `num_workers` | `int` | `0` | Number of DataLoader worker processes. Setting it to `0` loads data in the main process. |

