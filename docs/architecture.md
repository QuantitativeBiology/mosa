# Architecture Guide

This document describes how the MOSA model is structured internally, how data flows through it, and how the training loop works. It is intended for users who want to understand the model beyond the config file, or who plan to modify its internals.

## Overview

MOSA is a conditional Variational Autoencoder (VAE) that integrates multiple omic data sources into a shared latent space. Each omic modality has its own encoder and decoder. The per-view encoders produce embeddings that are fused into a single latent representation, which the decoders then use to reconstruct each modality.

## Source code layout

```
src/mosa/
  cli.py                  # Entry point: mosa train / mosa plot / mosa convert
  config.py               # MOSAConfig dataclass
  utils.py
  losses.py               # Loss functions
  callbacks.py            # Saves latent + reconstructions after training
  plot_utils.py           # Plotting and metrics
  convert.py              # CSV → MuData conversion

  model/
    mosavae.py            # MOSAVAE (LightningModule) — the main model
    encoder.py            # OmicEncoder + ViewDropout
    decoder.py            # OmicDecoder
    latent.py             # BaseLatentSpace, ConcatLatentSpace, PoELatentSpace
    discriminator.py      # Adversarial batch-correction discriminator
    mlp.py                # Shared MLP building block

  data/
    datamodule.py         # MuDataDataModule + MOSADataset + LazyZarrDataset
    batch.py              # MOSABatch dataclass + collate_fn
```

## Forward pass

The forward pass lives in `model/mosavae.py` and follows three stages.

### 1. Encode (per-view)

For each omic view, the encoder receives the omic data concatenated with conditional metadata:

```
input = [omic_features; conditionals]   # shape: [batch, input_dim + cond_dim]
embedding = encoder(input)              # shape: [batch, view_latent_dim]
```

- **View dropout**: During training, entire views are randomly zeroed with probability `view_dropout_prob`. This forces the model to learn useful representations even when some omics are missing.
- **Missing data**: Samples where a view is entirely absent (all features NaN) produce a zero embedding and are masked during fusion.

Each encoder is an MLP defined by `hidden_layer_dims` in the view config. For example, `[512, 256]` creates two hidden layers of size 512 and 256, using torch BatchNorm, PReLU activation, and dropout.

### 2. Fuse (latent space)

The per-view embeddings are combined into a joint latent space to produce three tensors:
- `mu` — mean of the approximate posterior
- `logvar` — log-variance of the approximate posterior
- `z` — sampled latent vector (via reparameterization trick)

Two fusion methods are available:

**Concatenation** (`fusion_method: concat`):
Concatenate all view embeddings into one long vector and project to mu and logvar through separate linear layers.

```
concat = [embedding_gexp; embedding_meth; ...]   # [batch, sum(view_dims)]
mu     = linear_mu(concat)                        # [batch, joint_latent_dim]
logvar = linear_logvar(concat)                    # [batch, joint_latent_dim]
```

**Product of Experts** (`fusion_method: poe`):
Each view produces its own mu and logvar through a shared projection head. These are combined via precision-weighted averaging, where precision = 1/variance. An isotropic N(0, I) prior is included as an additional "expert" so the posterior is well-defined even when views are missing.

```
For each view i:
    mu_i, logvar_i = shared_head(embedding_i)
    precision_i = exp(-logvar_i)

Joint precision  = 1 + sum(precision_i * view_mask_i)    # 1 from the prior
Joint mu         = sum(mu_i * precision_i * mask_i) / joint_precision
Joint logvar     = -log(joint_precision)
```

PoE requires all views to have the same last hidden dimension so they can share the projection head.

### 3. Decode (per-view)

Each decoder receives the latent vector `z` concatenated with the conditional metadata and produces a reconstruction:

```
input = [z; processed_conditionals]    # [batch, joint_latent_dim + cond_dim]
reconstruction = decoder(input)        # [batch, output_dim]
```

The decoder mirrors the encoder architecture (reversed hidden dims) but with a raw linear output layer (no activation), since the targets are z-scored continuous values.

Conditionals pass through a small projection (Linear + BatchNorm + PReLU) before being concatenated with `z`.

## Loss function

The total VAE loss combines several components:

```
loss = reconstruction + kl_weight * KL + contrastive_weight * contrastive - adv_weight * adversarial
```

| Component | Description | When active |
|-----------|-------------|-------------|
| **Reconstruction** | Per-feature MSE, masked to only count present features. Averaged across views. | Always |
| **KL divergence** | KL(q(z\|x) \|\| N(0, I)), encouraging the latent space to be smooth and regularized. | Always (but weight can be very small) |
| **Contrastive** | Pulls same-tissue samples closer in latent space using cosine similarity (pytorch-metric-learning). | `contrastive_weight > 0` |
| **Adversarial** | Cross-entropy loss from a discriminator that tries to predict model_type from z. The VAE tries to *fool* the discriminator (hence the negative sign). | `adv_weight > 0` |

### Macro loss

When `loss_type: macro` is set for a view, the MSE is balanced across model types. Instead of averaging over all samples equally, the loss first computes the mean MSE within each model type, then averages those group means. This prevents dominant groups (e.g., many Cell Lines vs few Organoids) from dominating the reconstruction objective.

### KL warmup

When enabled (`use_kl_scheduler: true`), the KL weight increases linearly from `kl_weight` to `kl_weight_final` over `kl_warmup_epochs`. This gives the model time to learn good reconstructions before the KL regularization pushes the latent space toward the prior.

## Training loop

MOSA uses **manual optimization** (PyTorch Lightning's `automatic_optimization = False`) to support a two-phase adversarial training step:

**Phase 1 — Discriminator update** (only when `adv_weight > 0`):
1. Forward pass the batch through encoders and latent space to get `z`
2. Pass `z.detach()` (no gradients to the VAE) through the discriminator
3. Compute cross-entropy loss against model_type labels
4. Update discriminator weights

**Phase 2 — VAE update**:
1. Compute reconstruction loss, KL divergence, and (optionally) contrastive loss
2. Pass `z` (with gradients) through the discriminator for the adversarial term
3. Combine into total loss: `recon + kl_weight * kl + contrastive_weight * contrastive - adv_weight * adversarial`
4. Update encoder, decoder, and latent space weights

The detachment in Phase 1 is critical: it trains the discriminator on the current latent space without affecting the VAE, creating the adversarial dynamic.

### Optimizer setup

- **VAE optimizer**: Adam over all encoder, decoder, and latent space parameters. Learning rate set by `learning_rate`.
- **Discriminator optimizer**: Separate Adam optimizer over discriminator parameters. Learning rate set by `adv_learning_rate`.
- **LR scheduling**: Optional StepLR applied to both optimizers at epoch boundaries.

### Multi-GPU training

When `trainer.devices > 1`, PyTorch Lightning enables Distributed Data Parallel (DDP):
- Gradients synchronized across GPUs at each step
- Data sharded across devices (each GPU processes a subset of batches)
- `sync_batchnorm` automatically enabled for correct batch statistics across GPUs
- SaveLatentAndReconCallback guarded with `is_global_zero` to prevent duplicate writes

Recommended: use `.zarr` format with `num_workers > 0` so each worker independently streams its batches from disk, avoiding memory bloat.

## Data pipeline

See [Data Pipeline & Batching](data-pipeline.md) for the full explanation of how data flows from MuData files through preprocessing, splitting, batching, and into the model.

Quick summary: `MuDataDataModule.setup()` handles loading, alignment, splitting, preprocessing, and batch construction. Key steps:

1. Load MuData file (.h5mu or .zarr)
2. Extract modalities, handle missing data
3. Build conditionals from .obs (model_type, tissue, mutations)
4. Train/val split (stratified by model_type)
5. Fit scalers on training data only
6. Compute class weights for imbalanced data
7. Create MOSADataset (eager) or LazyZarrDataset (lazy)

DataLoader + collate_fn produce MOSABatch objects ready for the model.

## Callbacks and outputs

After training completes, the `SaveLatentAndReconCallback` runs automatically:

1. Passes all training samples through the model in eval mode
2. Saves the latent vectors (`z`) and per-view reconstructions to CSV
3. Repeats for validation samples (if a validation set exists)

These CSV files are what the `mosa plot` command uses to generate visualizations.
