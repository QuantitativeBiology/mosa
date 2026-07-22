# Architecture Guide

For users who want to understand the model beyond the config file, or who plan to modify its internals.

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
    layers.py             # Shared MLP building block

  data/
    datamodule.py         # MuDataDataModule + MOSADataset + LazyZarrDataset
    batch.py              # MOSABatch dataclass + collate_fn
```

## Forward pass

The forward pass lives in `model/mosavae.py` and follows three stages.

### Encode (per-view)

For each omic view, the encoder receives the omic data only; conditional metadata enters on the decoder side:

```
embedding = encoder(omic_features)   # shape: [batch, view_latent_dim]
```

During training, entire views are randomly zeroed with probability `view_dropout_prob` (view dropout), forcing the model to learn useful representations even when some omics are missing. Samples where a view is entirely absent produce a zero embedding and are masked during fusion.

Each encoder is an MLP defined by `hidden_layer_dims` in the view config. For example, `[512, 256]` creates two hidden layers of size 512 and 256, using BatchNorm, PReLU, and dropout.

### Fuse (latent space)

The per-view embeddings are combined into a joint latent space to produce `mu` (posterior mean), `logvar` (log-variance), and `z` (sampled via reparameterization).

Two fusion methods are available.

Concatenation (`fusion_method: concat`) concatenates all view embeddings into one vector and projects to mu and logvar through separate linear layers:

```
concat = [embedding_gexp; embedding_meth; ...]   # [batch, sum(view_dims)]
mu     = linear_mu(concat)                        # [batch, joint_latent_dim]
logvar = linear_logvar(concat)                    # [batch, joint_latent_dim]
```

Product of Experts (`fusion_method: poe`) supports two encoder-to-posterior variants:

`poe_use_shared_head: true` means each view encoder stops at a per-view embedding, typically `hidden_layer_dims[-1]`, and a shared projection head turns that embedding into `mu` and `logvar`.

`poe_use_shared_head: false` means each view encoder emits `mu` and `logvar` directly with a final linear layer of size `2 * joint_latent_dim`, and PoE combines those per-view posteriors via precision-weighted averaging.

An isotropic N(0, I) prior acts as an additional expert so the posterior remains well-defined when views are missing:

```
For each view i:
    mu_i, logvar_i = shared_head(embedding_i)
    precision_i = exp(-logvar_i)

Joint precision  = 1 + sum(precision_i * view_mask_i)    # 1 from the prior
Joint mu         = sum(mu_i * precision_i * mask_i) / joint_precision
Joint logvar     = -log(joint_precision)
```

PoE with a shared head requires all views to have the same last hidden dimension so they can share the projection head. In that case, the shared head itself has no activation on its output layer. Direct per-view `mu/logvar` output only requires that each encoder emits a vector of size `2 * joint_latent_dim`, again with no activation on the final layer that produces the statistics.

### Decode (per-view)

Each decoder receives `z` concatenated with conditional metadata and produces a reconstruction:

```
input = [z; processed_conditionals]    # [batch, joint_latent_dim + cond_dim]
reconstruction = decoder(input)        # [batch, output_dim]
```

The decoder mirrors the encoder architecture (reversed hidden dims) with a raw linear output layer, since the targets are z-scored continuous values. Conditionals pass through a small projection (Linear + BatchNorm + PReLU) before concatenation with `z`.

## Loss function

The total VAE loss:

```
loss = reconstruction + kl_weight * KL + contrastive_weight * contrastive - adv_weight * adversarial
```

Reconstruction is per-feature MSE, masked to only count present features, averaged across views. KL divergence pushes the latent space toward N(0, I). The contrastive term (pytorch-metric-learning) pulls same-tissue samples closer via cosine similarity. The adversarial term is cross-entropy from a discriminator predicting model_type from z — subtracted because the VAE tries to fool it. Contrastive and adversarial losses are only active when their respective weights are > 0.

### Macro loss

When `loss_type: macro` is set for a view, MSE is balanced across model types: the loss computes mean MSE within each model type first, then averages those group means. This prevents dominant groups (e.g., many Cell Lines vs few Organoids) from dominating the reconstruction objective.

### KL warmup

When enabled (`use_kl_scheduler: true`), the KL weight increases linearly from `kl_weight` to `kl_weight_final` over `kl_warmup_epochs`, giving the model time to learn good reconstructions before regularization kicks in.

## Training loop

MOSA uses manual optimization (Lightning's `automatic_optimization = False`) for two-phase adversarial training.

Phase 1 — discriminator update (only when `adv_weight > 0`): forward pass to get `z`, pass `z.detach()` through the discriminator, compute cross-entropy against model_type labels, update discriminator weights. The detachment trains the discriminator on the current latent space without affecting the VAE.

Phase 2 — VAE update: compute reconstruction, KL, and optionally contrastive loss. Pass `z` (with gradients this time) through the discriminator for the adversarial term. Combine into total loss and update encoder, decoder, and latent space weights.

### Optimizers

The VAE uses Adam over all encoder, decoder, and latent space parameters (`learning_rate`). The discriminator gets a separate Adam optimizer (`adv_learning_rate`). Optional StepLR scheduling applies to both.

### Multi-GPU

When `trainer.devices > 1`, Lightning enables DDP: gradients are synchronized across GPUs, data is sharded across devices, `sync_batchnorm` is enabled automatically, and the save callback is guarded to rank 0 only. Use `.zarr` with `num_workers > 0` so each worker streams independently from disk.

## Data pipeline

### MuData as the data layer

MOSA uses [MuData](https://mudata.readthedocs.io/) as its on-disk format — a multi-modal extension of AnnData that stores multiple omic modalities, shared sample metadata, and per-feature masks in a single container. This replaced separate per-modality CSVs and a samplesheet. All modalities, metadata, and masks are co-located, incomplete data is first-class (per-feature boolean masks and per-view presence indicators), and two storage backends are supported: `.h5mu` (eager, in-memory) and `.zarr` (lazy, per-batch).

The `mosa convert` CLI converts raw CSVs into MuData. See [Data Pipeline](data-pipeline.md#converting-csvs-to-mudata) for details.

### Loading strategies

`MuDataDataModule` auto-detects the format from the file extension.

For h5mu, `mudata.read()` loads everything into memory. Dense arrays are extracted, scalers fitted, and `MOSADataset` objects created with pre-tensored data. Straightforward for datasets that fit in RAM.

For zarr, only `.obs` metadata and `.var` feature names are read at setup. A `LazyZarrDataset` stores sample indices and scaler parameters; actual feature data is read from the zarr store per batch via vectorized slicing (`__getitems__`). Each DataLoader worker opens its own zarr handle, making this safe for multi-process and multi-GPU training.

### Processing pipeline

`MuDataDataModule.setup()` loads and validates the MuData structure, extracts modalities and handles missing samples via masks, builds conditional vectors from `.obs` (one-hot model_type + tissue + mutation columns), does a stratified train/val split by model_type, fits StandardScaler on training data only, computes inverse-frequency class weights, and creates the dataset objects.

DataLoader + `collate_fn` produce `MOSABatch` objects ready for the model. See [Data Pipeline & Batching](data-pipeline.md) for details.

## Callbacks and outputs

After training, `SaveLatentAndReconCallback` runs on rank 0: it passes training, validation, full, and (if `inference: true`) corrected samples through the model in eval mode and saves latent vectors and per-view reconstructions to parquet. These are what `mosa plot` uses.
