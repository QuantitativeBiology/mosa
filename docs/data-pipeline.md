# Data Pipeline & Batching

How data flows from .h5mu/.zarr files to GPU-ready batches.

## Quick overview

```
MuData file → MuDataDataModule.setup() → MOSADataset/LazyZarrDataset
  → DataLoader with collate_fn → MOSABatch → model.forward()
```

## MuData format

MOSA expects data in MuData format (`.h5mu` or `.zarr`):

- `.obs` — sample metadata (model_id, model_type, tissue, mutation_*)
- `.mod[view_name]` — each omic modality (gexp_voom, meth_combat, etc.)
  - `.X` — feature matrix [samples, features]
  - `.layers['mask']` — boolean: True where data exists
- View names in config must match modality names in .mod

Convert from CSVs:

```bash
mosa convert --samplesheet data/samplesheet.csv \
  --view gexp_voom:data/gexp.csv \
  --view meth_combat:data/meth.csv \
  --output data.h5mu
```

Or use `.zarr` for lazy loading on large datasets:

```bash
mosa convert ... --output data.zarr --format zarr
```

## Setup process

`MuDataDataModule.setup()` does the heavy lifting:

1. **Load & validate** — checks MuData structure, required columns
2. **Extract modalities** — converts sparse to dense, handles missing samples
3. **Build conditionals** — one-hot encode model_type (required), tissue (optional), mutations (optional) → concatenate into single vector per sample
4. **Train/val split** — stratified by model_type to preserve class proportions
5. **Fit scalers** — StandardScaler fitted on training data only (prevents leakage)
6. **Compute class weights** — inverse-frequency weighting for rare classes
7. **Create datasets** — MOSADataset (eager) or LazyZarrDataset (lazy)

Key detail: scalers are fit on `train_idx` only, then applied to all data.

## Eager vs. lazy loading

**MOSADataset (h5mu)**: Loads all data to memory at setup.
- Use for datasets < 10 GB
- Simpler, no per-batch I/O

**LazyZarrDataset (zarr)**: Reads each sample from disk on demand.
- Use for datasets > 10 GB
- Better for multi-GPU (each worker reads independently)
- Can use cloud storage (S3)

## Batching

Each sample from the dataset is a dict:

```python
{
    "encoder_inputs": {"gexp": [5000], "meth": [485000]},  # 1D per view
    "decoder_targets": {"gexp": [5000], "meth": [485000]},
    "missing_masks": {"gexp": [5000], "meth": [485000]},  # bool: True=present
    "conditionals": [42],  # flattened [batch_onehot, tissue?, mutations?]
    "tissue_labels": [5],  # one-hot
    "source_ids": 1,  # which model_type (int)
    "sample_weights": 1.2,  # class rebalancing weight
    "sample_name": "ACH-000042",
}
```

`collate_fn()` stacks B samples into a `MOSABatch`:

```python
MOSABatch(
    encoder_inputs={"gexp": [B, 5000], "meth": [B, 485000]},
    decoder_targets={"gexp": [B, 5000], "meth": [B, 485000]},
    missing_masks={"gexp": [B, 5000], "meth": [B, 485000]},  # bool
    conditionals=[B, 42],
    tissue_labels=[B, 5],
    source_ids=[B],
    sample_weights=[B],
    sample_names=["s1", "s2", ..., "sB"],
)
```

**Key point**: `torch.stack()` adds batch dimension, so shapes go from 1D → 2D (except strings).

## How masks work

`missing_masks[view]` tells the encoder which features actually exist:

```python
sample_mask = batch.missing_masks[view].any(dim=1)  # [B] bool
if sample_mask.any():
    emb[sample_mask] = encoder(x[sample_mask], cond[sample_mask])
else:
    emb = zeros
```

Samples missing a view → zero embedding. Fusion and loss respect masks to only count present features.

## Class balancing

Three mechanisms:

1. **Stratified split** — train & val have same class proportions
2. **Sample weights** — rare classes get higher gradient contribution
3. **Macro loss** (per-view) — balance loss per group instead of per-sample

## Training vs. validation

**Train DataLoader**: `shuffle=True`, uses class weights
**Val DataLoader**: `shuffle=False`, deterministic

Both use same scalers fitted on training data only.

## Multi-GPU

When `devices > 1`, PyTorch Lightning handles distribution. With `.zarr` + `num_workers > 0`, each worker reads its own batches (recommended for large data).
