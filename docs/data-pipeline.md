# Data Pipeline & Batching

How data flows from .h5mu/.zarr files to GPU-ready batches.

```
MuData file → MuDataDataModule.setup() → MOSADataset/LazyZarrDataset
  → DataLoader with collate_fn → MOSABatch → model.forward()
```

## MuData format

MOSA expects data in MuData format (`.h5mu` or `.zarr`). The structure:

- `.obs` — sample metadata (model_id, model_type, tissue, mutation_*)
- `.mod[view_name]` — each omic modality, where `.X` is the feature matrix and `.layers['mask']` is a boolean mask (True where data exists)
- `.obsm[view_name]` — boolean per-view presence indicator
- View names in config must match modality names in `.mod`

## Converting CSVs to MuData

The `mosa convert` command builds a MuData file from CSV tables.

### Input file formats

The samplesheet (required) has one row per sample with `model_id`, `model_type`, and `tissue` columns:

```
model_id,model_type,tissue
ACH-000001,Cell Line,Lung
ACH-000002,Cell Line,Skin
TCGA-A1-A0SO,Tumor,Breast
```

Omic CSVs (one per modality) have features as rows and samples as columns. The first column is the feature index. MOSA transposes these automatically:

```
,ACH-000001,ACH-000002,TCGA-A1-A0SO
GENE_A,12.3,8.1,15.2
GENE_B,0.5,1.2,
GENE_C,7.8,,6.1
```

Missing values (empty cells or NaN) are allowed — they become masked features.

The mutations CSV (optional) uses the same format but with binary values (0/1). Gene names become `mutation_<gene>` columns in `.obs`.

### Command

```bash
mosa convert \
  --samplesheet data/samplesheet.csv \
  --view gexp_voom:data/gexp_voom.csv \
  --view meth_combat:data/meth_combat.csv \
  --output data/dataset.h5mu \
  [--mutations data/mutations.csv] \
  [--format h5mu]
```

Use `--format zarr` for a zarr store instead of h5mu.

### What the conversion does

The conversion loads the samplesheet (indexed by `model_id`), transposes each omic CSV from features x samples to samples x features, computes the sample union across all views intersected with the samplesheet, and aligns everything to that common sample set. Samples missing from a view get NaN rows via `reindex`. NaN positions become `False` in the mask layer and NaN values in `.X` are replaced with 0.0. Per-view presence indicators are stored in `.obsm[view_name]`. Mutations (if provided) are added as `mutation_`-prefixed columns in `.obs`.

Samples don't need to appear in every view — missing views are handled via masks.

### h5mu vs zarr

| | h5mu | zarr |
|---|---|---|
| Loading | Entire file into memory at setup | Only metadata; samples read lazily per batch |
| Best for | Datasets that fit in RAM (< 10 GB) | Large datasets, multi-GPU, cloud storage |
| I/O pattern | Single read at start | Per-batch vectorized reads |
| Workers | No special handling | Each DataLoader worker opens its own handle |
| File structure | Single `.h5mu` file | Directory with chunked arrays |

### Inspecting the output

```python
import mudata
mdata = mudata.read("data/dataset.h5mu")

print(mdata)                          # overview: n_obs, modalities
print(mdata.obs.head())               # sample metadata
print(list(mdata.mod.keys()))         # modality names (must match config views)
print(mdata.mod["gexp_voom"].X.shape) # (n_samples, n_features)
print(mdata.mod["gexp_voom"].layers["mask"].sum())  # non-missing feature count
print(mdata.obsm["gexp_voom"].sum())  # samples present in this view
```

## Setup process

`MuDataDataModule.setup()` loads and validates the MuData structure, extracts modalities (converting sparse to dense), builds conditional vectors (one-hot model_type + optional tissue + optional mutations), does a stratified train/val split by model_type, fits StandardScaler on training data only (prevents leakage), computes inverse-frequency class weights, and creates `MOSADataset` or `LazyZarrDataset` objects.

Scalers are always fit on `train_idx` only, then applied to all data.

## Eager vs. lazy loading

`MOSADataset` (h5mu) loads all data to memory at setup. Simple, no per-batch I/O. Use for datasets < 10 GB.

`LazyZarrDataset` (zarr) reads batches from disk on demand via vectorized zarr slicing (`__getitems__`). Indices are sorted before each read for contiguous I/O. Each worker opens its own zarr handle, making it safe for multi-process and multi-GPU training. Can use cloud storage (S3).

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

`collate_fn()` stacks B samples into a `MOSABatch` — `torch.stack()` adds the batch dimension, so shapes go from 1D to 2D (except strings):

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

## How masks work

`missing_masks[view]` tells the encoder which features actually exist:

```python
sample_mask = batch.missing_masks[view].any(dim=1)  # [B] bool
emb = encoder(x, conditionals)   # encode full batch (required for DDP symmetry)
emb[~sample_mask] = 0.0          # zero out missing samples after encoding
```

The full batch is always encoded so that every DDP rank executes the same operations — otherwise NCCL deadlocks from asymmetric control flow. Samples missing a view get zero embeddings. Fusion and loss respect masks to only count present features.

## Class balancing

Three mechanisms work together: stratified splitting ensures train and val have the same class proportions, per-sample inverse-frequency weights give rare classes higher gradient contribution, and macro loss (per-view) balances loss per group instead of per-sample.

## Training vs. validation

The train DataLoader shuffles and uses class weights. The val DataLoader doesn't shuffle and is deterministic. Both use the same scalers fitted on training data only.

## Multi-GPU

When `devices > 1`, Lightning handles distribution. With `.zarr` + `num_workers > 0`, each worker reads its own batches independently.
