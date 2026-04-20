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

The `mosa convert` command builds a MuData file from CSV tables. It validates all inputs before writing anything — if validation fails, you get a clear error message with no partial output.

See [CLI Reference — convert](cli.md#convert) for the full flag list.

### Samplesheet

The samplesheet has one row per sample. Required columns:

| Column | Required | Purpose |
|---|---|---|
| `model_id` | yes | Unique sample identifier. Must match column headers in omic CSVs exactly (case-sensitive). |
| `model_type` | yes | Sample class (e.g. `Cell Line`, `Tumor`). Used for conditional encoding, class balancing, and batch correction. Add the column even if all samples share the same value. |
| `tissue` | no | Tissue of origin. Used for tissue conditioning if present. Omitting it disables tissue conditioning. |

```
model_id,model_type,tissue
ACH-000001,Cell Line,Lung
ACH-000002,Cell Line,Skin
TCGA-A1-A0SO,Tumor,Breast
```

### Omic CSVs

Each omic CSV must be **features × samples**: features as rows, samples as columns. The first column is the feature index. MOSA transposes these automatically during conversion.

```
,ACH-000001,ACH-000002,TCGA-A1-A0SO
GENE_A,12.3,8.1,15.2
GENE_B,0.5,1.2,
GENE_C,7.8,,6.1
```

Missing values (empty cells or NaN) are allowed and become masked positions in `.layers["mask"]`.

**Orientation is the most common source of errors.** If your CSV is already samples × features (rows are samples, columns are features), the conversion will raise an error when it detects that row names match samplesheet sample IDs. The detection threshold is 50%: if more than half of the samplesheet IDs appear in the CSV row index, the CSV is considered transposed. Fix: re-export the file with features as rows.

If fewer than 10% of samplesheet IDs appear in the CSV column names, the converter emits a warning. This usually means sample ID formats differ between files (e.g. `ACH-000001` vs `ACH000001`). The conversion proceeds, but you may end up with 0 samples if the IDs don't overlap at all.

All values must be numeric. Empty cells and `NaN` are treated as missing data. String placeholders like `NA` or `N/A` are not accepted and will cause an error naming the offending column.

### Mutations CSV

Optional. Same format as omic CSVs (features × samples), but values should be binary (0/1). Each row (gene) becomes a `mutation_<gene>` column in `.obs`. Missing values are filled with 0.

### What the conversion does

The converter loads the samplesheet indexed by `model_id`, transposes each omic CSV to samples × features, and computes the union of samples across all views intersected with the samplesheet. Everything is aligned to that common set. Samples missing from a view get NaN rows via `reindex`; those NaN positions become `False` in `.layers["mask"]` and 0.0 in `.X`. Per-view presence indicators (boolean arrays) are stored in `.obsm[view_name]`. Mutations are added as `mutation_`-prefixed columns in `.obs`.

Samples don't need to appear in every view — partial coverage is handled via masks.

### h5mu vs zarr

| | h5mu | zarr |
|---|---|---|
| Loading | Entire file into memory at setup | Only metadata; samples read lazily per batch |
| Best for | Datasets that fit in RAM (< 10 GB) | Large datasets, multi-GPU, cloud storage |
| I/O pattern | Single read at start | Per-batch vectorized reads |
| Workers | No special handling | Each DataLoader worker opens its own handle |
| File structure | Single `.h5mu` file | Directory with chunked arrays |

### Verifying the output

Run `mosa inspect` on the output file to verify the conversion:

```bash
mosa inspect --input data/dataset.h5mu
```

Example output:

```
MuData: 850 samples x 2 modalities
  File: data/dataset.h5mu

Modalities:
  gexp_voom: 5000 features | 720/850 samples present, 84.7% values non-missing | min=-3.21, mean=0.412, max=14.6
  meth_combat: 485000 features | 850/850 samples present, 100.0% values non-missing | min=0.001, mean=0.489, max=0.999

Sample metadata (obs):
  model_type: Cell Line: 700, Tumor: 150
  tissue: Lung: 210, Breast: 180, Skin: 120 ... (23 unique values)

Sample IDs (first 5): ACH-000001, ACH-000002, ACH-000003, ACH-000005, ACH-000007  ... (850 total)

Per-view sample presence (obsm):
  gexp_voom: 720/850 samples
  meth_combat: 850/850 samples
```

What to check:

- **Sample count** matches what you expect from your samplesheet and view CSVs.
- **Modality names** match the view names you will use in your config file.
- **Samples present** per modality is plausible. 0 samples in a modality means no sample IDs overlapped between that CSV and the samplesheet — almost always an ID format mismatch.
- **Data range** is in the expected range for that data type (e.g. methylation beta values should be 0–1, voom-transformed expression is typically –5 to 15).
- **Sample IDs** look like real sample identifiers, not feature names. If you see gene names or CpG IDs here, the CSV was transposed.

### Troubleshooting

| Error / symptom | Likely cause | Fix |
|---|---|---|
| `missing required column 'model_id'` | Samplesheet has no `model_id` column, or it is named differently | Rename the column to `model_id` |
| `missing required column 'model_type'` | Samplesheet has no `model_type` column | Add the column; use a single value if all samples are the same type |
| `CSV appears to be samples x features` | Omic CSV is transposed (rows are samples) | Transpose the CSV: features as rows, samples as columns |
| `column 'X' contains non-numeric values` | CSV has string placeholders for missing data | Replace `NA`, `N/A`, `null`, etc. with empty cells or leave blank |
| `No samples found … that appear in any view CSV` | Sample ID format mismatch between samplesheet and CSVs | Ensure IDs are identical in both files, including capitalisation and separators |
| 0 samples present in a modality after conversion | ID overlap below detection threshold | Check ID format; use `--debug` to see per-view sample counts |
| Data range looks wrong (e.g. expression values are 0–1) | Wrong CSV passed for a modality | Check that each `--view name:path` pair points to the correct file |

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
