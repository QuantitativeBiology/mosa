# Getting Started

## Requirements

Python 3.11+, pip, and ideally a dedicated virtual environment.

## Installation

```bash
git clone <repo-url>
cd mosa

python -m venv .mosa_venv
source .mosa_venv/bin/activate   # macOS / Linux
# .mosa_venv\Scripts\activate    # Windows

pip install -e ".[dev]"
```

After installation, the `mosa` command is available:

```bash
mosa --help
```

## Preparing your data

You need a samplesheet CSV with columns `model_id` and `model_type` (required) and `tissue` (optional), one omic CSV per modality in features × samples format, and optionally a mutations CSV (binary, same format). See [Data Pipeline](data-pipeline.md#converting-csvs-to-mudata) for full format requirements and troubleshooting.

Example samplesheet:
```
model_id,model_type,tissue
ACH-000001,Cell Line,Lung
ACH-000002,Cell Line,Skin
TCGA-A1-A0SO,Tumor,Breast
```

Convert these to MuData:

```bash
mosa convert \
  --samplesheet data/samplesheet.csv \
  --view gexp_voom:data/gexp.csv \
  --view meth_combat:data/meth.csv \
  --output data.h5mu \
  [--mutations data/mutations.csv]
```

For large datasets (> 10 GB), use zarr for lazy loading:
```bash
mosa convert ... --output data.zarr --format zarr
```

NaN imputation and mask creation are handled automatically. Verify the output with:

```bash
mosa inspect --input data.h5mu
```

## Write a config

```yaml
data_path: data/data.h5mu  # or data.zarr

views:
  gexp_voom:
    hidden_layer_dims: [512, 256]
  meth_combat:
    hidden_layer_dims: [512, 256]

output_dir: outputs/my_experiment
```

See [Configuration Reference](configuration.md) for all options.

## Training

```bash
mosa train --config configs/my_experiment.yaml
```

Add `--debug` for detailed logging. When training finishes, MOSA saves latent representations and reconstructions:

```
outputs/my_experiment/
  lightning_logs/          # training metrics
  train/
    latent.parquet
    recon_<view>.parquet
  val/
    latent.parquet
    recon_<view>.parquet
  full/
    latent.parquet
    recon_<view>.parquet
  inference/               # only if inference: true
    latent.parquet
    recon_<view>.parquet
```

To enable corrected inference (target batch forcing):

```yaml
inference: true
target_batch: Tumor    # optional; empty uses first available model_type
```

## Generating plots

```bash
mosa plot --config configs/my_experiment.yaml
```

Override the output directory if needed:

```bash
mosa plot --config configs/my_experiment.yaml --output-dir outputs/other_run
```

This generates UMAPs, loss curves, reconstruction quality plots, and clustering metrics under `outputs/my_experiment/plots/`.

## Quick example

```bash
source .mosa_venv/bin/activate
mosa train --config configs/example.yaml --debug
mosa plot --config configs/example.yaml
```

## Next steps

See the [Configuration Reference](configuration.md) for all YAML options, the [Architecture Guide](architecture.md) to understand the model, or the [Developer Guide](developing.md) to extend MOSA.
