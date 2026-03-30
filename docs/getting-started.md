# Getting Started

This guide walks you through installing MOSA, preparing your data, and running your first training job.

## Requirements

- Python 3.11 or later
- pip (included with Python)
- Recommended: a dedicated virtual environment

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd mosa

# 2. Create and activate a virtual environment
python -m venv .mosa_venv
source .mosa_venv/bin/activate   # macOS / Linux
# .mosa_venv\Scripts\activate    # Windows

# 3. Install MOSA in editable mode (includes all dependencies)
pip install -e ".[dev]"
```

After installation, the `mosa` command becomes available in your terminal.

```bash
mosa --help
```

## Preparing your data

### Step 1: Gather CSVs

You need:
- Samplesheet with columns: `model_id`, `model_type`, `tissue`
- One omic CSV per modality (features × samples format)
- Mutations CSV (optional, binary features × samples)

Example samplesheet:
```
model_id,model_type,tissue
ACH-000001,Cell Line,Lung
ACH-000002,Cell Line,Skin
TCGA-A1-A0SO,Tumor,Breast
```

### Step 2: Convert to MuData

```bash
mosa convert \
  --samplesheet data/samplesheet.csv \
  --view gexp_voom:data/gexp.csv \
  --view meth_combat:data/meth.csv \
  --output data.h5mu \
  [--mutations data/mutations.csv]
```

For large datasets (> 10 GB), use `.zarr` for lazy loading:
```bash
mosa convert ... --output data.zarr --format zarr
```

This handles NaN imputation and creates masks automatically.

### Step 3: Write config

```yaml
data_path: data/data.h5mu  # or data.zarr

views:
  gexp_voom:
    hidden_layer_dims: [512, 256]
  meth_combat:
    hidden_layer_dims: [512, 256]

output_dir: outputs/my_experiment
```

See [Configuration Reference](configuration.md) for all options. No need to specify `samplesheet_path` or `mutations_path`—they're in the MuData file.

## Training

```bash
mosa train --config configs/my_experiment.yaml
```

Add `--debug` for detailed logging:

```bash
mosa train --config configs/my_experiment.yaml --debug
```

Training progress is printed to the terminal. When it finishes, MOSA saves the following files:

```
outputs/my_experiment/
  lightning_logs/          # training metrics (loss curves)
  train/
    latent.csv             # latent representations (training samples)
    recon_transcriptomics.csv  # reconstructions (training samples)
  val/
    latent.csv             # latent representations (validation samples)
    recon_transcriptomics.csv  # reconstructions (validation samples)
  full/
    latent.csv             # latent representations (all samples, original conditionals)
    recon_transcriptomics.csv  # reconstructions (all samples, original conditionals)
  inference/               # only if inference: true
    latent.csv             # corrected latent representations (all samples)
    recon_transcriptomics.csv  # corrected reconstructions (all samples)
```

To enable corrected inference (target batch forcing):

```yaml
inference: true
target_batch: Tumor    # optional; empty uses first available model_type
```

## Generating plots

After training, generate diagnostic plots:

```bash
mosa plot --config configs/my_experiment.yaml
```

If your outputs are in a different directory than what the config specifies:

```bash
mosa plot --config configs/my_experiment.yaml --output-dir outputs/other_run
```

This generates UMAP visualizations, loss curves, reconstruction quality plots, and clustering metrics under `outputs/my_experiment/plots/`.

## Quick example

```bash
# Activate environment
source .mosa_venv/bin/activate

# Train with the example config
mosa train --config configs/example.yaml --debug

# Generate plots
mosa plot --config configs/example.yaml
```

## Next steps

- [Configuration Reference](configuration.md) for all YAML options
- [Architecture Guide](architecture.md) to understand the model
- [Developer Guide](developing.md) to extend MOSA with new models
