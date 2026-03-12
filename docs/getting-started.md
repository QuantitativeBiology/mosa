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

MOSA expects two types of input files:

### Samplesheet (required)

A CSV file with sample metadata. It **must** contain these columns:

| Column | Description |
|--------|-------------|
| `model_id` | Unique sample identifier (e.g., `ACH-000001`) |
| `model_type` | Sample category used for batch correction (e.g., `Cell Line`, `Tumor`, `Organoid`) |
| `tissue` | Tissue of origin (e.g., `Lung`, `Skin`) |

Example:

```
,model_id,model_type,tissue
0,ACH-000001,Cell Line,Lung
1,ACH-000002,Cell Line,Skin
2,TCGA-A1-A0SO,Tumor,Breast
```

### Omic data files (one per view)

Each omic modality is a CSV file in **features x samples** format (features as rows, samples as columns). MOSA transposes these automatically at load time.

Example (`transcriptomics.csv`):

```
,ACH-000001,ACH-000002,TCGA-A1-A0SO
GENE_A,12.3,8.1,15.2
GENE_B,0.5,1.2,0.8
GENE_C,7.8,NaN,6.1
```

- Missing values (`NaN`) are handled automatically: MOSA builds per-sample feature masks and imputes missing values with zero after z-score normalization.
- Samples that appear in the samplesheet but not in an omic file (or vice versa) are aligned automatically. Only samples present in **all** files are used.

### Mutations file (optional)

If you want to include mutation status as a conditional input, provide a CSV in the same features x samples format, where features are gene names and values are binary (0/1).

## Writing a config file

All experiment settings live in a single YAML file. Copy the template and edit it:

```bash
cp configs/example.yaml configs/my_experiment.yaml
```

At minimum, you need to set:

```yaml
views:
  transcriptomics:
    path: data/transcriptomics.csv    # your omic CSV
    hidden_layer_dims: [512, 256]     # encoder/decoder layer sizes

samplesheet_path: data/samplesheet.csv
output_dir: outputs/my_experiment
```

You can add as many views as you like. See [Configuration Reference](configuration.md) for all available options.

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
  data/
    latent.csv             # latent representations (training samples)
    recon_transcriptomics.csv  # reconstructions (training samples)
  inference/
    latent.csv             # latent representations (validation samples)
    recon_transcriptomics.csv  # reconstructions (validation samples)
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
