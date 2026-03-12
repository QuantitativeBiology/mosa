# Plotting Guide

This document explains how MOSA's diagnostic plotting works: where the data comes from, how each plot is generated, how to modify existing plots, and how to add new ones.

## How plotting works

Plotting is a **post-training** step. It reads files that were saved during or after training and generates figures from them. No plotting happens during training itself.

```bash
mosa plot --config configs/example.yaml [--output-dir outputs/custom]
```

This calls `generate_all_plots()` in `src/mosa/plot_utils.py`, which:

1. Loads all data files from the output directory
2. Generates UMAP plots (latent space + per-view reconstructions)
3. Generates loss curve plots (from Lightning's metrics log)
4. Generates reconstruction scatter plots (input vs reconstructed)
5. Computes clustering quality metrics (Calinski-Harabasz, Davies-Bouldin)

## Data sources

Plotting reads from two distinct sources, each produced differently.

### Source 1: CSV files saved after training

The `SaveLatentAndReconCallback` runs at the end of training and saves:

```
{output_dir}/
  data/                          # Training split
    latent.csv                   # Joint latent z [samples x latent_dim]
    recon_{view_name}.csv        # Reconstructed features [samples x features]
  inference/                     # Validation split
    latent.csv
    recon_{view_name}.csv
```

These are produced by running the trained model in eval mode over train and val dataloaders, collecting the outputs, and writing them to CSV. The callback lives in `src/mosa/callbacks.py`.

**Used by:** UMAP plots, reconstruction scatter plots, clustering metrics.

### Source 2: Lightning metrics log

PyTorch Lightning automatically writes a `metrics.csv` file during training:

```
{output_dir}/lightning_logs/version_*/metrics.csv
```

Each row is a training step. Columns are the metric names passed to `self.log()` in the model's `training_step()` and `validation_step()`. The plotting code loads the latest version, groups by epoch, and averages.

**Used by:** All loss curve plots (total, KL, adversarial, discriminator, per-omic MSE, per-model_type MSE).

### Source 3: Original input CSVs

The original omic CSV paths from the YAML config are read during plotting to compare input vs reconstruction. These are the same CSVs used for training.

**Used by:** Reconstruction scatter plots (input mean vs reconstructed mean).

## What gets plotted

| Plot file | Data source | What it shows |
|---|---|---|
| `umap_z.png` | `data/latent.csv` + samplesheet | Latent space structure, colored by tissue, shaped by model_type |
| `umap_recon_{view}.png` | `data/recon_{view}.csv` + samplesheet | Reconstructed omic space (training split) |
| `umap_recon_corrected_{view}.png` | `inference/recon_{view}.csv` + samplesheet | Reconstructed omic space (validation split) |
| `loss_total.png` | `metrics.csv` | Total loss + components (recon, KL, adversarial) overlaid |
| `loss_kl.png` | `metrics.csv` (`train/kl`) | KL divergence over epochs |
| `loss_adv.png` | `metrics.csv` (`train/adv_loss`) | Adversarial loss for VAE |
| `loss_disc.png` | `metrics.csv` (`train/disc_loss`) | Discriminator loss |
| `mse_{view}.png` | `metrics.csv` (`train/recon_{view}`, per-group keys) | Per-view MSE with total, val, and per-model_type breakdown |
| `input_recon_sample_{view}_*.png` | Input CSVs + `recon_*.csv` + samplesheet | Per-sample mean input vs mean reconstruction, colored by model_type |
| `input_recon_feature_{view}_*.png` | Input CSVs + `recon_*.csv` | Per-feature mean input vs mean reconstruction |
| `clustering_metrics.csv` | Latent + recon CSVs + samplesheet | Calinski-Harabasz and Davies-Bouldin scores |

## How to edit existing plots

All plotting functions are in `src/mosa/plot_utils.py`. The file is organized in sections:

- **Style**: `configure_plot_style()` — global matplotlib rcParams (font sizes, grid, DPI). Edit this to change the look of all plots at once.
- **UMAP**: `compute_umap_embedding()`, `plot_umap()`, `_make_umap_plot()` — UMAP computation and the layered scatter plot. The `_UMAP_LAYERS` list at the top of the file controls marker style, alpha, size, and draw order per model_type.
- **Loss curves**: `_plot_single_loss()`, `_plot_composite_loss()`, `_plot_omic_mse()` — each takes a metric DataFrame and produces a matplotlib figure. Colors come from matplotlib's `tab20` colormap.
- **Reconstruction scatters**: `_plot_sample_scatter()`, `_plot_feature_scatter()` — scatter plots with a y=x identity line.
- **Clustering**: `_compute_clustering_metrics()` — computes scores, no plotting involved.

### Common modifications

**Change figure size or DPI**: Edit the `plt.rcParams` dict in `configure_plot_style()`, or pass `figsize` directly in individual plot functions.

**Change UMAP appearance**: Edit `_UMAP_LAYERS` to change marker shapes, sizes, transparency, or draw order. Edit `DEFAULT_PALETTE` to change tissue colors.

**Change loss plot colors**: The `components` list in `_plot_composite_loss()` maps metric keys to labels and colors. The per-omic MSE uses `tab20` sequentially — change `cmap(color_idx)` calls to use fixed colors.

**Add a new loss curve**: If the metric is already logged during training (appears in `metrics.csv`), add it to the `individual_losses` list in `_generate_loss_plots()`:

```python
individual_losses = [
    ("train/kl",        "KL Divergence Loss",       "loss_kl.png",   cmap(4)),
    ("train/my_metric", "My Custom Metric",         "loss_custom.png", cmap(8)),  # new
]
```

**Use a custom color palette**: Pass it to the CLI-accessible function:

```python
from mosa.plot_utils import generate_all_plots
generate_all_plots(output_dir, config, palette={"Lung": "blue", "Breast": "pink"})
```

## Adding new plots that need new data

This is the key constraint: **you can only plot what was saved or logged during training**. If you want a plot that requires data not currently available, you need to change the training code first.

### If the data is a scalar metric per step/epoch

Log it in the model's `training_step()` or `validation_step()`:

```python
self.log("train/my_new_metric", value)
```

It will automatically appear in `metrics.csv`. Then add a plotting function in `plot_utils.py` that reads it via `_load_lightning_metrics()`.

### If the data is a tensor or matrix (e.g., per-sample embeddings, attention weights)

Save it in the `SaveLatentAndReconCallback` (or a new callback). The callback has access to the model and dataloaders at the end of training:

```python
# In callbacks.py, inside _save_split():
results = model.predict(loader)
# Save additional data:
pd.DataFrame(results["my_data"]).to_csv(out_dir / "my_data.csv")
```

Then add corresponding loading logic in `_load_data_files()` and a new plot function.

### If the data needs to be collected across all epochs

Lightning's `self.log()` only supports scalars. For richer per-epoch data (e.g., latent space snapshots, gradient norms), you need a custom callback that hooks into `on_train_epoch_end()`:

```python
class MyCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Collect and save whatever you need
        ...
```

Register the callback in `cli.py` alongside the existing ones.

### Summary of the data flow

```
Training code                     Saved artefacts              Plotting code
─────────────                     ───────────────              ─────────────
self.log("train/kl", ...)    →   metrics.csv              →   _load_lightning_metrics()
self.log("train/recon", ...)                                   _plot_single_loss()

SaveLatentAndReconCallback   →   data/latent.csv          →   _load_data_files()
  model.predict(loader)          data/recon_*.csv              _make_umap_plot()
                                 inference/latent.csv          _generate_reconstruction_plots()
                                 inference/recon_*.csv

Config (view paths)          →   original CSVs            →   _load_data_files()
                                                               _generate_reconstruction_plots()
```

The implication is that plotting and training are decoupled by the file system. Plotting never touches the model or training loop — it only reads what was written. If a plot needs data that doesn't exist on disk, the training side must be changed first to produce it.
