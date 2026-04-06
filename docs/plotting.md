# Plotting Guide

How MOSA's diagnostic plotting works, how to modify existing plots, and how to add new ones.

## How plotting works

Plotting is a post-training step. It reads files saved during or after training and generates figures — no plotting happens during training itself.

```bash
mosa plot --config configs/example.yaml [--output-dir outputs/custom]
```

This calls `generate_all_plots()` in `src/mosa/plot_utils.py`, which loads all data files from the output directory, generates UMAP plots (latent space + per-view reconstructions), loss curve plots (from Lightning's metrics log), reconstruction scatter plots (input vs reconstructed), and clustering quality metrics (Calinski-Harabasz, Davies-Bouldin).

## Data sources

Plotting reads from two distinct sources.

### CSV/parquet files saved after training

`SaveLatentAndReconCallback` runs at the end of training and saves:

```
{output_dir}/
  data/                          # Training split
    latent.csv
    recon_{view_name}.csv
  inference/                     # Validation split
    latent.csv
    recon_{view_name}.csv
```

These are produced by running the trained model in eval mode over train and val dataloaders and writing the outputs. The callback lives in `src/mosa/callbacks.py`. Used by UMAP plots, reconstruction scatter plots, and clustering metrics.

### Lightning metrics log

Lightning writes a `metrics.csv` during training at `{output_dir}/lightning_logs/version_*/metrics.csv`. Each row is a training step; columns are metric names from `self.log()`. The plotting code loads the latest version, groups by epoch, and averages. Used by all loss curve plots.

### Original input CSVs

The original omic CSV paths from the YAML config are read to compare input vs reconstruction. Used by reconstruction scatter plots.

## What gets plotted

| Plot file | Data source | What it shows |
|---|---|---|
| `umap_z.png` | `data/latent.csv` + samplesheet | Latent space, colored by tissue, shaped by model_type |
| `umap_recon_{view}.png` | `data/recon_{view}.csv` + samplesheet | Reconstructed omic space (training split) |
| `umap_recon_corrected_{view}.png` | `inference/recon_{view}.csv` + samplesheet | Reconstructed omic space (validation split) |
| `loss_total.png` | `metrics.csv` | Total loss + components overlaid |
| `loss_kl.png` | `metrics.csv` | KL divergence over epochs |
| `loss_adv.png` | `metrics.csv` | Adversarial loss for VAE |
| `loss_disc.png` | `metrics.csv` | Discriminator loss |
| `mse_{view}.png` | `metrics.csv` | Per-view MSE with per-model_type breakdown |
| `input_recon_sample_{view}_*.png` | Input CSVs + recon CSVs + samplesheet | Per-sample mean input vs mean reconstruction |
| `input_recon_feature_{view}_*.png` | Input CSVs + recon CSVs | Per-feature mean input vs mean reconstruction |
| `clustering_metrics.csv` | Latent + recon CSVs + samplesheet | Calinski-Harabasz and Davies-Bouldin scores |

## How to edit existing plots

All plotting functions are in `src/mosa/plot_utils.py`, organized by section:

`configure_plot_style()` sets global matplotlib rcParams (font sizes, grid, DPI). Edit this to change the look of all plots at once.

`compute_umap_embedding()`, `plot_umap()`, `_make_umap_plot()` handle UMAP computation and the layered scatter plot. `_UMAP_LAYERS` at the top of the file controls marker style, alpha, size, and draw order per model_type.

`_plot_single_loss()`, `_plot_composite_loss()`, `_plot_omic_mse()` each take a metric DataFrame and produce a figure. Colors come from matplotlib's `tab20` colormap.

`_plot_sample_scatter()`, `_plot_feature_scatter()` produce scatter plots with a y=x identity line.

`_compute_clustering_metrics()` computes scores without plotting.

### Common modifications

To change figure size or DPI, edit `plt.rcParams` in `configure_plot_style()`, or pass `figsize` directly in individual plot functions.

To change UMAP appearance, edit `_UMAP_LAYERS` for marker shapes/sizes/transparency/draw order, or `DEFAULT_PALETTE` for tissue colors.

To change loss plot colors, edit the `components` list in `_plot_composite_loss()`. Per-omic MSE uses `tab20` sequentially.

To add a new loss curve for a metric already in `metrics.csv`, add it to the `individual_losses` list in `_generate_loss_plots()`:

```python
individual_losses = [
    ("train/kl",        "KL Divergence Loss",       "loss_kl.png",   cmap(4)),
    ("train/my_metric", "My Custom Metric",         "loss_custom.png", cmap(8)),  # new
]
```

To use a custom color palette:

```python
from mosa.plot_utils import generate_all_plots
generate_all_plots(output_dir, config, palette={"Lung": "blue", "Breast": "pink"})
```

## Adding new plots that need new data

You can only plot what was saved or logged during training. If a plot needs data that doesn't exist on disk, the training side must produce it first.

For scalar metrics per step/epoch, log them in `training_step()` or `validation_step()`:

```python
self.log("train/my_new_metric", value)
```

It will appear in `metrics.csv` automatically. Then add a plotting function in `plot_utils.py` that reads it via `_load_lightning_metrics()`.

For tensors or matrices (per-sample embeddings, attention weights, etc.), save them in `SaveLatentAndReconCallback` or a new callback:

```python
# In callbacks.py, inside _save_split():
results = model.predict(loader)
pd.DataFrame(results["my_data"]).to_csv(out_dir / "my_data.csv")
```

Then add loading logic in `_load_data_files()` and a new plot function.

For data collected across all epochs, Lightning's `self.log()` only supports scalars. For richer per-epoch data (latent space snapshots, gradient norms), write a custom callback that hooks into `on_train_epoch_end()`:

```python
class MyCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Collect and save whatever you need
        ...
```

Register the callback in `cli.py` alongside the existing ones.

### Data flow summary

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

Plotting and training are decoupled by the file system. Plotting never touches the model or training loop — it only reads what was written.
