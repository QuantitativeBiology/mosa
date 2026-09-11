# Plots

The `mosa plot` command writes figures to `<output_dir>/plots/` and creates a summary CSV in `<output_dir>/metrics/`.

This page explains what the plots display. It does not dictate what a good result looks like, as that interpretation depends entirely on your data and research question.

## Embeddings

| File | Source | Shows |
|---|---|---|
| `umap_z.png` | `full/latent.parquet` | The tool reduces the shared latent space to two dimensions using UMAP. |
| `umap_recon_<view>.png` | `full/recon_<view>.parquet` | The tool runs the same UMAP reduction on the reconstructed values for a specific view. |
| `umap_recon_corrected_<view>.png` | `inference/recon_<view>.parquet` | The tool plots the counterfactual reconstruction. This file only appears when `model.inference` is enabled. |

These scatter plots colour points by their `tissue` label and mark them by their `model_type`, providing separate legends for each. The reduction pipeline runs a 50-component PCA before applying UMAP.

A UMAP layout is a topological projection, not a metric measurement. Distances between clusters and the absolute orientation of the plot are not meaningful. Furthermore, running the same data through UMAP twice with different seeds produces visually different results. You should treat these plots as qualitative visualisations rather than quantitative evidence.

## Losses

| File | Shows |
|---|---|
| `loss_total.png` | The plot displays the total loss per epoch alongside its overlaid components. |
| `loss_kl.png` | The plot shows the KL divergence term over time. |
| `loss_adv.png` | The plot charts the adversarial loss from the perspective of the VAE. It only generates when `adv_weight > 0`. |
| `loss_disc.png` | The plot charts the discriminator's own loss. It only generates when `adv_weight > 0`. |
| `mse_<view>.png` | The plot displays the per-view reconstruction MSE. It breaks the metric down by `model_type` if the run logged those metrics. |

All of these loss figures read from `lightning_logs/version_N/metrics.csv`. The tool automatically selects the highest-numbered version directory and groups the metrics by epoch.

Only `mse_<view>.png` overlays the validation curve alongside the training curve. The other loss figures plot the training series only.

## Reconstruction scatter

| File | Shows |
|---|---|
| `input_recon_sample_<view>_<tag>.png` | The plot graphs the per-sample mean input against the per-sample mean reconstruction. Each point represents one sample. |
| `input_recon_feature_<view>_<tag>.png` | The plot graphs the per-feature mean input against the per-feature mean reconstruction. Each point represents one feature. |

Both files include an identity line for reference. The `<tag>` in the filename distinguishes the ordinary reconstruction from the counterfactual reconstruction generated when `model.inference` is enabled.

## Clustering metrics

The command writes `clustering_metrics.csv` to `<output_dir>/metrics/`, rather than placing it alongside the figures. It calculates Calinski-Harabasz and Davies-Bouldin scores on three different spaces: the latent space, each view's raw input, and each view's reconstruction. It calculates these against both `tissue` and `model_type` labels. It standardises the values prior to scoring.

| Column | Meaning |
|---|---|
| `dataset` | Indicates which matrix the tool scored: the latent space, `input_<view>`, or `recon_<view>`. |
| `label_type` | Indicates which labels the tool evaluated against: `tissue` or `model_type`. |
| `calinski_harabasz` | The ratio of between-cluster to within-cluster dispersion. A higher score means the labels separate more clearly. |
| `davies_bouldin` | The average similarity between each cluster and its most similar one. A lower score means the labels separate more clearly. |

The tool outputs `NaN` for both scores if the labels contain fewer than two categories, or if they contain as many categories as there are samples.

These are descriptive statistics evaluating label separation. They are not absolute quality verdicts. Whether labels *should* separate depends on your experimental design. For example, strong `tissue` separation combined with weak `model_type` separation is the intended outcome of a batch correction run. The same numbers would represent a failure in a different context.

See also: [plot](04-commands/05-plot.md) · [Outputs](06-outputs.md)
