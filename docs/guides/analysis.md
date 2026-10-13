# Analysing results

This page explains how to generate diagnostic plots from a completed training run and use them to examine the learned representation, reconstructed profiles, and training behaviour. These plots help identify patterns and problems that an aggregate reconstruction score can hide. Read them alongside the held-out results described in [Evaluation and optimization](evaluation.md).

The examples use the MOSA VAE run from [Getting started](../getting-started.md). Keep its input dataset and output directory available: plotting reads both the original measurements and saved model outputs.

## Generate the plots

Run [`mosa plot`](../reference/commands/plot.md) with the configuration used for training:

```bash
mosa plot --config docs/examples/experiment.yaml
```

The command uses `data.path` to read measurements and sample metadata, and `model.output_dir` to find the saved representations, reconstructions, and logs. It does not fit or reload the model. The input metadata must include `tissue`, which is used to colour the embedding plots.

For the tutorial, figures are saved under `outputs/tutorial/run/plots/`, and label-separation scores are written to `outputs/tutorial/run/metrics/clustering_metrics.csv`. If you copied a run to another directory, pass `--output-dir <run-directory>` to [`mosa plot`](../reference/commands/plot.md) so it reads and writes results there; the input dataset path still comes from the configuration.

A saved latent representation is required. Additional plot families depend on the available reconstructions and training logs. Compatible exports from another model can provide embedding and reconstruction plots, while the loss plots described below use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) CSV logs.

## Examine the sample representations

The embedding plots use [UMAP](https://umap-learn.readthedocs.io/en/latest/) to display samples in two dimensions. Each point represents one sample, coloured by `tissue` and marked by its `model_type` source label. This lets you examine tissue-related structure and whether samples tend to group by source.

| Figure | Data shown |
|---|---|
| `umap_z.png` | Shared latent coordinates from `full/latent.parquet`. |
| `umap_recon_<view>.png` | Reconstructed profiles for one view from `full/recon_<view>.parquet`. |
| `umap_recon_corrected_<view>.png` | Profiles reconstructed under a selected source condition, from `inference/recon_<view>.parquet`, when that export exists. |

The command uses the `full/` exports where available and falls back to `train/`. Before [UMAP](https://umap-learn.readthedocs.io/en/latest/), PCA reduces each matrix to at most 50 components, bounded by its sample and feature counts. A separate embedding is fitted for each matrix, so the axes do not define a common coordinate system across figures.

Use the plots to investigate specific patterns, such as samples from one tissue separating by source or a subset of reconstructed samples forming an isolated group. Follow those observations back to sample metadata and reconstruction results. Source mixing is useful when technical variation is being reduced, but source-associated biological differences may also contribute to separation. The visual arrangement alone cannot distinguish those explanations.

The `corrected` filename identifies conditional reconstruction. It does not establish that the selected source condition removes technical variation without affecting biology; see [Conditional reconstruction](../models/mosa-vae.md#conditional-reconstruction) for the operation's meaning.

## Review training behaviour

For the VAE, loss curves show how the training objectives change over epochs. Start with the per-view MSE plots to compare training and validation reconstruction. A widening gap can indicate that improved fit to training samples is not carrying over to held-out samples; inspect which views and source groups contribute to it.

| Figure | Data shown |
|---|---|
| `mse_<view>.png` | Training and validation MSE for a view, with source-specific curves where those metrics were logged. |
| `loss_total.png` | Total training loss alongside its components. |
| `loss_kl.png` | Training KL divergence. |
| `loss_adv.png` | VAE adversarial loss when adversarial training is enabled. |
| `loss_disc.png` | Discriminator loss when adversarial training is enabled. |

The total loss combines objectives with different weights and roles. Use the [training-objective description](../models/mosa-vae.md#training-objectives) when interpreting changes in its components, particularly when comparing runs with different loss weights. These curves show optimization progress; the held-out reconstruction scores provide a common evaluation measure across such runs.

The command reads the highest-numbered `lightning_logs/version_N/metrics.csv` and groups logged values by epoch. Only the MSE figures overlay validation curves; the other loss figures show training values. Keep independent experiments in separate output directories so the latest log and saved profiles belong to the same run.

## Compare input and reconstructed profiles

Reconstruction scatter plots compare averages from the original and reconstructed measurements. They help reveal systematic shifts in profile magnitude, such as consistently higher reconstructed means for one source or a subset of features.

| Figure | Meaning of each point |
|---|---|
| `input_recon_sample_<view>_<tag>.png` | One sample, with its mean input value on the horizontal axis and mean reconstructed value on the vertical axis. |
| `input_recon_feature_<view>_<tag>.png` | One feature, with its mean input value across samples against its mean reconstructed value. |

The identity line marks equal input and reconstructed means. Points above it have higher reconstructed means; points below it have lower reconstructed means. The filename tag is `recon` for ordinary reconstruction or `corrected` for conditional source reconstruction.

Input averages omit values stored as `NaN`, while reconstruction averages include the predicted values at those positions. With missing data, the two means can therefore cover different sets of measurements. Agreement of averages can also hide errors that cancel within a sample or feature. Use observed-entry, held-out evaluation to assess individual reconstruction errors.

## Assess separation by tissue and source

`metrics/clustering_metrics.csv` quantifies how samples group according to the supplied `tissue` and `model_type` labels. These labels define the groups being assessed; the calculation does not discover or assign new clusters.

Scores are computed from the latent coordinates, inputs, or reconstructed profiles after feature-wise standardization. They are not calculated from the two-dimensional [UMAP](https://umap-learn.readthedocs.io/en/latest/) plots. Each row identifies a matrix and a label type:

| Column | Meaning |
|---|---|
| `dataset` | `z` for latent coordinates, `z_inf` for an available inference latent export, or `input_<view>`, `recon_<view>`, and `recon_inf_<view>` for the corresponding profiles. |
| `label_type` | `tissue` or `model_type`, indicating which metadata labels define the groups. |
| `calinski_harabasz` | Separation between groups relative to variation within groups. Larger values indicate more distinct groups. |
| `davies_bouldin` | Similarity between each group and its most similar neighbour, accounting for within-group spread. Smaller values indicate more distinct groups. |

The implementations are provided by [scikit-learn](https://scikit-learn.org/stable/). Interpret the scores according to the labels: strong tissue separation may reflect relevant biology, while strong source separation may reflect technical effects, biological differences, or both. Assess those possibilities using the study design and sample annotations before treating a change in score as an improvement.

Samples with missing labels or missing matrix values are excluded from that calculation. Different matrices may therefore be scored on different subsets, particularly when comparing incomplete inputs with fully reconstructed profiles. Check the samples contributing to each comparison. If no usable samples remain, the row is omitted; when fewer than two label categories remain, or every sample has its own category, the scores are `NaN`.

The [saved-file reference](../reference/saved-files.md) describes the underlying exports if you want to analyse them in your own scripts.
