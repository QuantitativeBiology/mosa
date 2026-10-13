# Saved outputs

[Documentation](../index.md)

[`mosa train`](commands/train.md) asks the selected model to write its results to `model.output_dir`. The library uses a common [Parquet](https://parquet.apache.org/) convention for latent representations and reconstructions, while model files, logs, and exported splits depend on the implementation.

| Artifact | MOSA VAE | [MOFA](https://biofam.github.io/MOFA2/) integration |
|---|---|---|
| Latent representations | `train/`, `val/` when present, and `full/` | `full/`, containing fitted samples only |
| Reconstructions | Each configured view, on the input scale | Gaussian views only, on the input scale |
| Model file | Monitored `.ckpt` files when validation and checkpointing are enabled; explicit `.pt` saves through [Python](https://www.python.org/) | `mofa_model.hdf5` |
| Training log | [Lightning](https://lightning.ai/docs/pytorch/stable/) CSV metrics | Backend iteration output; no [Lightning](https://lightning.ai/docs/pytorch/stable/) loss CSV |
| Conditional reconstruction | Optional `inference/` split | Not implemented |

A custom model implements its own exports; the interface does not create these files automatically.

## VAE checkpoints and logs

A run with a validation split and `checkpoint_top_k > 0` writes:

```text
<output_dir>/
├── checkpoints/
│   ├── last.ckpt
│   └── mosa-epoch=NNN-val_loss=N.NNNN.ckpt
├── lightning_logs/version_N/
│   ├── metrics.csv
│   └── hparams.yaml
├── train/
├── val/
└── full/
```

Each exported split holds `latent.parquet` and `recon_<view>.parquet`. `model.inference: true` adds `inference/`. `last.ckpt` records the end of training; the other monitored checkpoints retain the best validation epochs up to `checkpoint_top_k`.

With `evaluation.test_size: 0`, there is no validation, early stopping, `val/`, or monitored `checkpoints/last.ckpt`. [Lightning](https://lightning.ai/docs/pytorch/stable/) may save an unmonitored checkpoint under its log directory. `checkpoint_top_k: 0` also disables monitored checkpoints. Through [Python](https://www.python.org/), `model.save(path)` can explicitly save a fitted model to `.ckpt` or `.pt`.

Checkpoints include data and model configuration, fitted preprocessing, and category information. The [Parquet](https://parquet.apache.org/) conventions are described below.


## MOFA model file

`mofa_model.hdf5` stores the model and adapter preprocessing metadata. Its neighbouring `full/` directory contains fitted-sample exports. Projection and reconstruction on the input scale require the saved preprocessing record.

## Parquet layout

A split directory uses these filenames:

```text
full/
├── latent.parquet
├── recon_gexp.parquet
└── recon_meth.parquet
```

`latent.parquet` is indexed by sample ID, with one column per latent dimension or retained factor.

Each `recon_<view>.parquet` has one row per sample and one column per feature, using the input feature names and order. Current implementations return supported reconstructions on the original input scale, after reversing their fitted preprocessing.

```python
import pandas as pd

z = pd.read_parquet("outputs/experiment-01/full/latent.parquet")
x_hat = pd.read_parquet("outputs/experiment-01/full/recon_gexp.parquet")
print(z.shape, x_hat.shape)
```

Reconstructions include estimates for missing entries. Use the input masks to compare predictions with observed measurements.

## Splits

For the MOSA VAE, `train/` and `val/` correspond to the fitting split, while `full/` includes both. Use `full/` for cohort-wide coordinates; distinguish fitted and held-out samples when assessing reconstruction.

For [MOFA](https://biofam.github.io/MOFA2/), `full/` contains only the samples passed to fitting. With a nonzero validation holdout, it excludes held-out samples because this backend does not export a separate validation split. Set `evaluation.test_size: 0` to fit and export the entire cohort, or use [`transform`](commands/transform.md) to project additional samples.

The VAE's optional `inference/` split changes the decoder's batch condition for conditional reconstruction. It is a separate export, not a held-out evaluation split.

## Cross-validation and optimization

The [`cross-validate`](commands/cross-validate.md) CLI prints pooled and per-fold scores, writes held-out `recon_<view>.parquet` files into `model.output_dir`, and writes `history.csv` when the model provides epoch history. Every row is reconstructed by the fold model that held that sample out. Fold models and their temporary directories are removed after scoring.

[`optimize`](commands/optimize.md) prints the best parameters and score. It keeps neither fold models nor a persisted study, and does not train the final model. See [Evaluation and optimization](../guides/evaluation.md).

## Projection and plots

[`transform`](commands/transform.md) writes `latent.parquet` directly into its `--output` directory, with optional `recon_<view>.parquet` files alongside it. It does not create a `full/` subdirectory. Once predictions succeed, it replaces projection files and removes stale reconstruction Parquets in that directory.

[`plot`](commands/plot.md) adds `plots/` and `metrics/clustering_metrics.csv` to a training output directory. See [Analysis and plotting](../guides/analysis.md) for their contents.

Use a separate directory for each run. A repeated training run can overwrite [Parquet](https://parquet.apache.org/) files while leaving earlier checkpoints and log versions in place.
