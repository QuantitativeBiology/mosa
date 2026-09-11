# Outputs

A finished `mosa train` run populates the directory specified in `model.output_dir` with the following structure:

```
<output_dir>/
├── last.ckpt                              most recent checkpoint
├── mosa-epoch=NNN-val/loss=N.NNNN.ckpt    best checkpoints, up to checkpoint_top_k
├── lightning_logs/version_N/
│   ├── metrics.csv                        per-epoch metrics
│   └── hparams.yaml
├── train/                                 training split
│   ├── latent.parquet
│   └── recon_<view>.parquet
├── val/                                   validation split, when test_size > 0
└── full/                                  every sample
```

If you run `mosa plot` later, the command adds a `plots/` directory and a `metrics/clustering_metrics.csv` file to this same location.

Setting `model.inference: true` creates an additional `inference/` split directory, which is explained below. Setting `evaluation.test_size: 0` disables the validation split entirely, meaning the run skips writing the `val/` directory and any best-epoch checkpoints. Read the [train](04-commands/01-train.md) page for details.

A MOFA run generates none of these files. The `MOFAModel.save_outputs()` method writes a single `mofa_model.hdf5` file containing both the model and its outputs. Because the training loop calls this method without arguments, the file lands in your working directory instead of under `output_dir`.

## Parquet outputs

The `latent.parquet` file contains one row per sample, indexed by the sample ID. It contains `joint_latent_dim` columns representing the shared latent representation.

Each `recon_<view>.parquet` file contains one row per sample and one column per feature for that specific view. The tool uses the view's original feature names for the column headers wherever they are available in the input data.

```python
import pandas as pd

z = pd.read_parquet("<output-dir>/full/latent.parquet")
z.shape        # (n_samples, joint_latent_dim)
z.index[:3]    # sample IDs
```

The tool writes reconstructions on their original scale. While the model operates in a preprocessed space internally, it inverts the scaling before writing the output. This ensures you can directly compare a reconstructed value against the corresponding value in your input file. If you use `preprocessing_mode: none`, the spaces coincide and no inversion occurs.

## Selecting a split

The `full/` directory covers every sample, including those held out for validation. The diagnostic plotting commands read from here. This split is the correct choice for downstream analysis across your entire cohort.

The `train/` and `val/` directories correspond to the exact split used during training. The `val/` rows represent the only data the model did not fit against. However, if you need a rigorous out-of-sample estimate, you should use the [cross-validate](04-commands/03-cross-validate.md) command rather than relying on this single split.

## Inference split

Enabling `model.inference: true` reconstructs the entire dataset a second time. During this second pass, the tool overrides every sample's `model_type` input, forcing it to match the value specified in `model.target_batch` (or the first available category if left unset). This produces a counterfactual representation: it shows what each sample would look like if it had originated from that specific source. The tool writes these results to the `inference/` directory.

## Checkpoints

The `last.ckpt` file captures the exact state at the end of training. The `mosa-epoch=...` checkpoints capture the best iterations based on validation loss, retaining up to `checkpoint_top_k` files.

You can pass any of these checkpoints to the [transform](04-commands/02-transform.md) command. Checkpoints bundle their own data configuration and their fitted scalers, meaning they do not require the original configuration file to run.

> Note: The checkpoint filename template includes the metric name `val/loss`. Because file systems interpret the `/` character as a directory separator, the best checkpoints land in per-epoch directories named `mosa-epoch=NNN-val/`, with each holding a single `loss=N.NNNN.ckpt` file. This is purely cosmetic; the files load normally.

## Metrics

The `lightning_logs/version_N/metrics.csv` file contains one row per epoch and one column per logged metric. These metrics include the total loss, per-view reconstruction loss, KL divergence, and the adversarial terms if you enabled them. Every training run creates a new incremented `version_N` directory within the same output path. The plotting commands automatically read the highest-numbered version directory they find.

See also: [Plots](07-plots.md) · [train](04-commands/01-train.md) · [transform](04-commands/02-transform.md)
