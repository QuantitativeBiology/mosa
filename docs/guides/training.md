# Training and projecting samples

This page covers configuring a training run, saving and reusing the fitted model, and projecting new samples. The examples use the dataset and configuration from [Getting started](../getting-started.md). If you have already completed that run, you can continue from **Choose a saved model** below. Run commands from the repository root with the MOSA environment active.

For your own experiment, first prepare a compatible dataset using [Preparing a dataset](preparing-data.md), then adapt the configuration to its paths and view names. The training procedure below uses the MOSA VAE; external integrations use the same commands but may differ in their use of validation and saved outputs.

## Configure the run

The [example configuration](../examples/experiment.yaml) separates the experiment into `data`, `model`, and `evaluation` settings. Before fitting your own data, check the following:

- `data.path` identifies the prepared [MuData](https://mudata.scverse.org/stable/) file, and `data.views` selects the omic views to load.
- `model.type` selects the integration model. For `mosa_vae`, `model.views` defines an encoder for each selected view using the same names. Network dimensions, training objectives, and hardware settings belong in this block.
- `model.output_dir` determines where the run's results are written. Choose a separate directory for each independent experiment, since rerunning training can replace exports while leaving existing logs and checkpoints in place.
- `evaluation.test_size` determines the fraction of samples reserved for validation. The example uses `0.2`, giving 48 training and 12 validation samples.

A nonzero validation holdout is stratified by `model_type`, preserving source proportions where the sample counts allow it. Each source needs enough samples to occur in both sets. The VAE fits preprocessing on training samples and reuses those statistics for validation, so held-out values do not determine scaling or imputation.

Keep a copy of the configuration with the results so you can identify the data, model settings, and validation split used for the run. The [configuration reference](../reference/configuration/shared.md) lists all fields and their defaults.

## Validate and train

Use [`mosa validate`](../reference/commands/validate.md) to check that the configuration matches the dataset before starting a fit:

```bash
mosa validate --config docs/examples/experiment.yaml
```

Once validation succeeds, start the run with [`mosa train`](../reference/commands/train.md):

```bash
mosa train --config docs/examples/experiment.yaml
```

The command loads the selected views, creates the training and validation subsets, fits the model, and writes its outputs. For the VAE, validation loss is monitored after each epoch. Early stopping can end training before `model.num_epochs` if that loss stops improving for the configured patience.

With validation enabled and `model.checkpoint_top_k` greater than zero, the run saves checkpoints under `outputs/tutorial/run/checkpoints/`. Latent coordinates and reconstructed profiles are exported under `train/`, `val/`, and `full/`; `full/` contains both subsets. Use validation or cross-validation results when assessing reconstruction accuracy, since the training samples have already contributed to fitting.

Setting `evaluation.test_size: 0` fits all samples. This disables validation-based early stopping and the monitored checkpoints described here. If you need a reusable VAE from a fit without validation, explicitly save it through the [Python interface](python.md#load-data-and-fit-a-model).

## Choose a saved model

A VAE checkpoint contains the model parameters, fitted preprocessing, configuration, and training state. It lets you reuse the model without fitting it again, or continue an interrupted training run.

`checkpoints/last.ckpt` records the latest saved epoch. Files named `mosa-epoch=…-val_loss=….ckpt` retain epochs selected by validation loss, up to `model.checkpoint_top_k`. Choose the file that corresponds to the state you want to use: the latest epoch and the epoch with the lowest validation loss can differ. Training exports use the weights at the end of fitting.

The following examples use `outputs/tutorial/run/checkpoints/last.ckpt`. For another run, replace this path with the selected checkpoint. The [saved-file reference](../reference/saved-files.md) explains the complete output layout.

## Resume VAE training

To continue from a checkpoint, pass it to [`mosa train`](../reference/commands/train.md) with `--resume`:

```bash
mosa train --config docs/examples/experiment.yaml \
  --resume outputs/tutorial/run/checkpoints/last.ckpt
```

Resumption restores the training state, including optimizer progress, through [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). Keep the data, architecture, and preprocessing settings consistent with the saved run. If training already reached the configured epoch limit, increase `model.num_epochs` before resuming; this setting is the total epoch limit, rather than a number of additional epochs.

Resume support depends on the integration model. The [MOFA adapter](../models/mofa.md) does not resume fitting and starts a fresh fit with a warning when `--resume` is supplied.

## Prepare new samples

Projection applies the saved model to another dataset. The new measurements must have the same input structure: every trained view must be present, with exactly the same feature names in the same order. A feature with no measurement should remain in the dataset with a false mask entry. Removing its column changes the network input.

Use source labels in `model_type` that were represented during fitting; both current implementations reject unknown sources. For the VAE, also provide the tissue and mutation annotations used by the fitted model, since they determine decoder conditioning.

The tutorial data generator already created 12 new samples with matching features and known source labels. Convert their tables with [`mosa convert`](../reference/commands/convert.md):

```bash
mosa convert --conditionals outputs/tutorial/new-input/metadata.csv \
  --view gexp:outputs/tutorial/new-input/gexp.csv \
  --view meth:outputs/tutorial/new-input/meth.csv \
  --output outputs/tutorial/new-data.h5mu
```

For your own samples, apply the same assay-specific preparation and feature identifiers used for training. Model preprocessing, such as fitted scaling, is restored from the checkpoint and reused during projection.

## Project and reconstruct

After training, use [`mosa transform`](../reference/commands/transform.md) to apply the fitted model to another dataset. It places each sample in the latent space learned during training, allowing new samples to be analysed using the same representation. With `--reconstruct`, it also estimates feature values for each omic view, including measurements missing from the input. The model reuses its saved parameters and preprocessing; this command does not train or update it.

The following command applies the tutorial model to the 12 new samples converted in the previous section. `--checkpoint` selects the saved model, `--input` supplies their prepared dataset, and `--output` chooses where to save the predictions:

```bash
mosa transform --checkpoint outputs/tutorial/run/checkpoints/last.ckpt \
  --input outputs/tutorial/new-data.h5mu \
  --output outputs/tutorial/projection --reconstruct
```

No experiment configuration is needed for this command: the saved model supplies the input settings and fitted state. The command writes `latent.parquet` directly into the selected output directory. In this example, it contains 12 rows and four latent coordinates, indexed by the new sample IDs.

`--reconstruct` also writes `recon_gexp.parquet` and `recon_meth.parquet`, with the trained feature names and predictions on the input scale. Omit this flag when you only need latent coordinates. Use a separate output directory for each projection, since existing projection files can be replaced.

For the VAE, these reconstructions use each sample's supplied conditions. To estimate profiles under a different source condition, enable `model.inference` and select a known source with `model.target_batch` in the training configuration. This produces an additional `inference/` export; the `--reconstruct` flag alone does not change source conditions. The [model guide](../models/mosa-vae.md#conditional-reconstruction) explains how to interpret these estimates.

Continue with [Analysing results](analysis.md) to inspect the saved profiles and representations, or [Evaluation and optimization](evaluation.md) to compare configurations using held-out reconstruction.
