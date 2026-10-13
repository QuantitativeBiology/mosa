# train

[Documentation](../../index.md)

Fits the configured model and exports results to `model.output_dir`. Model files and exported splits depend on the selected implementation.

```bash
mosa train --config <your-config>.yaml
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `-c`, `--config` | path | yes | Path to the [YAML](https://yaml.org/) configuration file. |
| `-r`, `--resume` | `CKPT` | no | Resume training from an existing [Lightning](https://lightning.ai/docs/pytorch/stable/) checkpoint (`.ckpt`). |
| `-d`, `--debug` | flag | no | Add effective settings, preprocessing details, epoch metrics, timings and tracebacks. |

## Description

The command loads data, creates a holdout stratified by `model_type` when `evaluation.test_size > 0`, builds the registered model, and calls `fit()` followed by `save_outputs()`.

For the VAE, setting `evaluation.test_size: 0` disables validation, early stopping and validation-monitored checkpoints. The run still produces its final output parquets, but no monitored `checkpoints/last.ckpt`. [Lightning](https://lightning.ai/docs/pytorch/stable/) may write an unmonitored checkpoint in its log directory. With a validation split, `model.checkpoint_top_k: 0` disables the monitored checkpoints too.

[MOFA](https://biofam.github.io/MOFA2/) reports backend iterations. It writes `mofa_model.hdf5`; validation samples do not drive fitting or early stopping, and `--resume` is unsupported and warns before fitting from scratch. Non-Gaussian views may not produce reconstruction parquets.

## Outputs

See [Saved outputs](../saved-files.md) for [Parquet](https://parquet.apache.org/) conventions and model-specific directory layouts.

## Limitations

- Training does not resume automatically if interrupted. A new run can overwrite exported results while leaving older logs and checkpoints in place. Use a fresh output directory for an independent run.
- A nonzero stratified holdout requires at least two samples per source and enough training and validation samples for all sources. Invalid splits raise an error.

