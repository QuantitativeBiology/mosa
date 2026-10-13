# transform

[Documentation](../../index.md)

Projects a dataset into the latent space of a saved model, without training.

```bash
mosa transform --checkpoint <output-dir>/checkpoints/last.ckpt \
               --input <new-samples>.h5mu \
               --output <projection-dir>/ \
               --reconstruct
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `-m`, `--checkpoint` | path | yes | Saved VAE model (.ckpt/.pt) or [MOFA](https://biofam.github.io/MOFA2/) model (.hdf5) |
| `-i`, `--input` | path | yes | Path to .h5mu or .zarr input data |
| `-o`, `--output` | path | yes | Directory to write latent.parquet (and reconstructions) |
| `-r`, `--reconstruct` | flag | no | Also write per-omic reconstruction parquets |
| `-d`, `--debug` | flag | no | Add diagnostic settings, timings and tracebacks |

## Description

The saved model supplies view names and mask configuration, so no [YAML](https://yaml.org/) file is needed. The registry identifies VAE `.ckpt`/`.pt` and [MOFA](https://biofam.github.io/MOFA2/) `.hdf5` files by their contents. Legacy [MOFA](https://biofam.github.io/MOFA2/) files without MOSA metadata use the default `mask` layer and support fitted-sample factors only; see [MOFA artifacts](../../models/mofa.md#read-the-saved-results).

Writes sample-indexed `latent.parquet` and, with `--reconstruct`, `recon_<view>.parquet` files. Predictions are computed before existing exports are replaced and stale reconstruction Parquets removed. Prediction failures leave existing files untouched; write failures can still leave incomplete exports.

## Limitations

- Projection reuses fitted preprocessing and categories without refitting.
- Unknown `model_type` categories are rejected.
- Input missing a view the checkpoint needs raises, naming both what the
  checkpoint expects and what the input holds. Extra views in the input are
  ignored.
- [MOFA](https://biofam.github.io/MOFA2/) returns learned factors for unchanged training samples and projects new
  samples using observed Gaussian features in the training coordinates. New
  samples must belong to groups seen during fitting. Reusing a fitted sample ID
  with changed values raises; use a new ID for new observations.
- A view whose feature names or their order differ from training raises,
  because the model reads features by position.
- For a [MOFA](https://biofam.github.io/MOFA2/) model, `--reconstruct` raises for views [MOFA](https://biofam.github.io/MOFA2/) fit with a
  `bernoulli` or `poisson` likelihood (see [Outputs](../saved-files.md)).

