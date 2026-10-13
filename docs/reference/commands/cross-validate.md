# cross-validate

[Documentation](../../index.md)

Runs k-fold cross-validation and reports held-out reconstruction error.

```bash
mosa cross-validate --config <your-config>.yaml
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `-c`, `--config` | path | yes | Path to [YAML](https://yaml.org/) config file |
| `-k`, `--folds` | int | no | Number of folds, (overrides evaluation.n_folds) |
| `-s`, `--strategy` | `stratified` \| `kfold` | no | Fold assignment: stratified balances model_type, kfold ignores it (overrides evaluation.strategy) |
| `-N`, `--no-shuffle` | flag | no | Disable shuffling; stratified folds still balance model_type |
| `-d`, `--debug` | flag | no | Add effective settings, fold diagnostics, epoch metrics and tracebacks |

The flags override `evaluation:` in the config for this run only. An unset flag
leaves the config value alone.

## Description

Each fold trains a fresh model from scratch and scores it on the held-out fold
through `reconstruct()`. The score is masked, variance-normalized mean squared
error (NMSE): squared errors over observed entries only, divided by that view's
variance across the whole dataset. Lower is better; 1.0 corresponds to predicting a single
mean across all observed entries in each view.

Illustrative output from a small synthetic run:

```
fold   gexp (NMSE)         meth (NMSE)         aggregate
0      1.1279              1.0164              1.0721
1      1.0565              1.0832              1.0699
2      1.0875              1.0885              1.0880
pooled 1.0916              1.0613              1.0765

Aggregate NMSE (pooled over folds): 1.0765
Per-fold spread (diagnostic, not an error bar on the above): ±0.0081
```

Pooled scores sum squared errors and observed-entry counts across folds before variance normalization, then average across views. Each entry is reconstructed by a model that excluded its sample from fitting, although the VAE may use the fold for early stopping. Per-fold spread describes variability, not uncertainty on the pooled score.

A fold that observes no entries for a view prints `n/a` for it and covers fewer
views in its aggregate, with a warning.

## Limitations

- The CLI writes `history.csv` (when epoch history exists) and held-out
  `recon_<view>.parquet` files to `model.output_dir`. It does not save a final
  model. Each fold trains into its own temporary directory, which is removed.
- It refuses models without out-of-sample projection, naming the model, before
  training anything.
- `n_folds` larger than the smallest `model_type` class is rejected under
  `stratified`, with the offending class named. `strategy: kfold` ignores class
  balance and only requires `n_folds` samples in total.
- A view with no observed entries or zero observed variance cannot be normalized and is rejected.

