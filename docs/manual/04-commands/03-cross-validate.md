# cross-validate

Runs k-fold cross-validation and reports held-out reconstruction error.

```bash
mosa cross-validate --config <your-config>.yaml
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `--config` | path | yes | Path to YAML config file |
| `--folds` | int | no | Number of folds, per trial for optimize (overrides evaluation.n_folds) |
| `--strategy` | `stratified` \| `kfold` | no | Fold assignment: stratified balances model_type, kfold ignores it (overrides evaluation.strategy) |
| `--no-shuffle` | flag | no | Assign folds as contiguous blocks of sample order instead of shuffling |

The flags override `evaluation:` in the config for this run only. An unset flag
leaves the config value alone.

## Description

Each fold trains a fresh model from scratch and scores it on the held-out fold
through `reconstruct()`. The score is masked, variance-normalized mean squared
error (NMSE): squared errors over observed entries only, divided by that view's
variance across the whole dataset. Lower is better; 1.0 is the error of
predicting each view's mean.

Output from a small synthetic run, for shape rather than for the numbers:

```
fold   gexp (NMSE)         meth (NMSE)         aggregate
0      1.1279              1.0164              1.0721
1      1.0565              1.0832              1.0699
2      1.0875              1.0885              1.0880
pooled 1.0916              1.0613              1.0765

Aggregate NMSE (pooled over folds): 1.0765
Per-fold spread (diagnostic, not an error bar on the above): ±0.0081
```

The pooled row is the estimate to quote. Squared errors are summed across folds
per view before normalizing, so every observed entry is scored exactly once, by
a model that never trained on it. The per-fold rows are diagnostics: folds can
cover different numbers of views, so the spread is an indicator of variability,
not an uncertainty on the pooled value.

A fold that observes no entries for a view prints `n/a` for it and covers fewer
views in its aggregate, with a warning.

## Limitations

- No artifacts are written. Each fold trains into a temporary directory with
  checkpointing disabled, so `model.output_dir` is never touched.
- It refuses models without out-of-sample projection, naming the model, before
  training anything.
- `n_folds` larger than the smallest `model_type` class is rejected under
  `stratified`, with the offending class named. `strategy: kfold` ignores class
  balance and only requires `n_folds` samples in total.
- A view with no observed entries anywhere, or zero variance across them,
  cannot be scored and raises rather than reporting a meaningless number.

See also: [Configuration](../03-configuration.md) ·
[optimize](04-optimize.md)
