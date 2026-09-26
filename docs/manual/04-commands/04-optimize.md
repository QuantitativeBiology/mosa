# optimize

Searches hyperparameters with Optuna, scoring each trial by cross-validation.

```bash
mosa optimize --config <your-config>.yaml \
              --search-space <search-space>.yaml \
              --trials 50
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `-c`, `--config` | path | yes | Path to YAML config file |
| `-p`, `--search-space` | path | yes | Path to search-space YAML (see configs/search_space.yaml) |
| `-n`, `--trials` | int | no | Number of Optuna trials (default: 20) |
| `-k`, `--folds` | int | no | Number of folds, per trial for optimize (overrides evaluation.n_folds) |
| `-s`, `--strategy` | `stratified` \| `kfold` | no | Fold assignment: stratified balances model_type, kfold ignores it (overrides evaluation.strategy) |
| `-N`, `--no-shuffle` | flag | no | Assign folds as contiguous blocks of sample order instead of shuffling |

## Description

Each trial samples values from the search space, applies them to the config's
`model:` block, and scores the result with `cross-validate`; see
[How trials are scored](../05-hyperparameter-search.md#how-trials-are-scored).
Output from a small synthetic run:

```
Trials: 6 (n_trials=6)
  completed: 6, pruned: 0

Best value (mean aggregate NMSE): 1.0869
Best params:
  learning_rate: 0.0006358358856676254
  joint_latent_dim: 8
  view_dropout_prob: 0.48495492608099716
```

## Limitations

- A search space naming something outside the model config is rejected before
  the study starts. What can and cannot be searched, and what happens to a
  failing trial, is in [Hyperparameter search](../05-hyperparameter-search.md).
- No model artifacts are kept. Like `cross-validate`, every trial trains into a
  temporary directory and `model.output_dir` is never touched.

See also: [Hyperparameter search](../05-hyperparameter-search.md) ·
[cross-validate](03-cross-validate.md)
