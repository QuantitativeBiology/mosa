# Hyperparameter search

The `mosa optimize` command requires a second YAML file describing the search space. This file is not a standard configuration file. It contains no paths, no model type, and no evaluation settings. It only contains a mapping from `model` field names to their corresponding probability distributions.

```yaml
learning_rate:
  dist: loguniform
  low: 1.0e-5
  high: 1.0e-2

joint_latent_dim:
  dist: categorical
  choices: [32, 64, 128]

view_dropout_prob:
  dist: uniform
  low: 0.0
  high: 0.5

batch_size:
  dist: int
  low: 16
  high: 128
```

You can find a working example in `configs/search_space.yaml` within the repository.

## Distributions

| `dist` | Required keys | Samples |
|---|---|---|
| `loguniform` | `low`, `high` | Float on a log scale. Use this for values spanning multiple orders of magnitude, such as learning rates and loss weights. |
| `uniform` | `low`, `high` | Float on a linear scale. Use this for bounded values such as probabilities. |
| `int` | `low`, `high` | Integer within an inclusive range. |
| `categorical` | `choices` | One of an explicitly provided list. Use this for values that are not naturally ordered, or where only specific sizes are valid. |

The tool parses this file before starting the study. It will reject the file immediately if any entry is not a mapping, lacks a `dist` key, uses an unknown `dist` value, or is missing a required key for its distribution type.

## Searchable parameters

You can search over the top-level fields within the `model` block, provided those fields apply to the model type you selected. If you list a name that is not a valid field for the configured model, the tool raises an error before starting the study.

You cannot search over nested per-view parameters, such as `views.gexp.hidden_layer_dims`. Only top-level fields are reachable.

You also cannot search over the `evaluation` block. Attempting to search fields like `test_size`, `n_folds`, `strategy`, or `shuffle` will produce a targeted error message. The evaluation protocol is the benchmark used to score the study; searching over it would optimise the measurement system rather than the model itself.

## Scoring

Each trial applies its sampled values to the configuration's `model` block and runs the [cross-validate](04-commands/03-cross-validate.md) command on the resulting setup. The objective function minimises the pooled aggregate NMSE.

The TPE sampler takes its seed from `model.random_seed`. Passing the same search space, configuration file, and data will reproduce an identical study.

The computational cost scales directly with `--trials` multiplied by `evaluation.n_folds`. A standard approach to bound this cost is to reduce `--folds` during the search phase, and then confirm the winning hyperparameter combination at the full fold count in a standalone run.

## Failed trials

The tool prunes trials rather than failing fatally when a sampled combination violates a configuration invariant, such as attempting PoE fusion with mismatched hidden layer dimensions. It also prunes trials that fail mid-execution during cross-validation. The overall study continues, and the tool reports the completed and pruned counts at the end.

If zero trials complete successfully, the tool surfaces the error message from the first failure, preventing you from having to dig through the logs to find the root cause. You can use the `--debug` flag to print full tracebacks for every failing trial.

See also: [optimize](04-commands/04-optimize.md) · [cross-validate](04-commands/03-cross-validate.md)
