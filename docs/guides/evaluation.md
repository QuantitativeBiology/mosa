# Evaluating and optimizing a model

This page explains how to measure reconstruction on samples excluded from fitting and use that evaluation to choose model parameters. Cross-validation evaluates a configuration across several training and validation splits. Hyperparameter optimization repeats that procedure for different parameter values to find a configuration with lower reconstruction error.

The examples use the dataset and configuration from [Getting started](../getting-started.md). Run them from the repository root with the MOSA environment active. You need the prepared dataset, but do not need a previously fitted model: evaluation trains fresh models for each split.

## Set the evaluation protocol

The `evaluation` block in the [tutorial configuration](../examples/experiment.yaml) specifies three folds, source-stratified assignment, and shuffling:

```yaml
evaluation:
  test_size: 0.2
  n_folds: 3
  strategy: stratified
  shuffle: true
```

With three folds, the 60 samples are divided into three groups of 20. Each group is held out once while a fresh model fits the other 40 samples, so every sample receives a reconstruction from a model that excluded it from fitting. `strategy: stratified` keeps the proportions of `model_type` source labels similar across folds. Each source must contain at least `n_folds` samples.

`test_size` controls the single holdout used by [`mosa train`](../reference/commands/train.md); it does not determine cross-validation folds. For cross-validation, `n_folds`, `strategy`, and `shuffle` control assignment, with `model.random_seed` providing the seed when shuffling is enabled.

Keep the dataset, split settings, and training budget consistent when comparing configurations. The tutorial's two-epoch fits demonstrate the procedure; choose a sufficient training budget before using the scores to select a model for a study.

## Evaluate held-out reconstruction

Use [`mosa cross-validate`](../reference/commands/cross-validate.md) to evaluate the configured model:

```bash
mosa cross-validate --config docs/examples/experiment.yaml
```

For each fold, MOSA fits the model and its preprocessing using the training samples, then reconstructs the held-out samples. It compares predictions with their observed measurements using the input masks. Missing measurements are excluded from scoring because their true values are unavailable.

### Scoring

MOSA reports normalized mean squared error (NMSE) for each view. Squared reconstruction errors are summed across folds and divided by the number of observed entries, giving a pooled mean squared error. This is then divided by the variance of that view's observed measurements across the dataset. The normalization makes the score relative to the view's variation rather than its measurement units.

A lower NMSE indicates smaller reconstruction errors relative to the variation in the observed measurements. The score reaches zero when every scored measurement is reconstructed exactly. For comparison, predicting a single mean calculated over all observed entries in a view gives a pooled NMSE of one, so scores below one improve on that constant prediction.

MOSA averages the pooled NMSE values across views to obtain the aggregate score used for configuration comparison and optimization. Each view contributes equally, so an omic with more features does not receive greater weight. Read the aggregate alongside the individual view scores: a lower aggregate may reflect an improvement in one omic accompanied by poorer reconstruction in another.

The report also includes scores for each fold and their spread, which help identify sensitivity to the sample split. That spread describes variation among the fold scores; it does not estimate uncertainty in the pooled aggregate.

A view with no observed entries or zero observed variance cannot be normalized. If only a particular fold has no observations for a view, that fold reports `n/a` for the view; the pooled calculation still uses observations from other folds.

### Saved results

The command prints the scores in the terminal and saves `recon_<view>.parquet` files in `model.output_dir`. Each row is predicted by the model for the fold that held that sample out, with the original sample and feature order preserved. When the model provides epoch histories, the command also saves `history.csv` with training and validation losses for each fold.

Fold models are temporary and are removed after evaluation. These reconstruction files let you examine held-out predictions, but they do not provide a saved model for later projection. Use a separate output directory when you want to keep results from several evaluation runs.

## Define a parameter search

After checking that cross-validation runs successfully, choose which model settings to optimize. A *search space* specifies the values allowed for each parameter. One *trial* samples a combination of those values and evaluates it using the same cross-validation procedure.

The supplied [search-space file](../examples/search-space.yaml) searches learning rate and latent width:

```yaml
learning_rate:
  dist: loguniform
  low: 0.0001
  high: 0.001
joint_latent_dim:
  dist: categorical
  choices: [2, 4]
```

For this search, each trial chooses a learning rate between `0.0001` and `0.001` on a logarithmic scale, and either two or four latent dimensions. The experiment configuration still supplies all other settings, including the dataset, model type, and evaluation protocol.

Search-space keys must be direct fields of the selected model's `model` block. Nested settings such as `views.gexp.hidden_layer_dims` and evaluation settings such as `n_folds` cannot be searched. Use parameter names belonging to the selected model; the [MOSA VAE configuration](../reference/configuration/mosa-vae.md) lists its fields.

The available distributions are:

| `dist` | Required keys | How values are selected |
|---|---|---|
| `loguniform` | `low`, `high` | Floating-point values on a logarithmic scale, suitable for learning rates or weights spanning orders of magnitude. |
| `uniform` | `low`, `high` | Floating-point values on a linear scale, for example probabilities within a chosen interval. |
| `int` | `low`, `high` | Integers between the bounds, including both endpoints. |
| `categorical` | `choices` | One value from an explicit list. |

MOSA checks the parameter names and distribution definitions before starting the study. A [larger VAE search-space example](../../configs/search_space.yaml) is also available in the repository.

## Run the optimization

Use [`mosa optimize`](../reference/commands/optimize.md) with the experiment configuration, search-space file, and trial count:

```bash
mosa optimize --config docs/examples/experiment.yaml \
  --search-space docs/examples/search-space.yaml --trials 2
```

MOSA uses [Optuna](https://optuna.org/) to sample parameter combinations and minimizes the pooled aggregate NMSE described above. Two trials with three folds require six model fits. Increasing either count increases the training cost; set the trial count according to the search size and available compute.

The sampler uses `model.random_seed`. Keeping that seed and the evaluation settings fixed helps make comparisons reproducible, although numerical results also depend on the model and training hardware.

At completion, the terminal report lists the best parameter values, their aggregate NMSE, and counts of completed and pruned trials. A trial that cannot complete because of an invalid configuration or evaluation failure is marked as pruned and supplies no score for selection. Review the failure messages before interpreting the search; rerun [`mosa optimize`](../reference/commands/optimize.md) with `--debug` for tracebacks. If no trial completes, there is no valid configuration to select.

## Train the selected configuration

Optimization reports parameter values but does not save a final fitted model or persist the [Optuna](https://optuna.org/) study. To use the selected configuration for projection, copy the experiment file:

```bash
cp docs/examples/experiment.yaml outputs/tutorial/selected.yaml
```

Edit `outputs/tutorial/selected.yaml`: copy the reported best values into its `model` block and set `model.output_dir` to a new directory, such as `outputs/tutorial/selected-run`. Then validate and fit it with [`mosa validate`](../reference/commands/validate.md) and [`mosa train`](../reference/commands/train.md):

```bash
mosa validate --config outputs/tutorial/selected.yaml
mosa train --config outputs/tutorial/selected.yaml
```

This produces the model files and exports needed for subsequent analysis and projection. The `evaluation.test_size` setting applies to this training run; with the tutorial value of `0.2`, it reserves a validation subset rather than fitting all 60 samples. Follow [Training and projection](training.md) to choose a checkpoint and apply it to another dataset.

## Evaluation limits

Choose splits that match the intended use of the model. Source-stratified and ordinary KFold assignment do not keep related patients, institutions, or collection times separate. If those boundaries matter, define an appropriate evaluation procedure in [Python](https://www.python.org/).

The VAE uses each held-out fold for validation monitoring and early stopping. Optimization also uses the resulting scores to select parameters. These scores therefore support model selection, but an independent test cohort is needed for an assessment unaffected by those choices.

NMSE measures reconstruction of observed entries. It does not establish accuracy for measurements that were never observed, biological validity of the representation, or the correctness of changing a decoder's source condition. Assess those properties with evidence suited to the study, alongside reconstruction error.
