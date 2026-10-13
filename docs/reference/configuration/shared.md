# Library configuration

[Documentation](../../index.md)

An experiment uses one [YAML](https://yaml.org/) file with required `data` and `model` blocks and an optional `evaluation` block. These settings describe the input, select the model, and define how samples are held out.

```yaml
data:
  path: data.h5mu
  views: [gexp, meth]

model:
  type: mosa_vae
  output_dir: outputs/experiment-01
  random_seed: 42
  views:
    gexp:
      hidden_layer_dims: [512, 256]
    meth:
      hidden_layer_dims: [512, 256]

evaluation:
  test_size: 0.1
  n_folds: 5
  strategy: stratified
  shuffle: true
```

The `views` mapping inside `model` is specific to the MOSA VAE. All other keys shown here are part of the shared experiment structure. For model-specific fields, see [MOSA VAE configuration](mosa-vae.md) or [MOFA configuration](mofa.md).

## Paths and validation

Relative paths are resolved from the current working directory, not the configuration file's location. Use a separate `model.output_dir` for each experiment; rerunning into the same directory can replace saved results.

Unknown keys in the data, model, and evaluation blocks are rejected, with a suggested spelling when possible. `model.type` selects the registered configuration class, so fields from another model are errors too.

```bash
mosa validate --config config.yaml
```

This checks paths, views, masks, required metadata, and model-specific constraints without training. Some checks require values or fitted state and happen later. See [validate](../commands/validate.md).

## Data

Every model receives the selected views, masks, feature names, and sample metadata. Whether a model uses the optional conditioning flags depends on its implementation.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `path` | `str` | `""` | [MuData](https://mudata.scverse.org/stable/) `.h5mu` file or `.zarr` directory. Required for CLI runs. |
| `views` | `list[str]` | `[]` | Non-empty list of views to load. Additional views in the file are ignored. |
| `mask_layer_name` | `str` | `mask` | Boolean observed-value mask layer in each selected view. |
| `discrete_views` | `set[str]` | `[]` | Selected views marked as discrete; the VAE skips scaling them. Must be a subset of `views`. |
| `use_tissue` | `bool` | `true` | Enable tissue conditioning in models that use it, when the column exists. |
| `use_mutations` | `bool` | `true` | Enable mutation conditioning in models that use it, when `mutation_*` columns exist. |

The `model_type` metadata column is required by the data layer. It supplies source labels for stratified splits and is available to each model. The MOSA VAE also uses it for decoder conditioning and optional adversarial training; [MOFA](https://biofam.github.io/MOFA2/) uses it to define groups.

Preparing a file, including the distinction between an observed-value mask and a view-presence column, is covered in [Preparing data](../../guides/preparing-data.md).

## Evaluation

| Key | Type | Default | Meaning |
|---|---|---|---|
| `test_size` | `float` | `0.1` | Validation fraction for [`train`](../commands/train.md), in `[0, 1)`. Zero trains on all samples. |
| `n_folds` | `int` | `5` | Fold count for cross-validation and each optimization trial; at least 2. |
| `strategy` | `str` | `stratified` | `stratified` balances `model_type` across folds; `kfold` ignores source labels. |
| `shuffle` | `bool` | `true` | Shuffle before fold assignment. |

[`train`](../commands/train.md) always stratifies a nonzero holdout by `model_type`; `strategy` and `shuffle` control cross-validation only. The library supplies validation data to `fit()`, but the model decides whether to use it. [MOFA](https://biofam.github.io/MOFA2/), for example, does not use it for fitting or early stopping.

With `shuffle: false`, KFold uses contiguous blocks of the input sample order. Stratified folds still balance source labels and need not be contiguous. The split seed comes from `model.random_seed` when shuffling is enabled.

Evaluation settings remain fixed during [hyperparameter optimization](../../guides/evaluation.md). The `--folds`, `--strategy`, and `--no-shuffle` flags override the corresponding settings for that command invocation.

## Model selection and shared settings

| Key | Default | Meaning |
|---|---|---|
| `type` | required | Registered implementation name: `mosa_vae`, `mofa`, or a name registered by your code. |
| `output_dir` | `outputs` | Directory the model uses for run artifacts and exported results. |
| `random_seed` | `42` | Seed used by the library's splits and search sampler, and passed to the model. |

The remaining `model` keys belong to the selected implementation. The library does not translate VAE settings into external-model equivalents. A custom registration must be imported before the configuration is loaded; see [Adding a model](../../development/adding-models.md#make-the-registration-available).
