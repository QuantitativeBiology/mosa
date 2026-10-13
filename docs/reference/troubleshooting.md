# Troubleshooting

[Documentation](../index.md)

Expected configuration, path, and data errors are printed to stderr with `Error:` and exit status 1. Add `--debug` after the command name for a traceback; report unexpected tracebacks without that flag as bugs.

## Help and progress

Use `mosa -h` to choose a command and `mosa <command> -h` for its purpose,
outputs, conditions, example and options. Every option has a short alias;
`-d` and `--debug` are equivalent and follow the command name.

Normal runs announce operations before expensive work, with counts and output
locations. Debug adds settings, preprocessing choices, timings and epoch metrics.
Progress and warnings use stderr; reports and score tables use stdout. For
example, [`mosa cross-validate -c config.yaml > scores.txt 2> progress.log`](commands/cross-validate.md)
separates the score table from diagnostics (training bars also use stdout).

The VAE's tqdm epoch/batch bar begins when training starts, after loading and
preprocessing. It remains enabled without validation and without `-d`. If it
is absent, check whether stdout was redirected or captured by your runner, and
whether `model.type` is `mofa`, which has iteration output rather than a
[Lightning](https://lightning.ai/docs/pytorch/stable/) epoch bar. With `-d`, completed VAE epochs also have text summaries
on stderr, including when `evaluation.test_size: 0`.

## Configuration

### `Unknown key 'laten_dim' in model:. Did you mean 'joint_latent_dim'?`

Correct the key spelling. Unknown keys are rejected with a suggested match or the valid-key list.

### `Config <path> must contain top-level 'data:' and 'model:' blocks`

The file parsed successfully as [YAML](https://yaml.org/), but it is not a valid MOSA configuration file. This usually happens when an indentation slip nests the `data` block under another key.

### `model.type is required (e.g. 'mosa_vae', 'mofa')`

Set `model.type` to the registered method name so the parser can select its configuration class.

## Data

### `data.path not found: <path>`

The specified [MuData](https://mudata.scverse.org/stable/) file or [zarr](https://zarr.dev/) directory does not exist. MOSA resolves paths relative to your working directory, not relative to the configuration file's location.

### `View '<name>' not in MuData. Available: [...]`

Your `data.views` list names a view that the file does not contain. Run [`mosa inspect`](commands/inspect.md) to see the names the file actually uses.

### `Mask layer 'mask' not in '<view>'. Available: [...]`

Set `data.mask_layer_name` to the observed-value mask's actual layer name. If absent, create a boolean layer marking measured entries as `True`; [`mosa convert`](commands/convert.md) writes this as `mask`.

### `MuData .obs missing 'model_type' column. Available: [...]`

The data file is missing the `model_type` column. This column is required by the shared data layer. It is used for stratified splits and interpreted by the selected model as described in [Library configuration](configuration/shared.md#data).

### `Cannot create a stratified train/val split: model_type class '<name>' has only 1 sample(s); need at least 2.`

A source needs enough samples in both splits. Set `evaluation.test_size: 0` to fit without a holdout, or revise the cohort if scientifically appropriate; do not merge distinct sources solely to bypass validation.

### `n_folds=5 exceeds the size of the smallest model_type class ('<name>', 3 samples); reduce n_folds, add more samples for that class, or use strategy='kfold'.`

Reduce the fold count, increase that source's sample count, or choose ordinary KFold if appropriate for the study.

## Conversion

### `View '<name>' (<path>): CSV appears to be samples x features (row overlap with conditionals IDs: 95%, threshold: 50%). MOSA expects features x samples: features as rows, samples as columns.`

Transpose the table so sample IDs are column headers and features are rows.

### `No samples found in the metadata that appear in any view.`

Compare the example IDs in the error for differences in prefixes, casing, or identifier systems. Use `--id-map` to map view IDs to the metadata's canonical IDs.

### `Conditionals '<path>' is missing required column 'model_id'.`

Name the required metadata columns exactly `model_id` and `model_type`.

### `Conditionals '<path>' has duplicate model_id values: [...]`

Your sample identifiers must be strictly unique.

### `View '<name>' (<path>): column '<col>' contains non-numeric values: [...]`

Check the named column for an extra index, a misplaced header, or an unrecognized missing-value marker.

### `Unsupported table format '<ext>'. Supported: .csv, .tsv, .txt, .parquet (optionally .gz/.bz2/.xz/.zip for the delimited formats).`

Convert the input to a supported format before running [`mosa convert`](commands/convert.md).

## Commands and saved files

### `Input <path> is missing view(s) ['meth'] required by the checkpoint. Checkpoint was trained on ['gexp', 'meth']; input has ['gexp'].`

Include every trained view with its original feature order; entirely masked rows are allowed for missing measurements. Extra views are ignored.

### `Cannot determine which model wrote '<path>': claimed by no model. Registered file types: {...}`

[`transform`](commands/transform.md) picks the model class from the checkpoint file. Each model declares the file types it writes: `.ckpt` or `.pt` for `mosa_vae`, `.hdf5` for `mofa`. Check that you passed the model file itself and not an output such as a parquet. If the message says the file is claimed by more than one model, two registered models accept the same file, and one of them has to tell its files apart.

### `View '<name>' features do not match the ones the model was trained on, in names or in order.`

Both models read features by position, so [`transform`](commands/transform.md) checks that each view lists the training features in the training order before using them. Reorder or rename the input's features to match the training data.

### `Search-space name(s) not a field of MOSAConfig: ['n_folds']. ['n_folds'] configure the evaluation protocol, which is held fixed for a study; set them in the config's evaluation block`

You cannot search the evaluation protocol parameters during a hyperparameter study. See the [Hyperparameter search](../guides/evaluation.md) page for details on what you can search.

### `No trial completed out of 50. First failure: ...`

All trials were pruned. Inspect the first reported failure or rerun with `--debug` for every failure traceback.

### `Plotting requires a 'tissue' column in the data's .obs; <path> has [...]. Training and cross-validation do not need it.`

Add tissue metadata before plotting. Training and cross-validation do not require it.

### `Output directory not found: <path>` and `No latent representations found under <path>`

Check that training completed and the selected output directory contains `full/latent.parquet` or the supported `train/` fallback.

## MOSA VAE

### Incompatible checkpoint configuration

The checkpoint must contain the data and model configuration expected by the loader. Use a compatible installation to load it, or retrain and save a compatible checkpoint.

### `target_batch '<name>' not in model_type categories: [...]`

Set `model.target_batch` to a source category represented in the data.

### `model.views must match data.views — missing in model: [...], extra in model: [...]`

Use exactly the same view names in `data.views` and the VAE's `model.views`.

### `PoE fusion requires all views to have the same last hidden dim, got {...}`

Match the final encoder widths across views, or use `poe_use_shared_head: false` for separate posterior heads.

## MOFA integration

### `MOFA projection requires training groups; unknown groups: [...]`

New samples must belong to a group represented during fitting, so its fitted centering and scaling can be reused. Correct the group label or fit a model including that group.

### `MOFA projection supports Gaussian views only`

The current weight-based projection supports continuous Gaussian views. Bernoulli and Poisson views require a likelihood-specific inference procedure.

### `The values of <n> fitted sample(s) differ from training`

The input reuses fitted sample IDs with changed observations. Those IDs retrieve learned factors. Use new sample IDs to project new observations, and preserve the training feature order.

### `MOFA only supports ASCII names, but <n> sample name(s) are not: [...]`

[MOFA](https://biofam.github.io/MOFA2/)'s model file reader decodes sample, feature, view and `model_type` names as ASCII, so a model trained on other names could not be opened again. The check runs before training. Rename the listed entries, for example with [`mosa convert --id-map`](commands/convert.md) for sample IDs.

### `Cannot reconstruct with '<path>': the file does not record how MOFA preprocessed the data.`

The [MOFA](https://biofam.github.io/MOFA2/) model file lacks the preprocessing metadata needed to reverse centering and scaling, so reconstructions cannot be returned to the input scale. [`transform`](commands/transform.md) without `--reconstruct` still works. Retrain the model to reconstruct.

### `model_type values [...] name the same group once written as text (e.g. 1 and '1').`

[MOFA](https://biofam.github.io/MOFA2/) stores group names as text, so a number and a string that read the same would be merged into one group. Give each `model_type` a distinct label.

### `MOFA dropped every factor during training: none explained more than drop_r2 of the variance in any view and group.`

Check data quality and the factor-retention threshold. `model.drop_r2: null` disables dropping; `0` still drops factors explaining no variance.

### `Cannot reconstruct view '<name>': MOFA fit it with a bernoulli likelihood (guessed from its values)`

[MOFA](https://biofam.github.io/MOFA2/) fits views whose values are all 0 or 1 as `bernoulli`, and views whose values are all integers as `poisson`. Their reconstructions are not on the data scale, so [`transform --reconstruct`](commands/transform.md) raises and [`train`](commands/train.md) skips their `recon_<view>.parquet`. `latent.parquet` is unaffected.

## Dependencies

### `the mofa model requires mofapy2 and mofax: pip install '.[mofa]'`

The [MOFA](https://biofam.github.io/MOFA2/) backend is optional and a base installation does not include it. Install the `mofa` extra as described in the [MOFA guide](../models/mofa.md#install-the-optional-dependencies).

### `AttributeError: module 'mudata' has no attribute 'set_options'`

The installed dependencies and code revision are incompatible. Update and reinstall as described in [Getting started](../getting-started.md#install-mosa), which installs the pinned compatibility range. The current code also guards use of `set_options` for [mudata](https://mudata.scverse.org/stable/) versions that do not expose it.

