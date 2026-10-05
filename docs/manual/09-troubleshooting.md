# Troubleshooting

The tool prints problems with your configuration, paths, or data as a single line (or short block) to stderr, prefixed with `Error:`, and exits with status 1. If you append `--debug` to any command, it prints the full Python traceback alongside the error message.

If you see a traceback print *without* using the `--debug` flag, it means MOSA did not anticipate the failure. You should report this as a bug.

## Config

### `Unknown key 'laten_dim' in model:. Did you mean 'joint_latent_dim'?`

You misspelled a configuration key. The tool rejects unknown keys rather than ignoring them, ensuring a typo never silently disables a setting. When the parser cannot find a close match, it prints the full list of valid keys instead of a suggestion.

### `Config <path> must contain top-level 'data:' and 'model:' blocks`

The file parsed successfully as YAML, but it is not a valid MOSA configuration file. This usually happens when an indentation slip nests the `data` block under another key.

### `model.type is required (e.g. 'mosa_vae', 'mofa')`

The `model` block requires a `type` key so the tool knows which configuration schema to apply to the rest of the block.

### `model.views must match data.views — missing in model: [...], extra in model: [...]`

Your two view lists disagree. Every view you load requires an architecture definition, and every architecture definition must apply to a loaded view.

### `PoE fusion requires all views to have the same last hidden dim, got {...}`

When you use `fusion_method: poe` alongside `poe_use_shared_head: true`, the shared mu and logvar head requires a single, uniform input size. You must either match the final `hidden_layer_dims` entry across all views, or set `poe_use_shared_head: false`.

## Data

### `data.path not found: <path>`

The specified MuData file or zarr directory does not exist. The tool resolves paths relative to your working directory, not relative to the configuration file's location.

### `View '<name>' not in MuData. Available: [...]`

Your `data.views` list names a view that the file does not contain. Run `mosa inspect` to see the names the file actually uses.

### `Mask layer 'mask' not in '<view>'. Available: [...]`

The specified view lacks a presence mask under the configured name. Files produced by `mosa convert` always include this mask. A file assembled manually or by other tools might lack it, or might store it under a different layer name. Update the `data.mask_layer_name` key to match your file.

### `MuData .obs missing 'model_type' column. Available: [...]`

The data file is missing the `model_type` column. This column is strictly required: the tool uses it as the batch label, the stratification key, and a conditional input.

### `Cannot create a stratified train/val split: model_type class '<name>' has only 1 sample(s); need at least 2.`

One of your data categories is too small to appear in both halves of the train/val split. You must merge it into another category, drop it entirely, or set `evaluation.test_size: 0` to skip the split.

### `n_folds=5 exceeds the size of the smallest model_type class ('<name>', 3 samples); reduce n_folds, add more samples for that class, or use strategy='kfold'.`

Stratified folds require at least `n_folds` members in every class. You must follow one of the suggestions in the error message.

## Conversion

### `View '<name>' (<path>): CSV appears to be samples x features (row overlap with conditionals IDs: 95%, threshold: 50%). MOSA expects features x samples: features as rows, samples as columns.`

Your input table is transposed. The tool detects this condition by checking whether your sample IDs appear as row labels instead of column headers.

### `No samples found in the metadata that appear in any view.`

The sample IDs do not match between your conditionals table and your view tables. The error message prints the first five IDs from each side. This usually makes the discrepancy visible: a prefix, a suffix, different casing, or an entirely different identifier system. Use the `--id-map` flag to apply a crosswalk if the two tables genuinely use different naming schemes.

### `Conditionals '<path>' is missing required column 'model_id'.`

Your metadata table lacks `model_id` and `model_type` columns, or they are named differently. The tool requires those exact column names.

### `Conditionals '<path>' has duplicate model_id values: [...]`

Your sample identifiers must be strictly unique.

### `View '<name>' (<path>): column '<col>' contains non-numeric values: [...]`

One of your view tables contains non-numeric data. This is often a stray index column, a header row read as data, or a placeholder string like `NA` that the pandas parser did not recognise as a missing value.

### `Unsupported table format '<ext>'. Supported: .csv, .tsv, .txt, .parquet (optionally .gz/.bz2/.xz/.zip for the delimited formats).`

You passed an unsupported file format, such as an Excel spreadsheet. You must convert it to a supported format first.

## Models and commands

### `MOFA projection requires training groups; unknown groups: [...]`

New samples must belong to a group represented during fitting, so its fitted centering and scaling can be reused. Correct the group label or fit a model including that group.

### `MOFA projection supports Gaussian views only`

The current weight-based projection supports continuous Gaussian views. Bernoulli and Poisson views require a likelihood-specific inference procedure.

### `The values of <n> fitted sample(s) differ from training`

The input reuses fitted sample IDs with changed observations. Those IDs retrieve learned factors. Use new sample IDs to project new observations, and preserve the training feature order.

### `Input <path> is missing view(s) ['meth'] required by the checkpoint. Checkpoint was trained on ['gexp', 'meth']; input has ['gexp'].`

The `transform` command requires every view the model was originally trained on. Extra views in the new input are perfectly fine, but missing views are fatal.

### `Checkpoint <path> was written by an older MOSA version, before the config was split into data and model sections, and cannot be loaded.`

The checkpoint was trained with a MOSA version from before July 2026, which stored its configuration in a format the current version cannot read. Retrain the model with your current version, or use a checkpoint that was.

### `Cannot determine which model wrote '<path>': claimed by no model. Registered file types: {...}`

`transform` picks the model class from the checkpoint file. Each model declares the file types it writes: `.ckpt` or `.pt` for `mosa_vae`, `.hdf5` for `mofa`. Check that you passed the model file itself and not an output such as a parquet. If the message says the file is claimed by more than one model, two registered models accept the same file, and one of them has to tell its files apart.

### `MOFA only supports ASCII names, but <n> sample name(s) are not: [...]`

MOFA's model file reader decodes sample, feature, view and `model_type` names as ASCII, so a model trained on other names could not be opened again. The check runs before training. Rename the listed entries, for example with `mosa convert --id-map` for sample IDs.

### `Cannot reconstruct with '<path>': the file does not record how MOFA preprocessed the data.`

The MOFA model file was written by mofapy2 directly, or by a MOSA version from before September 2026. Neither records how the data was centered and scaled, so reconstructions cannot be returned to the original scale. `transform` without `--reconstruct` still works. Retrain the model to reconstruct.

### `model_type values [...] name the same group once written as text (e.g. 1 and '1').`

MOFA stores group names as text, so a number and a string that read the same would be merged into one group. Give each `model_type` a distinct label.

### `MOFA dropped every factor during training: none explained more than drop_r2 of the variance in any view and group.`

MOFA found no structure above the `drop_r2` threshold, so no model was written. Check the input first: this is what pure noise produces. Setting `model.drop_r2: null` turns factor dropping off. A value of `0` still drops factors that explain nothing.

### `Cannot reconstruct view '<name>': MOFA fit it with a bernoulli likelihood (guessed from its values)`

MOFA fits views whose values are all 0 or 1 as `bernoulli`, and views whose values are all integers as `poisson`. Their reconstructions are not on the data scale, so `transform --reconstruct` raises and `train` skips their `recon_<view>.parquet`. `latent.parquet` is unaffected.

### `View '<name>' features do not match the ones the model was trained on, in names or in order.`

Both models read features by position, so `transform` checks that each view lists the training features in the training order before using them. Reorder or rename the input's features to match the training data.

### `target_batch '<name>' not in model_type categories: [...]`

The `model.target_batch` key must specify a category that your data actually contains.

### `Search-space name(s) not a field of MOSAConfig: ['n_folds']. ['n_folds'] configure the evaluation protocol, which is held fixed for a study; set them in the config's evaluation block`

You cannot search the evaluation protocol parameters during a hyperparameter study. See the [Hyperparameter search](05-hyperparameter-search.md) page for details on what you can search.

### `No trial completed out of 50. First failure: ...`

The hyperparameter study pruned every trial. The tool includes the first failure's message for context. Run the command with `--debug` to print tracebacks for all of them.

### `Plotting requires a 'tissue' column in the data's .obs; <path> has [...]. Training and cross-validation do not need it.`

Every diagnostic figure uses the `tissue` label for colouration. The `plot` command is the only command that strictly requires this column to exist.

### `Output directory not found: <path>` and `No latent representations found under <path>`

The `plot` command reads the directory structure that a finished `train` run leaves behind. These errors mean either the training run did not happen, or your `--output-dir` points somewhere else.

## Dependencies

### `the mofa model requires mofapy2 and mofax: pip install '.[mofa]'`

The MOFA backend is optional and a base `pip install -e .` does not include it. Install the `mofa` extra, as detailed in the [Getting started](01-getting-started.md) guide.

### `AttributeError: module 'mudata' has no attribute 'set_options'`

mudata 0.4 removed `set_options`, which MOSA uses. Your installation predates the version pin. Update and reinstall as described in [Getting started](01-getting-started.md#updating), which installs a compatible mudata.

See also: [validate](04-commands/08-validate.md) · [Limitations](10-limitations.md)
