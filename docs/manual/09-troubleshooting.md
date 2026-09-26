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

### `Cross-validation is not supported for the 'mofa' model: it has no out-of-sample projection.`

The MOFA implementation cannot project samples it did not fit on, making k-fold scoring mathematically impossible. The tool raises this error before any training starts.

### `Input <path> is missing view(s) ['meth'] required by the checkpoint. Checkpoint was trained on ['gexp', 'meth']; input has ['gexp'].`

The `transform` command requires every view the model was originally trained on. Extra views in the new input are perfectly fine, but missing views are fatal.

### `Checkpoint <path> was written by an older MOSA version, before the config was split into data and model sections, and cannot be loaded.`

The checkpoint was trained with a MOSA version from before July 2026, which stored its configuration in a format the current version cannot read. Retrain the model with your current version, or use a checkpoint that was.

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
