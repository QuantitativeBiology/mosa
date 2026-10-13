# Command reference

[Documentation](../../index.md)

Commands use `mosa <command> [options]`. Run `mosa --help` for the command list and `mosa <command> --help` for its options. All commands accept `-d` / `--debug` after the command name for diagnostic details and tracebacks.

## Prepare and check

| Command | Purpose |
|---|---|
| [convert](convert.md) | Align omic tables and metadata into MuData. |
| [inspect](inspect.md) | Summarise views, values, missingness, and metadata. |
| [validate](validate.md) | Check the configuration against the data structure without training. |

## Fit and evaluate

| Command | Purpose |
|---|---|
| [train](train.md) | Fit the selected model and export its results. |
| [cross-validate](cross-validate.md) | Train fold models and score held-out reconstruction. |
| [optimize](optimize.md) | Search model hyperparameters using cross-validation. |

## Analyse and reuse

| Command | Purpose |
|---|---|
| [plot](plot.md) | Generate diagnostics from compatible saved outputs and metadata. |
| [transform](transform.md) | Project samples through a saved model and optionally reconstruct views. |

The library dispatches model operations through the registry. Check [model capabilities](../../models/index.md#compare-supported-operations) for differences in validation, resumption, projection, and saved files.
