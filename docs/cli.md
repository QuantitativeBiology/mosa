# CLI Reference

All commands follow the pattern `mosa <command> [options]`. Run `mosa --help` or `mosa <command> --help` for inline help.

## Errors

Anything you can fix — a bad path, an unknown config key, data that does not match the model — is reported as a single line on stderr, and the command exits 1:

```
$ mosa train --config configs/my_experiment.yaml
Error: data.path not found: data/dataset.h5mu
```

Misspelled config keys suggest the intended name:

```
Error: Unknown key 'laten_dim' in model:. Did you mean 'joint_latent_dim'?
```

Adding `--debug` to any command prints the full traceback alongside the message. Reach for it when an error looks like a bug in MOSA rather than a problem with your config or data — a traceback shown without `--debug` is always worth reporting.

Use `mosa validate --config <path>` to check a config, and the data it points at, without training.

## train

Train a MOSA model from a YAML config file.

```bash
mosa train --config configs/my_experiment.yaml [--debug]
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `--config` | path | yes | Path to YAML config file |
| `--debug` | flag | no | Enable verbose debug logging |

See [Configuration Reference](configuration.md) for all config options.

## plot

Generate diagnostic plots from a completed training run.

```bash
mosa plot --config configs/my_experiment.yaml [--output-dir outputs/run] [--debug]
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `--config` | path | yes | Path to YAML config file |
| `--output-dir` | path | no | Override output directory (defaults to `output_dir` in config) |
| `--debug` | flag | no | Enable verbose debug logging |

Generates UMAPs, loss curves, reconstruction scatter plots, and clustering metrics under `<output_dir>/plots/`. See [Getting Started](getting-started.md) for expected output structure.

## convert

Convert CSV files to MuData format (`.h5mu` or `.zarr`).

```bash
mosa convert \
  --samplesheet data/samplesheet.csv \
  --view gexp_voom:data/gexp_voom.csv \
  --view meth_combat:data/meth_combat.csv \
  --output data/dataset.h5mu \
  [--mutations data/mutations.csv] \
  [--format h5mu] \
  [--debug]
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `--samplesheet` | path | yes | Samplesheet CSV. Required columns: `model_id`, `model_type`. Optional: `tissue`. |
| `--view` | `name:path` | yes (repeat) | Omic modality as `name:path`. Repeat for each modality. CSV must be features × samples. |
| `--mutations` | path | no | Binary mutations CSV (features × samples). Gene columns become `mutation_*` in `.obs`. |
| `--output` | path | yes | Output file path (`.h5mu` or `.zarr`) |
| `--format` | `h5mu`\|`zarr` | no | Output format. Default: `h5mu`. |
| `--debug` | flag | no | Enable verbose debug logging, including per-view sample counts |

The command validates all inputs before writing any output — samplesheet columns, CSV orientation, and numeric content. See [Data Pipeline — Converting CSVs](data-pipeline.md#converting-csvs-to-mudata) for full format requirements, orientation caveats, and troubleshooting.

## inspect

Print a human-readable summary of a MuData file. Use this to verify a conversion was correct.

```bash
mosa inspect --input data/dataset.h5mu [--debug]
```

| Flag | Type | Required | Description |
|---|---|---|---|
| `--input` | path | yes | Path to `.h5mu` file or `.zarr` directory |
| `--debug` | flag | no | Enable verbose debug logging |

Prints sample count, per-modality feature count and presence rate, data ranges, `.obs` metadata summary, and sample ID examples. See [Data Pipeline — Verifying the output](data-pipeline.md#verifying-the-output) for an annotated example.
