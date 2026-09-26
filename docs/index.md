# MOSA Documentation

MOSA (Multi-Omic Synthetic Augmentation) is an extensible framework for integrating multiple omic data sources into a shared latent space. It includes a built-in, configurable VAE, and it allows you to register and train your own custom models.

The manual pages are numbered in reading order. The first three cover everything needed to train a model. The rest serve as reference material.

1. [Getting started](manual/01-getting-started.md) — installation and the standard run
2. [Preparing data](manual/02-preparing-data.md) — input formats, conversion, verification
3. [Configuration](manual/03-configuration.md) — every config key, with defaults
4. [Commands](#commands) — one page per command, listed below
5. [Hyperparameter search](manual/05-hyperparameter-search.md) — the search-space format
6. [Outputs](manual/06-outputs.md) — what a run writes to disk
7. [Plots](manual/07-plots.md) — what each diagnostic figure shows
8. [The MOSA VAE](manual/08-the-mosa-vae.md) — what the built-in model knobs mean
9. [Troubleshooting](manual/09-troubleshooting.md) — error messages and their causes
10. [Limitations](manual/10-limitations.md) — known rough edges

## Commands

Every command uses the format `mosa <command> [options]`, and every flag has a short form (`-c` for `--config`, `-o` for `--output`, and so on). Running `mosa <command> -h` explains what the command does, with an example, and lists the same flags documented in these pages.

1. [train](manual/04-commands/01-train.md) — fit a model from a config
2. [transform](manual/04-commands/02-transform.md) — project data through a saved model
3. [cross-validate](manual/04-commands/03-cross-validate.md) — k-fold reconstruction scoring
4. [optimize](manual/04-commands/04-optimize.md) — Optuna hyperparameter search
5. [plot](manual/04-commands/05-plot.md) — diagnostic figures from a finished run
6. [convert](manual/04-commands/06-convert.md) — build a MuData file from tables
7. [inspect](manual/04-commands/07-inspect.md) — summarise a MuData file
8. [validate](manual/04-commands/08-validate.md) — check a config and its data

All eight commands accept the `-d`, `--debug` flag. The [Troubleshooting](manual/09-troubleshooting.md) section explains how failures are reported and what common error messages mean.

## Terminology

We use the term view to refer to a single omic data type, such as gene expression, methylation, or copy number. While the standard biological term is modality, MOSA uses view throughout the codebase, including the `--view` CLI flag and the `data.views` configuration key.
