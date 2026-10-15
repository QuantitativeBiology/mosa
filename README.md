<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-banner.svg">
    <img src="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-banner.svg" alt="MOSA — The Multi-Omic Synthetic Augmentation Library" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/mosalib/"><img src="https://img.shields.io/pypi/v/mosalib?color=137F82" alt="PyPI version"></a>
  <a href="https://github.com/QuantitativeBiology/mosa/actions/workflows/python.yml"><img src="https://github.com/QuantitativeBiology/mosa/actions/workflows/python.yml/badge.svg?branch=main" alt="CI status on main"></a>
  <a href="https://github.com/QuantitativeBiology/mosa/actions/workflows/docs.yml"><img src="https://github.com/QuantitativeBiology/mosa/actions/workflows/docs.yml/badge.svg?branch=main" alt="Documentation checks on main"></a>
  <a href="https://github.com/QuantitativeBiology/mosa/blob/main/pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-137F82" alt="Python 3.11 or newer"></a>
  <a href="https://github.com/QuantitativeBiology/mosa/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-BSD--3--Clause-137F82" alt="License: BSD 3-Clause"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="https://quantitativebiology.github.io/mosa/">Documentation</a> ·
  <a href="https://github.com/QuantitativeBiology/mosa/blob/main/CONTRIBUTING.md">Contributing</a> ·
  <a href="https://github.com/QuantitativeBiology/mosa/issues">Issues</a>
</p>

MOSA is a [Python](https://www.python.org/) library for multi-omics integration. It combines the MOSA variational autoencoder with tools for preparing data, training and evaluating models, optimizing their parameters, and analysing the results.

## The MOSA model

The [MOSA VAE](https://www.nature.com/articles/s41467-024-54771-4) is a conditional multiview variational autoencoder for bulk multi-omics integration. It learns a shared representation across omics and estimates missing measurements, allowing samples with incomplete profiles to be analysed together. Batch, tissue, and mutation information can condition reconstructions, while optional adversarial training discourages batch-associated information in the representation. See the [model guide](https://quantitativebiology.github.io/mosa/models/mosa-vae/) for available settings and their interpretation.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-vae-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-vae.svg">
    <img src="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-vae.svg" alt="MOSA VAE with one encoder and decoder per omic, a shared latent representation, decoder conditioning, and an optional source discriminator.">
  </picture>
</p>

## The library

The library provides command-line and [Python](https://www.python.org/) interfaces for running integration experiments from a configuration file. Prepare a dataset in [MuData](https://mudata.scverse.org/stable/), compare model settings through [cross-validation](https://quantitativebiology.github.io/mosa/guides/evaluation/) and [Optuna](https://optuna.org/) hyperparameter search, then use the fitted model to project new samples and analyse reconstructed profiles. Other integration methods can be added through the [model interface](https://quantitativebiology.github.io/mosa/development/adding-models/).

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-workflow-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-workflow.svg">
    <img src="https://raw.githubusercontent.com/QuantitativeBiology/mosa/main/docs/assets/brand/mosa-workflow.svg" alt="Omic matrices and metadata enter a common data interface for the MOSA VAE or a custom model. Library operations include configuration validation, training, cross-validation, hyperparameter optimization, saving results, and analysis plotting.">
  </picture>
</p>

## Quick start

### Install

Use [Python](https://www.python.org/) 3.11 or newer. Create a virtual environment to keep MOSA's dependencies separate from other projects.

#### From [PyPI](https://pypi.org/project/mosalib/)

Install MOSA and its dependencies from PyPI, then check that the command-line interface is available:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install mosalib
mosa --help
```

The final command lists MOSA's available commands. The package installs as `mosalib`; use `mosa` in the terminal or `import mosa` in Python.

#### From source

To work from a local copy of the code, clone the repository and install it in editable mode:

```bash
git clone https://github.com/QuantitativeBiology/mosa.git
cd mosa
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mosa --help
```

An editable installation uses the code in this checkout, so changes to the Python source are available without reinstalling the package.

On Windows, replace the activation command with `.venv\Scripts\activate`. Keep the environment active when running MOSA.

### Prepare and run

For a first run, follow [Getting started](https://quantitativebiology.github.io/mosa/getting-started/) using the supplied synthetic dataset. For your own data, follow [Preparing a dataset](https://quantitativebiology.github.io/mosa/guides/preparing-data/), then adapt the [example configuration](https://github.com/QuantitativeBiology/mosa/blob/main/docs/examples/experiment.yaml) with your data path, view names, and output directory. Save it as `config.yaml` and run:

```bash
mosa validate --config config.yaml
mosa train --config config.yaml
```

Validation checks the configuration against the dataset before training. After the fit, use [Analysing results](https://quantitativebiology.github.io/mosa/guides/analysis/) to read the saved outputs and generate diagnostic plots. The MOSA VAE supports CPU, [CUDA](https://developer.nvidia.com/cuda-toolkit), and Apple silicon through MPS; the example configuration uses CPU.

## Documentation

Use the [documentation](https://quantitativebiology.github.io/mosa/) to follow a complete task or look up a setting:

- [Using the library](https://quantitativebiology.github.io/mosa/#using-mosa): data preparation, training, evaluation, optimization, and plotting.
- [Models](https://quantitativebiology.github.io/mosa/models/): the MOSA VAE and external integrations.
- [Developing model integrations](https://quantitativebiology.github.io/mosa/#development): the interface, registration, persistence, and testing.
- [Reference](https://quantitativebiology.github.io/mosa/#reference): commands, configuration fields, saved files, and troubleshooting.

## Contributing

See [CONTRIBUTING.md](https://github.com/QuantitativeBiology/mosa/blob/main/CONTRIBUTING.md) for development setup, checks, and the model interface. Report bugs or discuss changes in [GitHub issues](https://github.com/QuantitativeBiology/mosa/issues).

## License

MOSA is licensed under the [BSD 3-Clause License](https://github.com/QuantitativeBiology/mosa/blob/main/LICENSE).
