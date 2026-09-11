# MOSA: Multi-Omic Synthetic Augmentation

MOSA is a framework for integrating multiple omic data sources into a shared latent space.

The tool provides a purpose-built conditional variational autoencoder (the MOSA VAE). The VAE supports per-view encoders and decoders with configurable MLP architectures. It provides two latent fusion methods: concatenation and Product of Experts (PoE). You can enable adversarial batch correction and contrastive learning, as well as condition the model on tissue, batch, and mutation inputs. It handles feature-level missing data and applies a macro (group-balanced) reconstruction loss alongside KL warmup scheduling.

Beyond the built-in VAE, MOSA is extensible. You can register custom models to use the same data ingestion, k-fold cross-validation, and Optuna hyperparameter search pipelines.

## Quick start

Install the tool and run the core pipeline:

```bash
pip install -e ".[dev]"

mosa inspect  --input <data>.h5mu
mosa validate --config <your-config>.yaml
mosa train    --config <your-config>.yaml
mosa plot     --config <your-config>.yaml
```

## Documentation

The [Getting started](docs/manual/01-getting-started.md) guide covers installation and your first run. The [Documentation index](docs/index.md) contains details on data preparation, configuration, and the command-line interface.
