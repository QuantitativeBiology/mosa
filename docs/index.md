# MOSA documentation

MOSA is a [Python](https://www.python.org/) library for multi-omics integration. Its command-line and [Python](https://www.python.org/) interfaces cover preparing datasets, training and evaluating models, optimizing hyperparameters, and analysing results, so an experiment can be configured and run within the same library.

The MOSA VAE is the library’s main model: a conditional multiview variational autoencoder for bulk multi-omics integration. It learns a shared representation across a configurable number of omic datasets and estimates missing measurements. Batch, tissue, and mutation information can condition reconstructed profiles, while optional adversarial training discourages batch-associated information in the representation.

For a first run, follow [Getting started](getting-started.md). It provides example data and a configuration, explains each step from installation to training, and shows how to read the saved results. Once that example runs, use the guides below to prepare your own data or continue with evaluation and projection.

## Using MOSA

Start with data preparation if your measurements are still in separate tables. If you already have a compatible [MuData](https://mudata.scverse.org/stable/) file, continue with training. Evaluation compares reconstruction on held-out samples; analysis helps you examine the fitted model's outputs.

| Task | Guide |
|---|---|
| Align omic tables and sample metadata into a dataset | [Preparing a dataset](guides/preparing-data.md) |
| Train a model and apply it to new samples | [Training and projection](guides/training.md) |
| Evaluate reconstruction and optimize model parameters | [Evaluation and optimization](guides/evaluation.md) |
| Read model outputs and generate diagnostic plots | [Analysing results](guides/analysis.md) |
| Use MOSA from a script or notebook | [Library API for Notebooks and Experiments](guides/python.md) |

## Models

The library manages data loading, sample splitting, and evaluation, while each model implements its own fitting and prediction. The [MOSA VAE guide](models/mosa-vae.md) explains its architecture, conditioning, and training objectives. For the optional external integration, see [MOFA fitting and projection](models/mofa.md). [Available models](models/index.md) compares their supported operations and restrictions.

## Development

To change or extend MOSA, use the [architecture guide](development/architecture.md) to locate the relevant implementation. Follow [Adding a model](development/adding-models.md) to connect a different integration method to the library, or use the guides for [modifying the VAE](development/modifying-vae.md) and [extending data handling](development/data-pipeline.md) when changing those components. [Documentation maintenance](development/documentation-site.md) explains how to edit and preview this site.

## Reference

Use the reference pages alongside the guides when you need the exact [configuration fields](reference/configuration/shared.md), [command options](reference/commands/index.md), or [saved-file layouts](reference/saved-files.md). The [model interface](reference/model-interface.md) specifies what an adapter must implement. For help with a failed command, see [Troubleshooting](reference/troubleshooting.md).

[Project README](../README.md) · [Contributing](../CONTRIBUTING.md) · [BSD 3-Clause License](../LICENSE)
