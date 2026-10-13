# Getting started

This page covers MOSA installation and a complete training example, from preparing input tables to reading the model's results. The example data and configuration are included in the repository, so you can follow the steps before preparing your own dataset.

## Install MOSA

MOSA requires [Python](https://www.python.org/) 3.11 or newer. Clone the repository and create a virtual environment to keep its dependencies separate from other [Python](https://www.python.org/) projects:

```bash
git clone https://github.com/QuantitativeBiology/mosa.git
cd mosa
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
mosa --help
```

On Windows, replace the activation command with `.venv\Scripts\activate`. The final command should display MOSA's available commands, confirming that the command-line interface is installed.

Keep this environment active and run the remaining commands from the `mosa` directory. The example uses paths relative to that directory, including paths written inside the configuration file.

## Prepare the example data

MOSA combines several omic datasets measured across a set of samples. Each omic dataset is called a *view*. Alongside the measurements, sample metadata identifies the source of each sample and can provide tissue or mutation information for the model.

For this example, generate two synthetic views named `gexp` and `meth`, together with sample metadata:

```bash
python docs/examples/make_data.py
```

The script writes the tables to `outputs/tutorial/input/`. They contain 60 samples, with 20 features in `gexp` and 12 in `meth`. Some measurements are missing, and the metadata assigns samples to three sources and two tissues. All measurements and labels are synthetic.

Convert these separate tables into a [MuData](https://mudata.scverse.org/stable/) file so MOSA can load the aligned views, metadata, and missing-value masks together:

```bash
mosa convert --conditionals outputs/tutorial/input/metadata.csv \
  --view gexp:outputs/tutorial/input/gexp.csv \
  --view meth:outputs/tutorial/input/meth.csv \
  --output outputs/tutorial/data.h5mu
mosa inspect --input outputs/tutorial/data.h5mu
```

Each `--view` argument gives a view name and its table path. Conversion aligns samples by identifier and records which measurements are observed. The [`inspect`](reference/commands/inspect.md) command then displays the stored dimensions, metadata, and missingness. Before continuing, check that it reports 60 samples and both views with the feature counts above.

The generator also writes 12 separate samples to `outputs/tutorial/new-input/` for the later projection example. Rerunning the generator replaces its tables.

## Configure the model

An experiment configuration tells MOSA which data to load, which model to train, and where to save the results. Use the supplied [experiment.yaml](examples/experiment.yaml) for this run; you do not need to create another file.

In that file, `data.path` points to the [MuData](https://mudata.scverse.org/stable/) file just created and `data.views` selects `gexp` and `meth`. The `model` block selects `mosa_vae` and defines a small network with 16 hidden units per view and four shared latent dimensions. These view names must match those in the dataset.

The configuration runs two epochs on CPU to keep the example short. Its `evaluation.test_size: 0.2` reserves 12 samples for validation, preserving the proportion from each source. The model fits on the remaining 48 samples and uses validation loss to monitor training.

## Validate and train

Use [`mosa validate`](reference/commands/validate.md) to check the configuration against the dataset before starting training:

```bash
mosa validate --config docs/examples/experiment.yaml
```

This checks compatibility, including view names and required metadata, without fitting the model. Once validation succeeds, start training with [`mosa train`](reference/commands/train.md):

```bash
mosa train --config docs/examples/experiment.yaml
```

Results are written to `outputs/tutorial/run/`, as specified by `model.output_dir`. For this example, check that the directory contains:

- `checkpoints/last.ckpt`: model parameters, fitted preprocessing, and training state, used to reload the model.
- `full/latent.parquet`: four latent coordinates for each of the 60 samples.
- `full/recon_gexp.parquet` and `full/recon_meth.parquet`: reconstructed measurements with the same feature columns as the input views.

When adapting the configuration for another experiment, choose a new output directory so its results are saved separately.

## Read the results

The [Parquet](https://parquet.apache.org/) files can be read into [pandas](https://pandas.pydata.org/) for analysis. In a [Python](https://www.python.org/) session, notebook, or script using the same environment and working directory, load the latent representation and one reconstructed view:

```python
import pandas as pd

z = pd.read_parquet("outputs/tutorial/run/full/latent.parquet")
x = pd.read_parquet("outputs/tutorial/run/full/recon_gexp.parquet")
print(z.shape)  # (60, 4)
print(x.shape)  # (60, 20)
```

Both tables use sample identifiers as their row index. The latent table gives a joint representation of the two views, while the reconstruction table estimates the `gexp` measurements, including entries missing from the input. The dimensions above confirm that all samples and the expected features were exported.

The `full/` results include training and validation samples. For assessing prediction accuracy, use held-out reconstruction results as described in [Evaluation and optimization](guides/evaluation.md).

You can now [project the 12 new samples](guides/training.md#prepare-new-samples) through the saved model. To replace the example with your own data, follow [Preparing a dataset](guides/preparing-data.md) for input formats, sample alignment, and metadata requirements.
