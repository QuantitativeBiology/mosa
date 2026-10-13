# Adding a model

This page explains how to connect another integration method to MOSA's data loading, training, prediction, and evaluation commands. You write an *adapter*: a class that accepts MOSA datasets, calls the method's implementation, and returns predictions in the format expected by the library. The adapter also defines how the fitted model and its results are saved.

The library supplies configuration loading, sample splits, cross-validation scoring, and hyperparameter search. Your implementation supplies the model-specific preprocessing, fitting, and prediction. To make it selectable, define a configuration class and register it together with the adapter under a model name.

The guide follows the complete [PCA adapter example](../examples/pca_adapter.py). It uses [NumPy](https://numpy.org/) to combine views, impute missing entries with training feature means, and fit a shared principal-component representation. Its small implementation makes the integration points easy to inspect. It leaves features unscaled and uses no source or tissue conditioning, so high-variance features have more influence on its fit.

## Run the example

Create the tutorial dataset using [Getting started](../getting-started.md), then run the example script from the repository root with the MOSA environment active:

```bash
python docs/examples/pca_adapter.py validate --config docs/examples/pca.yaml
python docs/examples/pca_adapter.py train --config docs/examples/pca.yaml
```

The script imports and registers `example_pca` before calling MOSA's CLI. This makes the method available to [`validate`](../reference/commands/validate.md) and [`train`](../reference/commands/train.md) in that process. Calling the ordinary `mosa` executable alone would not import this example adapter.

The supplied [configuration](../examples/pca.yaml) selects the method and its component count:

```yaml
data:
  path: outputs/tutorial/data.h5mu
  views: [gexp, meth]
model:
  type: example_pca
  output_dir: outputs/tutorial/pca
  n_components: 4
evaluation:
  test_size: 0
  n_folds: 3
```

Here, `test_size: 0` fits all 60 samples. Training saves `outputs/tutorial/pca/model.tutorial-pca` and writes four latent coordinates per sample, together with reconstructed views, under `outputs/tutorial/pca/full/`. Use this working implementation as a reference while developing your own method.

## Define the model settings

A model configuration subclasses `ModelConfig` as a dataclass. The base class supplies `output_dir` and `random_seed`; add fields for settings owned by the integration method. The PCA example adds one field:

```python
from dataclasses import dataclass

from mosa.config import ModelConfig


@dataclass
class ExamplePCAConfig(ModelConfig):
    n_components: int = 4

    def __post_init__(self):
        super().__post_init__()
        if self.n_components < 1:
            raise ValueError("n_components must be positive")
```

`__post_init__()` checks conditions that can be decided from the configuration alone. Checks requiring matrix values belong in fitting or prediction; for example, the PCA adapter checks during fitting that the requested component count does not exceed the matrix dimensions.

The inherited `from_yaml_dict()` rejects unknown keys and constructs the configuration. For nested objects, such as per-view settings, override that method to parse them explicitly and use `check_unknown_keys()` at each level. [MOSAConfig](../../src/mosa/models/mosa/config.py) provides an implementation of nested parsing.

You can also override `validate_against_data(data_cfg, eval_cfg, summary)` to check requirements visible in the dataset's structural summary, such as required metadata columns. Raise an error for an incompatible setup and return warning strings for conditions that allow the operation to continue.

## Implement the model operations

Subclass [MultiOmicModel](../../src/mosa/models/api.py). Its constructor receives the data and model configurations; retain them as `self.data_cfg` and `self.model_cfg`. The library uses these to build the model, and [`mosa transform`](../reference/commands/transform.md) reads the restored data configuration to select input views and masks.

The class must implement six methods:

| Method | Role in the workflow |
|---|---|
| `fit(train, val=None, resume_from=None)` | Fit preprocessing and model parameters from the training subset, using validation or resumption where supported. |
| `transform(data)` | Return shared coordinates for the supplied samples. |
| `reconstruct(data)` | Return predicted measurements for every selected view on the input scale. |
| `save_outputs(output_dir=None)` | Write run exports and any model files needed after training. |
| `save(path)` | Save the fitted state needed to use the model again. |
| `load(path, **kwargs)` | Restore a model that can predict without repeating the fit; implement this as a class method. |

The input is a [MultiOmicDataset](../../src/mosa/data/dataset.py). Its `views` and `masks` contain samples × features arrays, `metadata` is indexed by sample ID, and `feature_names` preserves each view's column order. Use the masks to identify observed entries, including observed zeros. Converting a missing entry to zero for computation must not make it an observed training target.

### Fit preprocessing with the model

The PCA adapter concatenates views in configured order, computes each feature's mean from observed training values, and fills missing entries with that mean before fitting the components. It retains the means, component matrix, and feature names for subsequent prediction. Validation and new samples reuse that state.

Apply the same principle to your method: fit scaling, imputation, category encodings, and feature selection using the supplied training subset. Otherwise, validation measurements could influence preprocessing before their predictions are evaluated. Accept the interface's `val` and `resume_from` arguments even if the method does not use them, and make that behaviour clear to callers. The PCA example ignores validation during fitting and rejects resumption.

Initialize `epoch_history` on each instance. Populate it when the method has an epoch-based training loop; leave it empty otherwise, as the PCA adapter does. The base class's default list is shared, so using it directly would mix histories between instances.

### Preserve prediction order and scale

`transform(data)` returns a [NumPy](https://numpy.org/) array with shape `(n_samples, latent_dim)`. `reconstruct(data)` returns a dictionary mapping each view to a `(n_samples, n_features)` array. Both must preserve the supplied sample order, and reconstructed columns must follow the input feature order. Reverse fitted scaling before returning reconstructions, because the evaluator compares them with measurements on the input scale.

For PCA, projection subtracts the stored feature means and applies the fitted components. Reconstruction maps those coordinates back into feature space, adds the means, and separates the concatenated matrix into views. The adapter checks feature names and order before either operation.

Supporting held-out evaluation also requires prediction to work immediately after fitting. [`mosa cross-validate`](../reference/commands/cross-validate.md) calls `fit(train, val)` and then `reconstruct(val)` without first calling `save_outputs()`. If a backend needs a saved artifact for prediction, create it internally when needed. Set `supports_out_of_sample = False` when the method cannot predict held-out samples; cross-validation and optimization then reject it before fitting.

## Save the model and its outputs

A reusable model file must include every dependency of prediction, not only fitted weights. The PCA example saves its means, components, feature order, and data and model configurations. `load()` restores them, allowing the same predictions without access to the training tables.

Declare the file extensions your class writes in `checkpoint_suffixes`, and call `require_checkpoint_suffix(path)` in `save()`. The example uses `(".tutorial-pca",)`. When a file is loaded through the registry, the suffix identifies its owning class. For a shared suffix such as `.hdf5`, override `owns_checkpoint()` to inspect a format identifier or file structure, returning false for unrelated files. Exactly one registered class must claim a model file.

`save_outputs()` handles the run exports separately. The PCA example writes sample-indexed `full/latent.parquet` and `full/recon_<view>.parquet` files with the original feature names, then saves the reusable model file. These exports follow the [saved-file conventions](../reference/saved-files.md) used by the plotting command. The adapter creates the directories and honours an explicit `output_dir` override.

Document which samples the exports contain. PCA's `full/` directory contains only the samples supplied to fitting, even if a validation subset was provided. A loaded PCA file can predict, but cannot repeat that cohort export because the file does not store the training dataset. That limitation belongs to this example; your adapter should define and document its own export behaviour.

## Register the implementation

Registration associates a configuration class, model class, and name. The PCA class is decorated with `@register_model("example_pca", ExamplePCAConfig)`, using `register_model` from [mosa.models.registry](../../src/mosa/models/registry.py).

When the library reads `model.type: example_pca`, it creates an `ExamplePCAConfig`. `build_model()` then finds the adapter registered for that exact configuration type. Give your method a unique name and its own configuration subclass so dispatch can distinguish it from other models. Reusing a registered name replaces its existing registration.

The decorator also sets the class's `registered_name`, which can be stored as a format identifier in saved models. Registration does not implement prediction or export; those operations must already be provided by the adapter.

Keep optional backend imports inside the operations that need them. This allows users to import MOSA without installing every external method. If a selected backend is missing, raise `MissingDependencyError` with installation instructions.

## Make the registration available

The decorator runs when its module is imported. That import must happen before configuration loading, model construction, or saved-model dispatch; installing a package alone does not register it in a running process.

For an adapter included in this repository, import its module in [mosa.models](../../src/mosa/models/__init__.py). For a separately packaged adapter, import its registration module at the beginning of the calling script or notebook, before `load_config()`, `build_model()`, or `load_model()`.

A small launcher can make the adapter available to the normal CLI. For a package whose registration module is `my_package.mosa_adapter`, save the following as `run_my_method.py`:

```python
import my_package.mosa_adapter
from mosa.cli import main

if __name__ == "__main__":
    main()
```

Replace the import with your adapter's module and use the launcher for each command:

```bash
python run_my_method.py validate --config config.yaml
python run_my_method.py train --config config.yaml
```

This gives [`validate`](../reference/commands/validate.md) and [`train`](../reference/commands/train.md) access to the custom registration. Use the same launcher for [`transform`](../reference/commands/transform.md), so the new process can also identify and load your saved-model format. The PCA example script includes this launcher at the end of its implementation.

## Check the integration

First verify held-out prediction through the registry using the example launcher:

```bash
python docs/examples/pca_adapter.py cross-validate --config docs/examples/pca.yaml
```

[`cross-validate`](../reference/commands/cross-validate.md) creates fresh adapters per fold, passes training and validation subsets, and scores predictions on observed entries. A successful run checks more of the integration than calling the model class directly.

To check saved-model loading and new-sample prediction, prepare the [tutorial's new-sample dataset](../guides/training.md#prepare-new-samples), then run:

```bash
python docs/examples/pca_adapter.py transform \
  --checkpoint outputs/tutorial/pca/model.tutorial-pca \
  --input outputs/tutorial/new-data.h5mu \
  --output outputs/tutorial/pca-projection --reconstruct
```

[`transform`](../reference/commands/transform.md) loads the adapter from its saved file and writes coordinates and reconstructed views. The example produces 12 rows with four latent coordinates and reconstructions with the trained feature columns.

For your own method, use a small dataset with partial missingness, samples missing an entire view, distinct feature names, and deliberately changed sample order. Check that predictions preserve identifiers and ordering, that preprocessing uses training values only, and that predictions agree before and after saving and loading. Also test the errors expected for incompatible features, categories, or likelihoods. The [model interface reference](../reference/model-interface.md#verification) gives the complete verification checklist.

Before exposing the method to parameter search, run [`mosa optimize`](../reference/commands/optimize.md) through its launcher using a small search over fields in its configuration class. Each trial constructs a fresh configuration, so its validation must reject invalid parameter combinations. Search is limited to direct model fields; nested settings require separate experiments.

Finally, document the method's actual capabilities. For example, this PCA adapter accepts new source labels, requires at least one observed training value per feature, and maps a fully masked sample to zero coordinates and the training-mean reconstruction. Other methods may have different restrictions. State them beside the relevant fitting or prediction instructions so users can check their data before starting an experiment.
