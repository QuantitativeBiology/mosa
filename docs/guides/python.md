# Library API for Notebooks and Experiments

This page explains how to load data, train and save a model, project new samples, and evaluate model settings from a [Python](https://www.python.org/) script or notebook. Use this interface when you need to control sample splits, inspect predictions in memory, or combine MOSA with your own analysis code.

The examples build on one another and use the files from [Getting started](../getting-started.md). Run them in the same session, from the repository root, with the MOSA environment active. The configuration selects the MOSA VAE; other registered models use the same interface with their own settings and supported operations.

## Load data and fit a model

Start by loading the experiment configuration and checking it against the dataset. Set a separate output directory for this example so its exports do not replace the earlier command-line run:

```python
from dataclasses import replace

from mosa.data.io import load_mudata
from mosa.models.registry import build_model
from mosa.utils import load_config, seed_everything, validate_config_against_data

cfg = load_config("docs/examples/experiment.yaml")
cfg.model = replace(cfg.model, output_dir="outputs/tutorial/python-run")

for warning in validate_config_against_data(cfg):
    print(warning)

seed_everything(cfg.model.random_seed)
data = load_mudata(cfg.data.path, cfg.data.views, cfg.data.mask_layer_name)
```

`load_mudata()` returns a `MultiOmicDataset` containing the selected view matrices, observed-value masks, feature names, and sample metadata. The model receives this container rather than reading the file itself.

Unlike [`mosa train`](../reference/commands/train.md), calling `model.fit()` directly does not create a validation split from `evaluation.test_size`. Split the sample indices explicitly, then pass the two subsets to the model. Here, [scikit-learn](https://scikit-learn.org/stable/) reserves the configured fraction while stratifying by source:

```python
import numpy as np
from sklearn.model_selection import train_test_split

train, val = data, None
if cfg.evaluation.test_size > 0:
    train_idx, val_idx = train_test_split(
        np.arange(data.n_samples),
        test_size=cfg.evaluation.test_size,
        stratify=data.metadata["model_type"],
        random_state=cfg.model.random_seed,
    )
    train = data.subset(train_idx)
    val = data.subset(val_idx)

model = build_model(cfg.data, cfg.model)
model.fit(train, val)
model.save_outputs()
model.save("outputs/tutorial/python-run/fitted.ckpt")
```

With the tutorial configuration, this fits 48 samples and uses the remaining 12 for validation. Setting `test_size` to zero leaves `train` as the complete dataset and `val` as `None`. The VAE fits preprocessing on the supplied training subset and reuses it for validation.

`save_outputs()` writes latent coordinates and reconstructed profiles under the model's output directory. `save()` explicitly writes the fitted VAE to `fitted.ckpt`, including the preprocessing and configuration needed to reload it. These are separate operations: exported predictions are useful for analysis, while the saved model is needed to predict from another dataset. Other model implementations use their own saved-file formats.

## Reload the model and predict new samples

Reloading lets you apply the fitted model in another session without repeating training. First prepare `outputs/tutorial/new-data.h5mu` using the [new-sample conversion example](training.md#prepare-new-samples), then load the checkpoint saved above:

```python
from mosa.models.registry import load_model

loaded_model = load_model("outputs/tutorial/python-run/fitted.ckpt")
new_data = load_mudata(
    "outputs/tutorial/new-data.h5mu",
    loaded_model.data_cfg.views,
    loaded_model.data_cfg.mask_layer_name,
)
z = loaded_model.transform(new_data)
x_hat = loaded_model.reconstruct(new_data)

print(z.shape)              # (12, 4)
print(x_hat["gexp"].shape)   # (12, 20)
```

`transform()` returns each sample's coordinates in the fitted latent space. `reconstruct()` returns predicted measurements for each view, including missing entries, on the input scale. Both operations reuse the fitted model and preprocessing without updating them.

The results are [NumPy](https://numpy.org/) arrays in the supplied sample order. `z` has one row per sample and one column per latent dimension; `x_hat` maps each view name to its samples × features matrix. To retain sample and feature identifiers for analysis, create [pandas](https://pandas.pydata.org/) tables:

```python
import pandas as pd

latent = pd.DataFrame(
    z,
    index=new_data.sample_names,
    columns=[f"z_{i}" for i in range(z.shape[1])],
)
recon_gexp = pd.DataFrame(
    x_hat["gexp"],
    index=new_data.sample_names,
    columns=new_data.feature_names["gexp"],
)
```

These calls return predictions in memory; they do not automatically write projection files. Save the tables yourself if needed. The input must retain the trained views, feature names and order, and supported source labels, as described in [Training and projection](training.md#prepare-new-samples).

## Evaluate a configuration

Use `cross_validate()` to fit fresh models across the configured folds and obtain held-out predictions. It takes the full dataset and configuration, rather than the model already fitted above:

```python
from mosa.models.evaluation import cross_validate

scores = cross_validate(data, cfg.data, cfg.model, cfg.evaluation)
print(scores["mean"])
print(scores["per_view"])
held_out_gexp = scores["reconstructions"]["gexp"]
```

`scores["mean"]` is the pooled aggregate NMSE, and `scores["per_view"]` contains the error and observed-entry count for each view. `scores["reconstructions"]` contains held-out predictions in the original dataset's sample and feature order. Use `data.masks["gexp"]` to select observed measurements when comparing `held_out_gexp` with the input values.

The function also returns per-fold scores and available epoch histories. It returns these results in memory; the additional reconstruction and history files described for [`mosa cross-validate`](../reference/commands/cross-validate.md) are written by the CLI. See [Evaluation and optimization](evaluation.md#scoring) for score interpretation.

## Optimize parameters and fit the selected model

Load the example search space and pass it to `optimize()`. The following call evaluates two parameter combinations, each using the three folds in the tutorial configuration:

```python
from mosa.models.optimize import load_search_space, optimize

search_space = load_search_space("docs/examples/search-space.yaml")
result = optimize(
    data,
    cfg.data,
    cfg.model,
    search_space,
    n_trials=2,
    eval_cfg=cfg.evaluation,
)
print(result["best_params"])
print(result["best_value"])
study = result["study"]
```

The result contains the best sampled parameters, their pooled aggregate NMSE, and the [Optuna](https://optuna.org/) study object for further inspection. Optimization trains temporary fold models to compare parameter combinations; it does not update the earlier model or leave a final fitted model.

Apply the selected parameters to a configuration copy and fit a new model. Reuse the explicit training and validation subsets, and choose a new output directory:

```python
selected_cfg = replace(cfg.model, **result["best_params"])
selected_cfg = replace(selected_cfg, output_dir="outputs/tutorial/python-selected")

seed_everything(selected_cfg.random_seed)
selected_model = build_model(cfg.data, selected_cfg)
selected_model.fit(train, val)
selected_model.save_outputs()
selected_model.save("outputs/tutorial/python-selected/fitted.ckpt")
```

This saves the selected VAE and its exports for analysis and future prediction. Retain the selected parameter values with the experiment records so you can identify the configuration used for that fit.

For a separately packaged model integration, import its registration module before calling `load_config()`, `build_model()`, or `load_model()`. [Adding a model](../development/adding-models.md) explains registration and includes a runnable adapter example.
