# Limitations

## Data and conversion

The `convert` command expects features-by-samples tables in `.csv`, `.tsv`, `.txt`, or `.parquet` formats. If your data uses a different layout or an unsupported format, you must adapt `src/mosa/data/io.py` manually. The tool does not offer a configuration option to change this behaviour.

You are responsible for data preprocessing. While MOSA handles scaling and missing-value imputation internally before passing data to the model, it expects you to perform any necessary normalisation, upstream batch-effect removal, and feature filtering beforehand. The `convert` command deliberately skips these steps to prevent data leakage; fitting a filter on the entire dataset would artificially inflate every subsequent cross-validation score.

## Model

The encoders and decoders strictly use MLP architectures. You can configure their depth and width, but you cannot change their underlying type. If you require a convolutional or attention-based component, you must write a custom model implementation to support it.

The model applies no special distributional treatment to discrete views. It calculates their reconstruction loss using the exact same squared error function applied to continuous views.

MOFA projects unseen samples using observed Gaussian features and training-derived preprocessing. Projection is restricted to known groups and uses minimum-norm least squares rather than posterior inference. Non-Gaussian projection and samples with no observed features are unsupported. See [Configuration](03-configuration.md#model-for-type-mofa) for the input requirements.

## Evaluation

The cross-validation pipeline scores reconstruction accuracy and nothing else. The reported metric is the masked, variance-normalized MSE from `reconstruct()`. This score provides no information about whether the resulting latent space is biologically meaningful, which is typically the primary concern for downstream analysis.

The hyperparameter search only evaluates top-level configuration fields. You cannot search over per-view architectural parameters like `views.<name>.hidden_layer_dims`.

## Outputs

Successive runs write into the same output directory. If you train twice using the exact same configuration, the tool creates a new `lightning_logs/version_N` directory but silently overwrites the output parquets. You must specify separate `output_dir` values to preserve the outputs of distinct runs.

## Scale

The tool loads all data into memory simultaneously. It reads both `.h5mu` and `.zarr` datasets entirely upfront. While the data module contains a lazy zarr reader, the codebase never constructs it and no caller passes the required `zarr_path`. As a result, using `.zarr` currently only provides an alternative storage format; it does not reduce the tool's memory footprint. The `scaler_sample_frac` parameter applies exclusively to this disabled path and currently has no effect.

## Integrations

MOSA operates as a standalone CLI with no built-in workflow-manager or experiment-tracking integrations. You can compose it with Snakemake, Nextflow, or bash loops exactly as you would any other command-line tool, but it does not ship with native support for them. It lacks MLflow or Weights & Biases hooks. The only metrics output is the Lightning CSV logs stored under `lightning_logs/`.

See also: [Troubleshooting](09-troubleshooting.md)
