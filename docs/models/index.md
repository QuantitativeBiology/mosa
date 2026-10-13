# Available models

This page introduces the integration models available through MOSA and compares the operations they support. The MOSA VAE is the library's main model and is included in the base installation. An optional adapter connects the external [MOFA](https://biofam.github.io/MOFA2/) model to the same data preparation, training, and evaluation commands.

Select the model with `model.type` in the experiment configuration. The library loads the dataset and manages sample splitting and scoring, while the selected model determines how measurements are preprocessed, combined, and reconstructed. Its settings and prediction requirements therefore need to match the experiment you intend to run.

## MOSA VAE

The [MOSA VAE](mosa-vae.md) is a conditional multiview variational autoencoder for bulk multi-omics integration. It learns a shared representation across a configurable number of omic datasets and reconstructs profiles for samples with incomplete measurements. Decoders use the source label and optional tissue and mutation information, while adversarial training can discourage source-associated information in the representation. A separate tissue-based contrastive objective can encourage tissue-related structure.

Use `model.type: mosa_vae` to select it. The [model guide](mosa-vae.md) explains how missing measurements, conditioning, and training objectives affect the model; the [VAE configuration reference](../reference/configuration/mosa-vae.md) lists the network and training settings.

## Optional MOFA integration

[MOFA+](https://biofam.github.io/MOFA2/) is a factor-analysis model for integrating omic datasets through shared latent factors. The adapter fits it through [`mofapy2`](https://biofam.github.io/MOFA2/) and reads saved models through [`mofax`](https://github.com/bioFAM/mofax). It provides another integration method within the MOSA workflow, using its own likelihoods and preprocessing rather than the VAE's neural-network settings.

Select it with `model.type: mofa` after installing the optional dependencies. The [integration guide](mofa.md) covers installation, configuration, and prediction. Check its requirements before planning held-out evaluation: the adapter's new-sample projection and reconstruction support Gaussian views and source groups represented during fitting.

## Compare supported operations

The table describes the current implementations. In both, the source or batch label comes from the dataset's `model_type` metadata column.

| Operation | MOSA VAE (`mosa_vae`) | [MOFA](https://biofam.github.io/MOFA2/) integration (`mofa`) |
|---|---|---|
| Missing measurements | Masks exclude missing entries from reconstruction loss; the model predicts complete profiles. | Missing measurements are handled during fitting; projection needs observed features from which to estimate factors. |
| Validation during fitting | Validation loss supports monitoring and early stopping. | Held-out samples do not drive fitting or early stopping. |
| [Prediction for new samples](../reference/commands/transform.md) | Fitted encoders and preprocessing produce latent coordinates and reconstructed profiles. | Factors are estimated by least squares using fitted Gaussian-view weights and preprocessing. |
| [Cross-validation](../reference/commands/cross-validate.md) and [parameter optimization](../reference/commands/optimize.md) | Held-out reconstruction is supported. | Requires Gaussian views and known source groups in the held-out data. |
| Source conditioning | Decoders use the source label; optional adversarial training discourages source information in the latent representation. | Source labels define groups used by the factor model and its preprocessing. |
| Tissue and mutation information | Optional decoder conditionals; tissue labels can also supervise the contrastive objective. | Not used as decoder conditionals or contrastive labels. |
| Reconstruction under another source condition | The decoder source label can be changed to a source seen during fitting. | Not implemented. |
| [Resuming training](../reference/commands/train.md) | Supported through [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) checkpoints. | Not supported. |
| Saved model format | `.ckpt` or `.pt`. | `.hdf5`. |

For new-sample prediction, retain every trained view with the same feature names and order. Missing measurements can remain masked within that structure, but both implementations reject source labels absent during fitting. Their model guides explain the remaining input and prediction requirements.

When comparing methods, use the same retained cohort, observed-value masks, and evaluation splits, and account for each method's preprocessing and likelihood assumptions. The common commands provide a consistent procedure for running the comparison; the per-view scores and downstream analyses show how the fitted models differ.

## Add another integration method

A method becomes available to the library through an adapter that implements the model interface and registers its configuration. Installing a package alone does not register it with MOSA. Follow [Adding a model](../development/adding-models.md) for a runnable example, or consult the [model interface reference](../reference/model-interface.md) for prediction and saved-model requirements.
