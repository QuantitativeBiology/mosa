# MOSA Documentation

MOSA (Multi-Omic Synthetic Augmentation) is a configurable VAE for integrating multiple omic data sources into a shared latent space. It is part of the Generative Multi-Omics Data Integration Library.

## Guides

| Guide | Audience | Contents |
|-------|----------|----------|
| [Getting Started](getting-started.md) | All users | Installation, data preparation, running your first experiment |
| [Configuration Reference](configuration.md) | All users | Every YAML config option with defaults and examples |
| [Architecture Guide](architecture.md) | Users who want to understand or modify the model | Model structure, forward pass, training loop, data pipeline |
| [Plotting Guide](plotting.md) | Users who want to understand or customize plots | Data sources, how to edit plots, how to add new ones |
| [Developer Guide](developing.md) | Users who want to extend MOSA | Adding fusion methods, losses, and new models |

## Quick reference

```bash
# Install
pip install -e ".[dev]"

# Train
mosa train --config configs/example.yaml [--debug]

# Plot
mosa plot --config configs/example.yaml [--output-dir outputs/custom]
```
