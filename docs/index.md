# MOSA Documentation

MOSA (Multi-Omic Synthetic Augmentation) is a configurable VAE for integrating multiple omic data sources into a shared latent space.

## Guides

[Getting Started](getting-started.md) covers installation, data preparation, and running your first experiment.

[CLI Reference](cli.md) lists every command with flags, types, and defaults.

[Configuration Reference](configuration.md) documents every YAML config option with defaults and examples.

[Architecture Guide](architecture.md) explains the model structure, forward pass, training loop, and data pipeline.

[Data Pipeline](data-pipeline.md) details how data flows from MuData files to GPU-ready batches, including CSV format requirements and conversion troubleshooting.

[Plotting Guide](plotting.md) covers plot data sources, customization, and how to add new plots.

[Developer Guide](developing.md) explains how to extend MOSA with new fusion methods, losses, and models.

## Quick reference

```bash
# Install
pip install -e ".[dev]"

# Convert CSVs to MuData
mosa convert --samplesheet data/samplesheet.csv --view gexp:data/gexp.csv --output data/dataset.h5mu

# Verify conversion
mosa inspect --input data/dataset.h5mu

# Train
mosa train --config configs/example.yaml [--debug]

# Plot
mosa plot --config configs/example.yaml [--output-dir outputs/custom]
```
