# MOSA — Multi-Omic Synthetic Augmentation

MOSA is a conditional variational autoencoder for integrating multiple omic data sources into a shared latent space. It is part of the Generative Multi-Omics Data Integration Library.

**Features:**
- Per-view encoders and decoders with configurable MLP architectures
- Two latent fusion methods: concatenation and Product of Experts (PoE)
- Optional adversarial batch correction and contrastive learning
- Tissue, batch, and mutation conditional inputs
- Feature-level missing data handling
- Macro (group-balanced) reconstruction loss
- KL warmup scheduling

## Quick start

```bash
pip install -e .
mosa train --config configs/example.yaml
mosa plot --config configs/example.yaml
```

## Documentation

- [Getting Started](docs/getting-started.md) — installation, data preparation, running your first experiment
- [Full documentation index](docs/index.md) — all guides and references
