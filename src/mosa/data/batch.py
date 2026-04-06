from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MOSABatch:
    """Collated batch of multi-omic samples with tensors indexed by batch."""

    encoder_inputs: dict[str, Tensor]   # {omic: [B, D_omic]} scaled features
    decoder_targets: dict[str, Tensor]  # {omic: [B, D_omic]} same as encoder_inputs
    missing_masks: dict[str, Tensor]    # {omic: [B, D_omic]} True where data present
    conditionals: Tensor                # [B, cond_dim] concatenated conditional vector
    tissue_labels: Tensor               # [B, n_tissues] one-hot tissue
    source_ids: Tensor                  # [B] integer model_type index
    sample_weights: Tensor              # [B] inverse-frequency class weight
    sample_names: list[str]             # [B] sample identifiers

    def to(self, device: torch.device) -> MOSABatch:
        """Move all tensors to the given device."""
        return MOSABatch(
            encoder_inputs={k: v.to(device) for k, v in self.encoder_inputs.items()},
            decoder_targets={k: v.to(device) for k, v in self.decoder_targets.items()},
            missing_masks={k: v.to(device) for k, v in self.missing_masks.items()},
            conditionals=self.conditionals.to(device),
            tissue_labels=self.tissue_labels.to(device),
            source_ids=self.source_ids.to(device),
            sample_weights=self.sample_weights.to(device),
            sample_names=self.sample_names,
        )


def collate_fn(samples: list[dict]) -> MOSABatch:
    """Stack individual samples into a batch."""
    omic_names = list(samples[0]["encoder_inputs"].keys())

    return MOSABatch(
        encoder_inputs={
            name: torch.stack([s["encoder_inputs"][name] for s in samples])
            for name in omic_names
        },
        decoder_targets={
            name: torch.stack([s["decoder_targets"][name] for s in samples])
            for name in omic_names
        },
        missing_masks={
            name: torch.stack([s["missing_masks"][name] for s in samples])
            for name in omic_names
        },
        conditionals=torch.stack([s["conditionals"] for s in samples]),
        tissue_labels=torch.stack([s["tissue_labels"] for s in samples]),
        source_ids=torch.stack([s["source_ids"] for s in samples]),
        sample_weights=torch.stack([s["sample_weights"] for s in samples]),
        sample_names=[s["sample_name"] for s in samples],
    )
