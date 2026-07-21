from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from mosa.data.dataset import MultiOmicDataset


class MultiOmicModel(ABC):
    """Interface for multi-omic integration models.

    All models receive data as MultiOmicDataset and return numpy arrays.
    Internal details (batching, framework choice, training loops) are
    implementation-specific and hidden behind this interface.
    """

    @abstractmethod
    def fit(
        self,
        train: MultiOmicDataset,
        val: MultiOmicDataset | None = None,
        resume_from: str | Path | None = None,
    ) -> None:
        """Train the model on the provided data.

        Parameters
        ----------
        train : MultiOmicDataset
            Training data.
        val : MultiOmicDataset or None
            Validation data. If None, no validation is performed.
        resume_from : str, Path, or None
            Path to a checkpoint to resume training from. If None, training
            starts from scratch.
        """

    @abstractmethod
    def transform(self, data: MultiOmicDataset) -> np.ndarray:
        """Project data into the learned latent space.

        Parameters
        ----------
        data : MultiOmicDataset
            Data to transform.

        Returns
        -------
        np.ndarray of shape [N, latent_dim]
            Latent representations.

        Notes
        -----
        Implementations that fit a persisted artifact lazily (e.g. MOFA, which
        builds its reader in save_outputs()) may require save_outputs() to be
        called after fit() before transform()/reconstruct() will work.
        Inductive models (e.g. the VAE) are usable immediately after fit().
        """

    @abstractmethod
    def reconstruct(self, data: MultiOmicDataset) -> dict[str, np.ndarray]:
        """Reconstruct omic views from data passed through the model.

        Parameters
        ----------
        data : MultiOmicDataset
            Data to reconstruct.

        Returns
        -------
        dict of {view_name: np.ndarray of shape [N, D_view]}
            Reconstructed feature matrices.
        """

    @abstractmethod
    def save_outputs(self, output_dir: str | Path | None = None) -> None:
        """Write latent representations and reconstructions.

        Writes to output_dir, or the model's configured output directory when
        output_dir is None.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Save model state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs) -> MultiOmicModel:
        """Load a saved model from disk."""
