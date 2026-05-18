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
    ) -> None:
        """Train the model on the provided data.

        Parameters
        ----------
        train : MultiOmicDataset
            Training data.
        val : MultiOmicDataset or None
            Validation data. If None, no validation is performed.
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
    def save(self, path: str | Path) -> None:
        """Save model state to disk."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path, **kwargs) -> MultiOmicModel:
        """Load a saved model from disk."""
