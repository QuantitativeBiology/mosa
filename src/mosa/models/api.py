from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mosa.config import DataConfig, ModelConfig

from mosa.data.dataset import MultiOmicDataset


class MultiOmicModel(ABC):
    """Interface for multi-omic integration models.

    All models receive data as MultiOmicDataset and return numpy arrays.
    Internal details (batching, framework choice, training loops) are
    implementation-specific and hidden behind this interface.
    """

    supports_out_of_sample: bool = True
    """Whether transform()/reconstruct() accept samples not seen during fit().
    False for transductive models, which cross_validate() rejects before
    training anything."""

    checkpoint_suffixes: tuple[str, ...] = ()
    """File suffixes save() writes. registry.load_model() offers a path to
    every model through owns_checkpoint(), which by default matches these."""

    registered_name: str
    """The name the model is registered under. Stamped by @register_model, so
    it is annotated rather than assigned: an unregistered subclass must not
    silently report an empty name."""

    epoch_history: list[dict] = []
    """Per-epoch training curve, populated by fit() where available: one dict
    per epoch with at least 'epoch', and 'train_loss'/'val_loss' when a val
    split was given. Empty for models with no per-epoch training loop (e.g.
    MOFA) or when fit() was called without validation data."""

    # Documents the constructor build_model() calls; abstract would change
    # how subclasses may be defined.
    def __init__(self, data_cfg: DataConfig, model_cfg: ModelConfig) -> None:  # noqa: B027
        """Initialize the model with data and model configurations."""
        ...

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
        Implementations that fit a persisted artifact lazily may require
        save_outputs() to be called after fit() before transform()/
        reconstruct() will work. Inductive models are usable immediately
        after fit().
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

    @classmethod
    def require_checkpoint_suffix(cls, path: str | Path) -> None:
        """Raise unless path has a suffix load_model() routes back to this class."""
        if Path(path).suffix not in cls.checkpoint_suffixes:
            raise ValueError(
                f"Cannot save to '{path}': {cls.__name__} files must end in one "
                f"of {list(cls.checkpoint_suffixes)}, or load_model() cannot "
                f"tell which model wrote them."
            )

    @classmethod
    def owns_checkpoint(cls, path: str | Path) -> bool:
        """Whether path is a file this class's save() wrote.

        Models sharing a suffix override this to inspect the file, so
        registry.load_model() finds exactly one owner.
        """
        return Path(path).suffix in cls.checkpoint_suffixes
