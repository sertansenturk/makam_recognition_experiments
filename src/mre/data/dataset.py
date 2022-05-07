from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray  # features (shape: num_samples x num_features)
    y: np.ndarray  # labels (shape: num_samples, )

    def __post_init__(self):
        """validates the Dataset

        Raises:
            ValueError: if X is empty
            ValueError: if y is empty
            ValueError: if y has more just the labels
            ValueError: if X and y has different number of samples
        """
        if self.X.size == 0:
            raise ValueError("X is empty.")
        if self.y.size == 0:
            raise ValueError("y is empty.")
        if self.y.ndim > 1:
            raise ValueError(
                f"y should have a single dimension (only labels). "
                f"y.ndim = {self.y.ndim}"
            )
        if self.X.shape[0] != self.y.size:
            raise ValueError(
                f"First dimension of X should be equal to the size of y: "
                f"X.shape = {self.X.shape}, y.size = {self.y.size}"
            )

    def __str__(self) -> str:
        return (
            f"Dataset with {self.y.size} samples and {self.X.shape[1]} "
            "feature dimensions."
        )
