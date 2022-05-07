from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray

    def __post_init__(self):
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
