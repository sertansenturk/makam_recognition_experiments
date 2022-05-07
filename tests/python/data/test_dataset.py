import numpy as np
import pytest

from mre.data.dataset import Dataset


class TestDataSet:
    def test_dataset(self):
        X = np.array([[1, 2], [2, 2], [3, 4]])
        y = np.array(["class1", "class2", "class2"])

        result = Dataset(X, y)

        np.testing.assert_array_equal(result.X, X)
        np.testing.assert_array_equal(result.y, y)

    def test_dataset_wrong_y_ndim(self):
        X = np.array([[1, 2], [2, 2], [3, 4]])
        y = np.array(
            [
                ["class1", "irrelevant"],
                ["class2", "irrelevant"],
                ["class2", "irrelevant"],
            ]
        )

        with pytest.raises(ValueError, match="y should have a single dimension"):
            Dataset(X, y)

    @pytest.mark.parametrize(
        "X,y",
        [
            (np.array([[1, 2], [2, 2], [3, 4]]),
             np.array(["class1", "class2"])),  # y has less
            (np.array([[1, 2], [2, 2]]),
             np.array(["class1", "class2", "class2"])),  # X has less
        ],
    )
    def test_dataset_X_y_mismatch(self, X, y):
        with pytest.raises(
            ValueError, match="First dimension of X should be equal to the size of y"
        ):
            Dataset(X, y)

    def test_dataset_y_empty(self):
        X = np.array([[1, 2], [2, 2], [3, 4]])
        y = np.array([])

        with pytest.raises(ValueError, match="y is empty"):
            Dataset(X, y)

    def test_dataset(self):
        X = np.array([])
        y = np.array(["class1", "class2", "class2"])

        with pytest.raises(ValueError, match="X is empty"):
            Dataset(X, y)
