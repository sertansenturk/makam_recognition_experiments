import pandas as pd
import pytest

from mre.experimentation.architecture import Architecture
from mre.experimentation.cross_validator import CrossValidator


@pytest.fixture(scope="module")
def architectures():
    return [
        Architecture(
            "ml_model", estimator=None, param_grid={"param1": ["val1", "val2"]}
        ),
        Architecture(
            "very_long_ml_model_name",
            estimator=None,
            param_grid={"param1": ["val1", "val2"]},
        ),
    ]


@pytest.fixture(scope="module")
def cv():
    return CrossValidator(num_splits=5, num_trials=3)


class TestCrossValidator:
    def test_max_architecture_name_len(self, cv, architectures):
        result = CrossValidator._max_architecture_name_len(architectures)

        expected = 23
        assert result == expected
