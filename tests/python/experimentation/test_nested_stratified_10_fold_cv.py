from unittest import mock

import numpy as np
import pytest

from mre.experimentation.architecture import Architecture
from mre.experimentation.nested_stratified_10_fold_cv import NestedStratified10FoldCV
from mre.data.dataset import Dataset


@pytest.fixture
def dataset(scope="module"):
    X = np.array([[1, 2], [2, 2], [3, 4]])
    y = np.array(["class1", "class2", "class2"])

    return Dataset(X=X, y=y)


@pytest.fixture
def architectures(scope="module"):
    return [
        Architecture(
            "ml_model",
            estimator="some_estimator",
            param_grid={"param1": ["val1", "val2"]},
        )
    ]


@pytest.fixture
def mock_estimator(scope="module"):
    def mock_one_estimator(estimator_id):
        estimator = mock.MagicMock()
        estimator.best_params_ = f"mocked_best_params_{estimator_id}"
        estimator.cv_results_ = f"mocked_cv_results_{estimator_id}"

        return estimator

    return mock_one_estimator


@pytest.fixture(scope="module")
def cv():
    return NestedStratified10FoldCV(num_splits=2, num_trials=3)


class TestNestedStratified10FoldCV:
    @mock.patch(
        "mre.experimentation.NestedStratified10FoldCV._display_model_results_at_trial",
        return_value=None,
    )
    @mock.patch(
        "mre.experimentation.NestedStratified10FoldCV._collect_model_results_at_trial",
        return_value=None,
    )
    @mock.patch(
        "mre.experimentation.NestedStratified10FoldCV._cross_validate",
        return_value="scores",
    )
    @mock.patch(
        "mre.experimentation.NestedStratified10FoldCV._setup",
        return_value=("inner_cv", "outer_cv"),
    )
    def test_run(
        self,
        mock_setup,
        mock_cross_validate,
        mock_collect_model_results_at_trial,
        mock_display_model_results_at_trial,
        cv,
        dataset,
        architectures,
    ):
        cv.run(dataset=dataset, architectures=architectures)

        mock_setup.assert_has_calls([mock.call(ii) for ii in range(cv.num_trials)])
        mock_cross_validate.assert_has_calls(
            [
                mock.call(dataset, arch, "inner_cv", "outer_cv")
                for arch in architectures
                for _ in range(cv.num_trials)
            ]
        )
        mock_collect_model_results_at_trial.assert_has_calls(
            [
                mock.call(
                    "scores",
                    [],  # since results_list is mocked it's not populated
                    ii,
                    arch,
                )
                for arch in architectures
                for ii in range(cv.num_trials)
            ]
        )
        mock_display_model_results_at_trial.assert_has_calls(
            [
                mock.call("scores", arch, 8)  # length of "ml_model"
                for arch in architectures
                for ii in range(cv.num_trials)
            ]
        )

    def test_collect_model_results_at_trial(self, cv, mock_estimator):
        scores = {
            "train_score": ["train_sc1", "train_sc2"],
            "test_score": ["test_sc1", "test_sc2"],
            "fit_time": ["fit_t1", "fit_t2"],
            "score_time": ["score_t1", "score_t2"],
            "estimator": [mock_estimator(1), mock_estimator(2)],
        }
        results_list = [
            {
                "architecture": "ml_model",
                "train_score": "train_sc0",
                "test_score": "test_sc0",
                "fit_time": "fit_t0",
                "score_time": "score_t0",
                "split_id": 1,
                "trial_id": 2,
                "best_params_": "mocked_best_params_0",
                "cv_results_": "mocked_cv_results_0",
            }
        ]
        trial_id = 3
        architecture = Architecture(
            "ml_model",
            estimator="some_estimator",
            param_grid={"param1": ["val1", "val2"]},
        )

        cv._collect_model_results_at_trial(scores, results_list, trial_id, architecture)
        expected_results_list = [
            {
                "architecture": "ml_model",
                "train_score": "train_sc0",
                "test_score": "test_sc0",
                "fit_time": "fit_t0",
                "score_time": "score_t0",
                "split_id": 1,
                "trial_id": 2,
                "best_params_": "mocked_best_params_0",
                "cv_results_": "mocked_cv_results_0",
            },
            {  # appended to the existing items
                "architecture": "ml_model",
                "train_score": "train_sc1",
                "test_score": "test_sc1",
                "fit_time": "fit_t1",
                "score_time": "score_t1",
                "split_id": 0,
                "trial_id": 3,
                "best_params_": "mocked_best_params_1",
                "cv_results_": "mocked_cv_results_1",
            },
            {
                "architecture": "ml_model",
                "train_score": "train_sc2",
                "test_score": "test_sc2",
                "fit_time": "fit_t2",
                "score_time": "score_t2",
                "split_id": 1,
                "trial_id": 3,
                "best_params_": "mocked_best_params_2",
                "cv_results_": "mocked_cv_results_2",
            },
        ]
        print(results_list)
        print(expected_results_list)

        assert results_list == expected_results_list
