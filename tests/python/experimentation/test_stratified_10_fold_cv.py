from unittest import mock

import numpy as np
import pytest

from mre.experimentation.architecture import Architecture
from mre.experimentation.stratified_10_fold_cv import Stratified10FoldCV
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
def mock_grid_search_cv(scope="module"):
    def mock_one_grid_search_cv(num_splits):
        clf = mock.MagicMock()
        clf.best_index_ = 1
        clf.best_params_ = "mocked_best_params"
        clf.cv_results_ = {
            **{
                f"split{split_id}_train_score": [
                    f"train_sc_split_{split_id}_param_0",
                    f"train_sc_split_{split_id}_param_1",
                    f"train_sc_split_{split_id}_param_2",
                ]
                for split_id in range(num_splits + 1)
            },
            **{
                f"split{split_id}_test_score": [
                    f"test_sc_split_{split_id}_param_0",
                    f"test_sc_split_{split_id}_param_1",
                    f"test_sc_split_{split_id}_param_2",
                ]
                for split_id in range(num_splits + 1)
            },
            "mean_fit_time": ["fit_t0", "fit_t1", "fit_t2"],
            "mean_score_time": ["score_t0", "score_t1", "score_t2"],
        }

        return clf

    return mock_one_grid_search_cv


@pytest.fixture(scope="module")
def cv():
    return Stratified10FoldCV(num_splits=2, num_trials=3)


class TestStratified10FoldCV:
    @mock.patch(
        "mre.experimentation.Stratified10FoldCV._display_model_results_at_trial",
        return_value=None,
    )
    @mock.patch(
        "mre.experimentation.Stratified10FoldCV._collect_model_results_at_trial",
        return_value=None,
    )
    @mock.patch(
        "mre.experimentation.Stratified10FoldCV._cross_validate",
        return_value="clf",
    )
    @mock.patch(
        "mre.experimentation.Stratified10FoldCV._setup",
        return_value=("outer_cv"),
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
                mock.call(dataset, arch, "outer_cv")
                for arch in architectures
                for _ in range(cv.num_trials)
            ]
        )
        mock_collect_model_results_at_trial.assert_has_calls(
            [
                mock.call(
                    "clf",
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
                mock.call("clf", arch, 8)  # length of "ml_model"
                for arch in architectures
                for ii in range(cv.num_trials)
            ]
        )

    def test_collect_model_results_at_trial(self, cv, mock_grid_search_cv):
        clf = mock_grid_search_cv(cv.num_splits)
        results_list = [
            {
                "architecture": "ml_model",
                "train_score": "train_sc0",
                "test_score": "test_sc0",
                "fit_time": "fit_t0",
                "score_time": "score_t0",
                "split_id": 1,
                "trial_id": 2,
                "best_params_": "mocked_best_params",
                "cv_results_": "mocked_cv_results_0",
            }
        ]
        trial_id = 3
        architecture = Architecture(
            "ml_model",
            estimator="some_estimator",
            param_grid={"param1": ["val1", "val2"]},
        )

        cv._collect_model_results_at_trial(clf, results_list, trial_id, architecture)
        expected = [
            {
                "architecture": "ml_model",
                "train_score": "train_sc0",
                "test_score": "test_sc0",
                "fit_time": "fit_t0",
                "score_time": "score_t0",
                "split_id": 1,
                "trial_id": 2,
                "best_params_": "mocked_best_params",
                "cv_results_": "mocked_cv_results_0",
            },
            {  # appended to the existing items
                "architecture": "ml_model",
                "train_score": "train_sc_split_0_param_1",
                "test_score": "test_sc_split_0_param_1",
                "fit_time": "fit_t1",
                "score_time": "score_t1",
                "split_id": 0,
                "trial_id": 3,
                "best_params_": "mocked_best_params",
                "cv_results_": clf.cv_results_,
            },
            {
                "architecture": "ml_model",
                "train_score": "train_sc_split_1_param_1",
                "test_score": "test_sc_split_1_param_1",
                "fit_time": "fit_t1",
                "score_time": "score_t1",
                "split_id": 1,
                "trial_id": 3,
                "best_params_": "mocked_best_params",
                "cv_results_": clf.cv_results_,
            },
        ]

        assert results_list == expected
