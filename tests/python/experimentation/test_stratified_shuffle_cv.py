from unittest import mock

import numpy as np
import pytest

from mre.experimentation.architecture import Architecture
from mre.experimentation.stratified_shuffle_cv import StratifiedShuffleCV
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
    return StratifiedShuffleCV(num_splits=2, num_trials=3)


class TestStratifiedShuffleCV:
    pass

    @mock.patch(
        "mre.experimentation.StratifiedShuffleCV._display_model_results_at_trial",
        return_value=None,
    )
    @mock.patch(
        "mre.experimentation.StratifiedShuffleCV._collect_model_results_at_trial",
        return_value=None,
    )
    @mock.patch(
        "mre.experimentation.StratifiedShuffleCV._cross_validate",
        return_value="scores",
    )
    @mock.patch(
        "mre.experimentation.StratifiedShuffleCV._setup",
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
        mock_estimator0 = mock_estimator(0)
        mock_estimator1 = mock_estimator(1)

        trial_result = {
            "estimator": mock_estimator1,
            "test_score": "test_sc1",
            "refit_train_score": "refit_train_sc1",
            "test_predictions": "test_preds1",
            "test_labels": "test_labels1",
            "test_predicted_probs": {
                "probs": "predicted_probs1",
                "labels": "clf_classes",
            },
            "refit_time": "refit_t1",
            "best_params_": "best_params_1",
            "split_train_scores": "split_train_scores1",
            "mean_split_train_score": "mean_split_train_score1",
            "std_split_train_score": "std_split_train_score1",
            "split_validation_scores": "split_validation_scores1",
            "mean_split_validation_score": "mean_split_validation_score1",
            "std_split_validation_score": "std_split_validation_score1",
            "confusion_matrix": "confusion_matrix1",
            "roc_auc_score": "roc_auc_score1",
        }
        results_list = [
            {
                "estimator": mock_estimator0,
                "test_score": "test_sc0",
                "refit_train_score": "train_sc0",
                "test_predictions": "test_preds0",
                "test_labels": "test_labels0",
                "test_predicted_probs": {
                    "probs": "predicted_probs0",
                    "labels": "clf_classes",
                },
                "refit_time": "refit_t0",
                "best_params_": "best_params_0",
                "split_train_scores": "split_train_scores0",
                "mean_split_train_score": "mean_split_train_score0",
                "std_split_train_score": "std_split_train_score0",
                "split_validation_scores": "split_validation_scores0",
                "mean_split_validation_score": "mean_split_validation_score0",
                "std_split_validation_score": "std_split_validation_score0",
                "confusion_matrix": "confusion_matrix0",
                "roc_auc_score": "roc_auc_score0",
                "architecture": "ml_model",
                "trial_id": 2,
                "cv_results_": "mocked_cv_results_0",
            }
        ]
        trial_id = 3
        architecture = Architecture(
            "ml_model",
            estimator="some_estimator",
            param_grid={"param1": ["val1", "val2"]},
        )

        cv._collect_model_results_at_trial(
            trial_result, results_list, trial_id, architecture
        )
        expected_results_list = [
            {
                "estimator": mock_estimator0,
                "test_score": "test_sc0",
                "refit_train_score": "train_sc0",
                "test_predictions": "test_preds0",
                "test_labels": "test_labels0",
                "test_predicted_probs": {
                    "probs": "predicted_probs0",
                    "labels": "clf_classes",
                },
                "refit_time": "refit_t0",
                "best_params_": "best_params_0",
                "split_train_scores": "split_train_scores0",
                "mean_split_train_score": "mean_split_train_score0",
                "std_split_train_score": "std_split_train_score0",
                "split_validation_scores": "split_validation_scores0",
                "mean_split_validation_score": "mean_split_validation_score0",
                "std_split_validation_score": "std_split_validation_score0",
                "confusion_matrix": "confusion_matrix0",
                "roc_auc_score": "roc_auc_score0",
                "architecture": "ml_model",
                "trial_id": 2,
                "cv_results_": "mocked_cv_results_0",
            },
            {
                "estimator": mock_estimator1,
                "test_score": "test_sc1",
                "refit_train_score": "refit_train_sc1",
                "test_predictions": "test_preds1",
                "test_labels": "test_labels1",
                "test_predicted_probs": {
                    "probs": "predicted_probs1",
                    "labels": "clf_classes",
                },
                "refit_time": "refit_t1",
                "best_params_": "best_params_1",
                "split_train_scores": "split_train_scores1",
                "mean_split_train_score": "mean_split_train_score1",
                "std_split_train_score": "std_split_train_score1",
                "split_validation_scores": "split_validation_scores1",
                "mean_split_validation_score": "mean_split_validation_score1",
                "std_split_validation_score": "std_split_validation_score1",
                "confusion_matrix": "confusion_matrix1",
                "roc_auc_score": "roc_auc_score1",
                "architecture": "ml_model",
                "trial_id": 3,
                "cv_results_": "mocked_cv_results_1",
            },
        ]
        print(results_list)
        print(expected_results_list)

        assert results_list == expected_results_list
