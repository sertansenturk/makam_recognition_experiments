import logging

from functools import reduce
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
)

from mre.config import config
from mre.experimentation.architecture import Architecture
from mre.experimentation.cross_validator import CrossValidator
from mre.data.dataset import Dataset

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()

VERBOSE = 0
N_JOBS = -1
CONFUSION_MATRIX_COL_LABEL = "Predicted Label"
CONFUSION_MATRIX_ROW_LABEL = "True Label"


@dataclass
class StratifiedShuffleCV(CrossValidator):
    def run(self, dataset: Dataset, architectures: List[Architecture]):
        max_arch_name_len = self._max_architecture_name_len(architectures)

        results_list = []
        for ii in range(self.num_trials):
            print(f"Trial {ii}")

            inner_cv, outer_cv = self._setup(ii)
            for arch in architectures:
                trial_result = self._cross_validate(dataset, arch, inner_cv, outer_cv)
                self._collect_model_results_at_trial(
                    trial_result, results_list, ii, arch
                )
                self._display_model_results_at_trial(
                    trial_result, arch, max_arch_name_len
                )

        self.results = pd.DataFrame(results_list)

    def _setup(self, random_state: int):
        # 90 (train + validation), 10 test
        # one split; use num_trials instead to repeat
        outer_cv = StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=random_state
        )

        # 80 training, 10 validation
        inner_cv = StratifiedShuffleSplit(
            n_splits=self.num_splits, test_size=1 / 9, random_state=random_state
        )

        return inner_cv, outer_cv

    def _cross_validate(
        self, dataset: Dataset, architecture: Architecture, inner_cv, outer_cv
    ) -> Dict:
        inner_idx, test_idx = next(outer_cv.split(dataset.X, dataset.y))

        num_test_samples_per_class = pd.Series(dataset.y[test_idx]).value_counts()
        assert (num_test_samples_per_class == num_test_samples_per_class[0]).all()
        logging.debug("test_len: %d", len(test_idx))

        clf = GridSearchCV(
            estimator=architecture.estimator,
            param_grid=architecture.param_grid,
            cv=inner_cv,
            return_train_score=True,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
        )

        X_inner = dataset.X[inner_idx, :]
        y_inner = dataset.y[inner_idx]
        clf.fit(X_inner, y_inner)

        X_test = dataset.X[test_idx, :]
        y_test = dataset.y[test_idx]

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        trial_result = {
            "estimator": clf,
            "test_score": clf.score(X_test, y_test),
            "refit_train_score": clf.score(X_inner, y_inner),
            "test_predictions": y_pred,
            "test_labels": y_test,
            "test_predicted_probs": {"probs": y_prob, "labels": clf.classes_},
            "refit_time": clf.refit_time_,
            "best_params_": clf.best_params_,
        }

        # aggregate split scores
        train_scores, validation_scores = self._get_train_validation_scores(
            trial_result
        )
        trial_result["split_train_scores"] = train_scores
        trial_result["mean_split_train_score"] = np.mean(train_scores)
        trial_result["std_split_train_score"] = np.std(train_scores)

        trial_result["split_validation_scores"] = validation_scores
        trial_result["mean_split_validation_score"] = np.mean(validation_scores)
        trial_result["std_split_validation_score"] = np.std(validation_scores)

        # evaluate
        trial_result["confusion_matrix"] = self._compute_confusion_matrix(trial_result)
        trial_result["roc_auc_score"] = self._compute_roc_auc_score(trial_result)

        return trial_result

    def _collect_model_results_at_trial(
        self,
        trial_result: Dict,
        results_list: List,
        trial_id: int,
        architecture: Architecture,
    ):
        results_list.append(
            {
                **trial_result,
                "architecture": architecture.name,
                "trial_id": trial_id,
                "cv_results_": trial_result["estimator"].cv_results_,
            }
        )

    def _get_train_validation_scores(self, trial_result):
        best_idx = trial_result["estimator"].best_index_
        train_scores = []
        validation_scores = []
        for ii in range(self.num_splits):
            train_scores.append(
                trial_result["estimator"].cv_results_[f"split{ii}_train_score"][
                    best_idx
                ]
            )
            validation_scores.append(
                trial_result["estimator"].cv_results_[f"split{ii}_test_score"][best_idx]
            )

        return train_scores, validation_scores

    @staticmethod
    def _compute_roc_auc_score(trial_result):
        return roc_auc_score(
            trial_result["test_labels"],
            trial_result["test_predicted_probs"]["probs"],
            multi_class="ovr",  # balance dataset, so doesn't matter
            labels=trial_result["estimator"].classes_,
        )

    @staticmethod
    def _compute_confusion_matrix(trial_result):
        return pd.DataFrame(
            confusion_matrix(
                trial_result["test_labels"],
                trial_result["test_predictions"],
                labels=trial_result["estimator"].classes_,
            ),
            columns=pd.MultiIndex.from_product(
                [[CONFUSION_MATRIX_COL_LABEL], trial_result["estimator"].classes_]
            ),
            index=pd.MultiIndex.from_product(
                [[CONFUSION_MATRIX_ROW_LABEL], trial_result["estimator"].classes_]
            ),
        )

    def _display_model_results_at_trial(
        self,
        trial_result: Dict,
        architecture: Architecture,
        max_architecture_name_len: int,
    ):
        print(
            f"   {architecture.name:<{max_architecture_name_len}}, "
            f'Test acc: {trial_result["test_score"]:.2f}, '
            f'Train acc: {trial_result["mean_split_train_score"]:.2f}∓'
            f'{trial_result["std_split_train_score"]:.2f}, '
            f'Refit time: {trial_result["refit_time"]:.1f} sec, '
            f"Best Params: {trial_result['best_params_']}"
        )

    def get_confusion_matrix(self, architecture=None):
        relevant_results = (
            self.results
            if architecture is None
            else self.results[self.results.architecture == architecture]
        )
        return reduce(np.add, relevant_results.confusion_matrix)

    def plot_confusion_matrix(self, architecture=None):
        conf_mat = (
            self.get_confusion_matrix(architecture)
            .droplevel(level=0, axis=0)
            .droplevel(level=0, axis=1)
            .replace({0: np.nan})
        )
        ax = sns.heatmap(
            conf_mat,
            annot=True,
            # annot_kws={"size": 16},
            xticklabels=3,
            cbar=False,
            linewidths=0.5,
        )
        # ax.tick_params(left=False, bottom=False)

        plt.xlabel(CONFUSION_MATRIX_COL_LABEL)
        plt.ylabel(CONFUSION_MATRIX_ROW_LABEL)

        return ax
