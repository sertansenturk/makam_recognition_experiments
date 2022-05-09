import logging

from typing import List
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)

from mre.experimentation.architecture import Architecture
from mre.experimentation.cross_validator import CrossValidator
from mre.data.dataset import Dataset

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

VERBOSE = 0
N_JOBS = -1


@dataclass
class Stratified10FoldCV(CrossValidator):
    def run(self, dataset: Dataset, architectures: List[Architecture]) -> pd.DataFrame:
        max_arch_name_len = self._max_architecture_name_len(architectures)

        results_list = []
        for ii in range(self.num_trials):
            logger.info("Trial %d", ii)

            # there is a single loop used in 10-fold cross-validation
            # we name it as "outer" to contrast with the nested case
            outer_cv = self._setup(ii)
            for arch in architectures:
                clf = self._cross_validate(dataset, arch, outer_cv)
                self._collect_model_results_at_trial(clf, results_list, ii, arch)
                self._display_model_results_at_trial(clf, arch, max_arch_name_len)

        self.results = pd.DataFrame(results_list)

    def _setup(self, random_state):
        outer_cv = StratifiedKFold(
            n_splits=self.num_splits, shuffle=True, random_state=random_state
        )

        return outer_cv

    def _cross_validate(
        self, dataset: Dataset, architecture: Architecture, outer_cv
    ) -> GridSearchCV:
        clf = GridSearchCV(
            estimator=architecture.estimator,
            param_grid=architecture.param_grid,
            cv=outer_cv,
            return_train_score=True,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
        )
        clf.fit(dataset.X, dataset.y)

        return clf

    def _collect_model_results_at_trial(
        self, clf, results_list, trial_id, architecture
    ):
        for ns in range(self.num_splits):
            results_list.append(
                {
                    "architecture": architecture.name,
                    "train_score": clf.cv_results_[f"split{ns}_train_score"][
                        clf.best_index_
                    ],
                    "test_score": clf.cv_results_[f"split{ns}_test_score"][
                        clf.best_index_
                    ],
                    "fit_time": clf.cv_results_["mean_fit_time"][clf.best_index_],
                    "score_time": clf.cv_results_["mean_score_time"][clf.best_index_],
                    "split_id": ns,
                    "trial_id": trial_id,
                    "best_params_": clf.best_params_,
                    "cv_results_": clf.cv_results_,
                }
            )

    def _display_model_results_at_trial(
        self, clf, architecture, max_architecture_name_len
    ):

        cv_res = clf.cv_results_
        best_idx = clf.best_index_
        print(
            f"  {architecture.name:<{max_architecture_name_len}}, "
            f'Mean/std test acc : {cv_res["mean_test_score"][best_idx]:.2f}∓'
            f'{cv_res["std_test_score"][best_idx]:.2f}, '
            f'Mean/std train acc: {cv_res["mean_train_score"][best_idx]:.2f}∓'
            f'{cv_res["std_train_score"][best_idx]:.2f}, '
            f'Max fit time: {max(cv_res["mean_fit_time"]):.1f} sec, '
            f"Best Params: {clf.best_params_}"
        )
