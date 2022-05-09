import logging

from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    cross_validate,
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
class NestedStratified10FoldCV(CrossValidator):
    def run(self, dataset: Dataset, architectures: List[Architecture]) -> pd.DataFrame:
        max_arch_name_len = self._max_architecture_name_len(architectures)

        results_list = []
        for ii in range(self.num_trials):
            logger.info("Trial %d", ii)

            inner_cv, outer_cv = self._setup(ii)
            for arch in architectures:
                scores = self._cross_validate(dataset, arch, inner_cv, outer_cv)
                self._collect_model_results_at_trial(scores, results_list, ii, arch)
                self._display_model_results_at_trial(scores, arch, max_arch_name_len)

        self.results = pd.DataFrame(results_list)

    def _setup(self, random_state: int):
        inner_cv = StratifiedKFold(
            n_splits=self.num_splits, shuffle=True, random_state=random_state
        )
        outer_cv = StratifiedKFold(
            n_splits=self.num_splits, shuffle=True, random_state=random_state
        )

        return inner_cv, outer_cv

    def _cross_validate(
        self, dataset: Dataset, architecture: Architecture, inner_cv, outer_cv
    ) -> Dict:
        clf = GridSearchCV(
            estimator=architecture.estimator,
            param_grid=architecture.param_grid,
            cv=inner_cv,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
        )

        return cross_validate(
            clf,
            X=dataset.X,
            y=dataset.y,
            cv=outer_cv,
            verbose=VERBOSE,
            n_jobs=N_JOBS,
            return_estimator=True,
            return_train_score=True,
        )

    def _collect_model_results_at_trial(
        self,
        scores: Dict,
        results_list: List,
        trial_id: int,
        architecture: Architecture,
    ):
        for ns in range(self.num_splits):
            results_list.append(
                {
                    "architecture": architecture.name,
                    "train_score": scores["train_score"][ns],
                    "test_score": scores["test_score"][ns],
                    "fit_time": scores["fit_time"][ns],
                    "score_time": scores["score_time"][ns],
                    "split_id": ns,
                    "trial_id": trial_id,
                    "best_params_": scores["estimator"][ns].best_params_,
                    "cv_results_": scores["estimator"][ns].cv_results_,
                }
            )

    def _display_model_results_at_trial(
        self, scores: Dict, architecture: Architecture, max_architecture_name_len: int
    ):
        best_params_str = [str(est.best_params_) for est in scores["estimator"]]
        most_common_best_params = max(best_params_str, key=best_params_str.count)

        print(
            f"   {architecture.name:<{max_architecture_name_len}}, "
            f'Test acc: {np.mean(scores["test_score"]):.2f}∓'
            f'{np.std(scores["test_score"]):.2f}, '
            f'Train acc: {np.mean(scores["train_score"]):.2f}∓'
            f'{np.std(scores["train_score"]):.2f}, '
            f'Max fit time: {max(scores["fit_time"]):.1f} sec, '
            f"Best Params: {most_common_best_params}"
        )
