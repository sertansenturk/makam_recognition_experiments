from abc import abstractclassmethod

# from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd

# import numpy as np
# from sklearn.model_selection import (
#     cross_validate,
#     GridSearchCV,
#     RepeatedStratifiedKFold,
#     StratifiedKFold,
#     StratifiedShuffleSplit,
# )

from mre.experimentation.architecture import Architecture
from mre.data.dataset import Dataset


# VERBOSE = 0
# N_JOBS = -1


# class SupportedCrossValidationTechniques(Enum):
#     stratified_random_split_cv = auto()
#     stratified_shuffle_split_cv = auto()
#     stratified_10_fold_cv = auto()
#     nested_stratified_10_fold_cv = auto()


@dataclass
class CrossValidator:
    """dataclass to run cross validation"""

    num_splits: int = 10
    num_trials: int = 10
    results: pd.DataFrame = field(default=None, init=False)

    @abstractclassmethod
    def run(self, dataset: Dataset, architectures: List[Architecture]):
        pass

    @abstractclassmethod
    def _setup(self, random_state):
        pass

    @abstractclassmethod
    def _cross_validate(self, dataset, inner_cv, outer_cv, architecture: Architecture):
        pass

    @abstractclassmethod
    def _summarize_model_at_trial(self, scores, architecture: Architecture, max_architecture_name_len: int):
        pass

    @abstractclassmethod
    def _collect_nested_scores(
        self, score, nested_scores: Dict, trial_id: int, architecture: Architecture
    ):
        pass

    @classmethod
    def _max_architecture_name_len(cls, architectures: List[Architecture]) -> int:
        return max(len(arch.name) for arch in architectures)
