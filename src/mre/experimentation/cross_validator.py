from abc import abstractmethod

# from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from mre.experimentation.architecture import Architecture
from mre.data.dataset import Dataset


@dataclass
class CrossValidator:
    """dataclass to run cross validation"""

    num_splits: int = 10
    num_trials: int = 10
    results: pd.DataFrame = field(default=None, init=False)

    @abstractmethod
    def run(self, dataset: Dataset, architectures: List[Architecture]):
        pass

    @abstractmethod
    def _setup(self, random_state):
        pass

    @abstractmethod
    def _cross_validate(self, dataset, architecture: Architecture, *cv_splits):
        pass

    @abstractmethod
    def _collect_model_results_at_trial(
        self, scores, results_list: Dict, trial_id: int, architecture: Architecture
    ):
        pass

    @abstractmethod
    def _display_model_results_at_trial(
        self, scores, architecture: Architecture, max_architecture_name_len: int
    ):
        pass

    @classmethod
    def _max_architecture_name_len(cls, architectures: List[Architecture]) -> int:
        return max(len(arch.name) for arch in architectures)

    def box_plot_best_models_by_trial(self):
        ax = sns.boxplot(
            x="trial_id",
            y="test_score",
            hue="architecture",
            data=self.results,
            palette="Set3",
        )
        plt.grid()

        return ax

    def box_plot_best_models(self):
        ax = sns.boxplot(
            x="architecture",
            y="mean",
            data=(
                self.results.groupby(["trial_id", "architecture"])["test_score"]
                .agg(["mean", "std"])
                .reset_index()
            ),
            palette="Set3",
        )
        plt.grid()

        return ax
