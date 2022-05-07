from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Architecture:
    name: str  # human-readable name
    estimator: field(default_factory=None)  # e.g. a sklearn classifier
    param_grid: Dict  # hyperparameters to grid search
