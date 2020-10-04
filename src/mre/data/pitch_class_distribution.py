import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from tomato.audio.pitchdistribution import PitchDistribution
from tqdm import tqdm

from ..config import config
from .data import Data

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class PitchClassDistribution(Data):
    """class to extract pitch class distribution (PCD) from the predominant
    melody of each audio recording
    """
    RUN_NAME = cfg.get("mlflow", "pitch_class_distribution_run_name")

    KERNEL_WIDTH = cfg.getfloat("pitch_class_distribution", "kernel_width")
    NORM_TYPE = cfg.get("pitch_class_distribution", "norm_type")
    STEP_SIZE = cfg.getfloat("pitch_class_distribution", "step_size")

    FILE_EXTENSION = ".json"

    def __init__(self):
        """instantiates a PredominantMelodyMakam object
        """
        super().__init__()
        self.transform_func = PitchDistribution.from_cent_pitch

    def transform(self,  # pylint: disable-msg=W0221
                  norm_melody_paths: List[str]):
        """extracts PCDs from the tonic normalized predominant melody of each
        audio recording and saves the features to a temporary folder.

        IMPORTANT: The method assumes the predominant melody is already
        converted to cent scale by normalizing with respect to the tonic
        frequency of the audio recording.

        Parameters
        ----------
        norm_melody_paths : List[str]
            paths of the predominant melody features to extract PCDs

        Raises
        ------
        ValueError
            if norm_melody_paths is empty
        """
        if not norm_melody_paths:
            raise ValueError("norm_melody_paths is empty")

        if self.tmp_dir is not None:
            self._cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()
        for path in tqdm(norm_melody_paths,
                         total=len(norm_melody_paths)):
            melody = np.load(path)

            distribution: PitchDistribution = self.transform_func(
                melody,  # pitch values sliced internally
                kernel_width=self.KERNEL_WIDTH,
                norm_type=self.NORM_TYPE,
                step_size=self.STEP_SIZE)
            distribution.to_pcd()

            tmp_file = Path(self._tmp_dir_path(),
                            Path(path).stem + self.FILE_EXTENSION)
            distribution.to_json(tmp_file)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PCD extractor settings
        """
        return {
            "kernel_width": self.KERNEL_WIDTH,
            "norm_type": self.NORM_TYPE,
            "step_size": self.STEP_SIZE}
