import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from tomato import __version__ as tomato_version
from tomato.audio.pitchdistribution import PitchDistribution
from tqdm import tqdm

from ..config import config
from ..mlflow_common import get_run_by_name
from .data import Data

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class PitchClassDistribution(Data):
    """class to extract pitch class distribution (PCD) from the predominant
    melody of each audio recording
    """
    RUN_NAME = cfg.get("mlflow", "pitch_class_distribution_run_name")
    IMPLEMENTATION_SOURCE = (
        f"https://github.com/sertansenturk/tomato/tree/{tomato_version}")
    KERNEL_WIDTH = cfg.getfloat("pitch_class_distribution", "kernel_width")
    NORM_TYPE = cfg.get("pitch_class_distribution", "norm_type")
    STEP_SIZE = cfg.getfloat("pitch_class_distribution", "step_size")
    FILE_EXTENSION = ".json"

    def transform(self, norm_melody_paths: List[str]):
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
            distribution = PitchDistribution.from_cent_pitch(
                melody,  # pitch values sliced internally
                kernel_width=self.KERNEL_WIDTH,
                norm_type=self.NORM_TYPE,
                step_size=self.STEP_SIZE)
            distribution.to_pcd()

            tmp_file = Path(self._tmp_dir_path(),
                            f"{Path(path).stem}{self.FILE_EXTENSION}")

            distribution.to_json(tmp_file)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PCD extractor settings and mlflow run id
            where predominant melody features are stored
        """
        tags = {
            "kernel_width": self.KERNEL_WIDTH,
            "norm_type": self.NORM_TYPE,
            "step_size": self.STEP_SIZE}
        tags["source_run_id"] = get_run_by_name(
            self.EXPERIMENT_NAME,
            cfg.get("mlflow", "predominant_melody_normalize_run_name")
            )["run_id"]

        return tags
