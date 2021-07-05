import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from tomato.audio.predominantmelody import (
    PredominantMelody as PredominantMelodyTransformer,
)
from tqdm import tqdm

from mre.config import config
from mre.data.data import Data

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class PredominantMelodyMakam(Data):
    """class to extract predominant melody from audio recordings using the
    method explained in:

    Atlı, H. S., Uyar, B., Şentürk, S., Bozkurt, B., and Serra, X. (2014).
    Audio feature extraction for exploring Turkish makam music. In Proceedings
    of 3rd International Conference on Audio Technologies for Music and Media
    (ATMM 2014), pages 142–153, Ankara, Turkey.
    """

    RUN_NAME = cfg.get("mlflow", "predominant_melody_makam_run_name")

    def __init__(self):
        """instantiates a PredominantMelodyMakam object"""
        super().__init__()
        self.transformer = PredominantMelodyTransformer()
        self.transform_func = self.transformer.extract
        self.source_run_id = None

    def transform(self, audio_paths: List[str]):  # pylint: disable-msg=W0221
        """extracts predominant melody from each audio recording and
        saves the features to a temporary folder

        Parameters
        ----------
        audio_paths : List[str]
            paths of the audio recordings to extract predominant melody

        Raises
        ------
        ValueError
            if audio_paths is empty
        """
        if not audio_paths:
            raise ValueError("audio_paths is empty")

        if self.tmp_dir is not None:
            self._cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()
        for path in tqdm(audio_paths, total=len(audio_paths)):
            output: Dict = self.transform_func(path)

            melody: np.array = output["pitch"]

            tmp_file = Path(self._tmp_dir_path(), Path(path).stem + self.FILE_EXTENSION)
            np.save(tmp_file, melody)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PredominantMelodyMakam settings
        """
        return self.transformer.get_settings()
