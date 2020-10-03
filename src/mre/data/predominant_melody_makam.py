import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np

from tomato import __version__ as tomato_version
from tomato.audio.predominantmelody import \
    PredominantMelody as PredominantMelodyTransformer
from tqdm import tqdm

from ..config import config
from ..mlflow_common import get_run_by_name
from .data import Data

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
    IMPLEMENTATION_SOURCE = (
        f"https://github.com/sertansenturk/tomato/tree/{tomato_version}")

    def __init__(self):
        """instantiates a PredominantMelodyExtractor object
        """
        super().__init__()
        self.transformer = PredominantMelodyTransformer()

    def transform(self, audio_paths: List[str]):
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
            output = self.transformer.extract(path)
            melody = output["pitch"]

            tmp_file = Path(self._tmp_dir_path(),
                            Path(path).stem + self.FILE_EXTENSION)

            np.save(tmp_file, melody)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PredominantMelodyMakam settings and
            mlflow run id where audio recordings are stored
        """
        tags = self.transformer.get_settings()
        tags["source_run_id"] = get_run_by_name(
            self.EXPERIMENT_NAME,
            cfg.get("mlflow", "audio_run_name"))["run_id"]

        return tags
