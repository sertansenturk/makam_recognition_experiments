import csv
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import mlflow
from tomato import __version__ as tomato_version
from tomato.audio.predominantmelody import PredominantMelody \
    as PredominantMelodyExtractor
from tqdm import tqdm

from ..config import config
from ..mlflow_common import get_run_by_name
from .audio import Audio
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
    EXPERIMENT_NAME = cfg.get("mlflow", "data_processing_experiment_name")
    RUN_NAME = cfg.get("mlflow", "predominant_melody_makam_run_name")
    IMPLEMENTATION_SOURCE = (
        f"https://github.com/sertansenturk/tomato/tree/{tomato_version}")
    FILE_EXTENSION = ".pitch"

    def __init__(self):
        """instantiates an Audio object
        """
        super().__init__()
        self.extractor = PredominantMelodyExtractor()

    @classmethod
    def from_mlflow(cls):
        """return predominant melody file paths from the relevant mlflow run
        Returns
        -------
        List[Path]
            path of the predominant melody files logged in mlflow as artifacts
        Raises
        ------
        ValueError
            if the predominant melody features have not been logged in mlflow
        """
        mlflow_run = get_run_by_name(cls.EXPERIMENT_NAME, cls.RUN_NAME)
        if mlflow_run is None:
            raise ValueError("Predominant melodies are not logged in mlflow")

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(mlflow_run.run_id)
        artifact_names = [ff.path for ff in artifacts
                          if ff.path.endswith(cls.FILE_EXTENSION)]

        artifact_paths = [client.download_artifacts(mlflow_run.run_id, an)
                          for an in artifact_names]

        logger.info("Returning the paths of %d predominant melody features.",
                    len(artifact_paths))

        return artifact_paths

    def extract(self, audio_paths: List[str]):
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
            output = self.extractor.extract(path)
            pitch = output["pitch"]

            tmp_file = Path(self._tmp_dir_path(),
                            f"{Path(path).stem}{self.FILE_EXTENSION}")

            with open(tmp_file, "w") as f:
                wr = csv.writer(f)
                wr.writerows(pitch)

            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PredominantMelodyMakam settings and
            mlflow run id where audio recordings are stored
        """
        tags = self.extractor.get_settings()
        tags["source_run_id"] = get_run_by_name(
            Audio.EXPERIMENT_NAME, Audio.RUN_NAME)["run_id"]

        return tags
