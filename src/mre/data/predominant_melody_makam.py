import csv
import logging
import os
import tempfile
from pathlib import Path
from typing import List

import mlflow
from tomato import __version__ as tomato_version
from tomato.audio.predominantmelody import PredominantMelody
from tqdm import tqdm

from ..config import config
from ..mlflow_common import get_run_by_name

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class PredominantMelodyMakam():
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

    def __init__(self):
        """instantiates an Audio object
        """
        self.tmp_dir = None
        self.extractor = PredominantMelody()

    # def from_mlflow():
    #     pass

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

        self.tmp_dir = tempfile.TemporaryDirectory()
        for path in tqdm(audio_paths, total=len(audio_paths)):
            output = self.extractor.extract(path)
            pitch = output["pitch"]

            tmp_file = os.path.join(self._tmp_dir_path(),
                                    f"{Path(path).stem}.pitch")

            with open(tmp_file, "w") as f:
                wr = csv.writer(f)
                wr.writerows(pitch)

            logger.debug("Saved to %s.", tmp_file)

    def log(self):
        """Logs the predominant melody as artifacts to an mlflow run

        Raises
        ------
        ValueError
            If a run with the same experiment and run name is already logged
            in mlflow
        """
        mlflow_run = get_run_by_name(self.EXPERIMENT_NAME, self.RUN_NAME)
        if mlflow_run is not None:
            raise ValueError(
                "There is already a run for %s:%s. Overwriting is not "
                "permitted. Please delete the run manually if you want "
                "to log the annotations again."
                % (self.RUN_NAME, mlflow_run.run_id))

        mlflow.set_experiment(self.EXPERIMENT_NAME)
        with mlflow.start_run(run_name=self.RUN_NAME):
            mlflow.set_tags(self.extractor.get_settings())
            mlflow.log_artifacts(self._tmp_dir_path())

        self._cleanup()

    def _tmp_dir_path(self) -> Path:
        """returns the path of the temporary directory, where the feature
        files are saved

        Returns
        -------
        Path
            path of the temporary directory
        """
        return Path(self.tmp_dir.name)

    def _cleanup(self):
        """deletes the temporary directory, where the feature files are saved
        """
        self.tmp_dir.cleanup()
