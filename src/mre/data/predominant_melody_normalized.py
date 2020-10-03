import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tomato import __version__ as tomato_version
from tomato.converter import Converter
from tqdm import tqdm

from ..config import config
from ..mlflow_common import get_run_by_name
from .data import Data

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class PredominantMelodyNormalized(Data):
    """normalizes the predominant melody with respect to the tonic frequency
    such that the tonic frequency is 0 cents
    """
    RUN_NAME = cfg.get("mlflow", "predominant_melody_normalize_run_name")
    IMPLEMENTATION_SOURCE = (
        f"https://github.com/sertansenturk/tomato/tree/{tomato_version}")

    def transform(self,
                  pitch_paths: List[str],
                  tonic_frequencies: pd.Series):
        """reads predominant melody features from the given paths, converts
        the features into cent scale by normalizing with respect to the tonic
        frequency and saves them into a temporary folder

        Parameters
        ----------
        pitch_paths : List[str]
            paths of the predominant melody feature to convert
        tonic_frequencies: pandas.Series
            tonic frequency corresponding to each predominant melody file

        Raises
        ------
        ValueError
            if pitch_paths is empty

        ValueError
            if tonic_frequencies is empty

        ValueError
            if pitch_paths has a duplicate path

        ValueError
            if tonic_frequencies has a duplicate index

        ValueError
            if pitch_paths filenames and tonic_frequencies
            indices do not match
        """
        if not pitch_paths:
            raise ValueError("pitch_paths is empty!")

        if tonic_frequencies.empty:
            raise ValueError("tonic_frequencies is empty!")

        mel_mbids = [Path(pp).stem for pp in pitch_paths]
        tonic_mbids = list(tonic_frequencies.index)
        if len(set(mel_mbids)) != len(mel_mbids):
            raise ValueError("pitch_paths has a duplicate path!")
        if len(set(tonic_mbids)) != len(tonic_mbids):
            raise ValueError("tonic_mbids has a duplicate index!")

        if set(mel_mbids) != set(tonic_mbids):
            raise ValueError("MBIDs of pitch_paths and "
                             "tonic_mbids do not match!")

        if self.tmp_dir is not None:
            self._cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()
        for path, freq in tqdm(zip(pitch_paths, tonic_frequencies),
                               total=len(tonic_frequencies)):
            pitch = np.load(path)
            pitch[:, 1] = Converter.hz_to_cent(pitch[:, 1], freq)

            tmp_file = Path(self._tmp_dir_path(),
                            Path(path).stem + self.FILE_EXTENSION)

            np.save(tmp_file, pitch)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PredominantMelodyMakam settings and
            mlflow run id where audio recordings are stored
        """
        pitch_source_run_id = get_run_by_name(
            self.EXPERIMENT_NAME,
            cfg.get("mlflow", "predominant_melody_makam_run_name")
            )["run_id"]
        tonic_source_run_id = get_run_by_name(
            self.EXPERIMENT_NAME,
            cfg.get("mlflow", "annotation_run_name")
            )["run_id"]
        tags = {
            "predominant_melody_source_run_id": pitch_source_run_id,
            "tonic_source_run_id": tonic_source_run_id}

        return tags
