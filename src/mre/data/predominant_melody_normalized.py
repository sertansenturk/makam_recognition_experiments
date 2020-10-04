import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tomato.converter import Converter
from tqdm import tqdm

from ..config import config
from .data import Data

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class PredominantMelodyNormalized(Data):
    """normalizes the predominant melody with respect to the tonic frequency
    such that the tonic frequency is 0 cents
    """
    RUN_NAME = cfg.get("mlflow", "predominant_melody_normalize_run_name")

    MIN_FREQ = cfg.getfloat("tonic_normalization", "min_freq")

    def __init__(self):
        """instantiates a PredominantMelodyMakam object
        """
        super().__init__()
        self.transform_func = Converter.hz_to_cent

    def transform(self,  # pylint: disable-msg=W0221
                  melody_paths: List[str],
                  tonic_frequencies: pd.Series):
        """reads predominant melody features from the given paths, converts
        the features into cent scale by normalizing with respect to the tonic
        frequency and saves them into a temporary folder

        Parameters
        ----------
        melody_paths : List[str]
            paths of the predominant melody feature to convert
        tonic_frequencies: pandas.Series
            tonic frequency corresponding to each predominant melody file

        Raises
        ------
        ValueError
            if melody_paths is empty

        ValueError
            if tonic_frequencies is empty

        ValueError
            if melody_paths has a duplicate path

        ValueError
            if tonic_frequencies has a duplicate index

        ValueError
            if melody_paths filenames and tonic_frequencies
            indices do not match
        """
        if not melody_paths:
            raise ValueError("melody_paths is empty!")

        if tonic_frequencies.empty:
            raise ValueError("tonic_frequencies is empty!")

        mel_mbids = [Path(pp).stem for pp in melody_paths]
        tonic_mbids = list(tonic_frequencies.index)
        if len(set(mel_mbids)) != len(mel_mbids):
            raise ValueError("melody_paths has a duplicate path!")
        if len(set(tonic_mbids)) != len(tonic_mbids):
            raise ValueError("tonic_mbids has a duplicate index!")

        if set(mel_mbids) != set(tonic_mbids):
            raise ValueError("MBIDs of melody_paths and "
                             "tonic_mbids do not match!")

        if self.tmp_dir is not None:
            self._cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()
        for path, freq in tqdm(zip(melody_paths, tonic_frequencies),
                               total=len(tonic_frequencies)):
            melody: np.array = np.load(path)

            melody[:, 1]: np.array = self.transform_func(
                melody[:, 1], freq, min_freq=self.MIN_FREQ)

            tmp_file = Path(self._tmp_dir_path(),
                            Path(path).stem + self.FILE_EXTENSION)
            np.save(tmp_file, melody)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PredominantMelodyNormalized settings
        """
        return {"min_freq": self.MIN_FREQ}
