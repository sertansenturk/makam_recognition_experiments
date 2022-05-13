import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from tqdm import tqdm

from mre.config import config
from mre.data.data import Data
from mre.data.tdms_feature import TDMSFeature

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class TimeDelayedMelodySurface(Data):
    """class to extract time-delayed melody surface (TDMS) from the predominant
    melody of each audio recording
    """

    RUN_NAME = cfg.get("mlflow", "time_delayed_melody_surface_run_name")

    KERNEL_WIDTH = cfg.getfloat("time_delayed_melody_surface", "kernel_width")
    STEP_SIZE = cfg.getfloat("time_delayed_melody_surface", "step_size")

    FILE_EXTENSION = ".json"

    def __init__(self):
        """instantiates a TDML object"""
        super().__init__()
        self.transform_func = TDMSFeature.from_hz_pitch

    def transform(  # pylint: disable-msg=W0221
        self,
        melody_paths: List[str],
        tonic_frequencies: pd.Series,
    ):
        """extracts TDMLs from predominant melody of each audio recording by
        normalizing with respect to the tonic frequency and saves the features
        to a temporary folder.

        Parameters
        ----------
        melody_paths : List[str]
            paths of the predominant melody features to extract PCDs
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
            raise ValueError("melody_paths is empty")

        if tonic_frequencies.empty:
            raise ValueError("tonic_frequencies is empty!")

        mel_mbids = [Path(pp).stem for pp in melody_paths]
        tonic_mbids = list(tonic_frequencies.index)
        if len(set(mel_mbids)) != len(mel_mbids):
            raise ValueError("melody_paths has a duplicate path!")
        if len(set(tonic_mbids)) != len(tonic_mbids):
            raise ValueError("tonic_mbids has a duplicate index!")

        if set(mel_mbids) != set(tonic_mbids):
            raise ValueError("MBIDs of melody_paths and tonic_mbids do not match!")

        if self.tmp_dir is not None:
            self._cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()  # pylint: disable-msg=R1732
        for path in tqdm(melody_paths, total=len(melody_paths)):
            melody = np.load(path)
            mbid = Path(path).stem

            tdml = self.transform_func(
                melody,  # pitch values sliced internally
                ref_freq=tonic_frequencies.loc[mbid],
                kernel_width=self.KERNEL_WIDTH,
                step_size=self.STEP_SIZE,
            )

            tmp_file = Path(self._tmp_dir_path(), mbid + self.FILE_EXTENSION)
            tdml.to_json(tmp_file)
            logger.debug("Saved to %s.", tmp_file)

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, PCD extractor settings
        """
        # return {
        #     "kernel_width": self.KERNEL_WIDTH,
        #     "norm_type": self.NORM_TYPE,
        #     "step_size": self.STEP_SIZE,
        # }
