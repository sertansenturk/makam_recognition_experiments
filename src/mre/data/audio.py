import logging
import os
import tempfile
from pathlib import Path

import pandas as pd
from compmusic import dunya
from tqdm import tqdm

from ..config import config

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()
dunya.set_token(config.read_secrets().get("tokens", "dunya"))


class Audio():
    """class to download recordings"""
    EXPERIMENT_NAME = cfg.get("mlflow", "data_processing_experiment_name")
    RUN_NAME = cfg.get("mlflow", "audio_run_name")
    AUDIO_SOURCE = "https://dunya.compmusic.upf.edu"

    def __init__(self):
        """instantiates an Audio object
        """
        self.tmp_dir = None

    def from_mlflow(self):
        # mre.mlflow.get_mlflow_run
        pass

    def from_dunya(self, annotation_df: pd.DataFrame):
        """Downloads the audio recordings specified in the annotations from
        Dunya

        Parameters
        ----------
        annotation_df : pd.DataFrame
            annotations of the audio recordings

        Raises
        ------
        dunya.conn.HTTPError
            if an HTTP error other than "404 - Not Found" is encountered
        """
        self.tmp_dir = tempfile.TemporaryDirectory()

        failed_mbids = dict()
        num_recordings = len(annotation_df)
        for idx, anno in tqdm(annotation_df.iterrows(), total=num_recordings):
            tmp_file = os.path.join(self._tmp_dir_path(), f"{anno.mbid}.mp3")

            try:
                mp3_content = dunya.docserver.get_mp3(anno.dunya_uid)
                with open(tmp_file, "wb") as f:
                    f.write(mp3_content)
                logger.debug("%d/%d: Saved to %s.",
                             idx, num_recordings, tmp_file)
            except dunya.conn.HTTPError as err:
                if "404 Client Error: Not Found for url:" in str(err):
                    logger.error("%d/%d: %s. Skipping...",
                                 idx, num_recordings, str(err))
                    failed_mbids[anno.mbid] = {
                        "type": "dunya.conn.HTTPError",
                        "reason": "404_url_not_found",
                        "message": str(err)
                    }
                else:
                    self._cleanup()
                    raise err
        logger.info("Downloaded %d recordings to %s",
                    num_recordings - len(failed_mbids), self._tmp_dir_path())
        logger.warning("Failed to download %d recordings", len(failed_mbids))

        return failed_mbids

    def _tmp_dir_path(self) -> Path:
        """returns the path of the temporary directory, where the audio files
        are downloaded

        Returns
        -------
        Path
            path of the temporary directory
        """
        return Path(self.tmp_dir.name)

    def _cleanup(self):
        """deletes temporary directory, where the audio files are downloaded
        """
        self.tmp_dir.cleanup()