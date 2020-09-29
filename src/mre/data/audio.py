import logging
import tempfile
from pathlib import Path
from typing import Dict

import pandas as pd
from compmusic import dunya
from tqdm import tqdm

from ..config import config
from .data import Data

logger = logging.Logger(__name__)  # pylint: disable-msg=C0103
logger.setLevel(logging.INFO)

cfg = config.read()


class Audio(Data):
    """class to download recordings"""
    EXPERIMENT_NAME = cfg.get("mlflow", "data_processing_experiment_name")
    RUN_NAME = cfg.get("mlflow", "audio_run_name")
    AUDIO_SOURCE = "https://dunya.compmusic.upf.edu"
    FILE_EXTENSION = ".mp3"

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
        dunya.set_token(config.read_secrets().get("tokens", "dunya"))
        if self.tmp_dir is not None:
            self._cleanup()
        self.tmp_dir = tempfile.TemporaryDirectory()

        failed_mbids = dict()
        num_recordings = len(annotation_df)
        for idx, anno in tqdm(annotation_df.iterrows(), total=num_recordings):
            tmp_file = Path(self._tmp_dir_path(),
                            anno.mbid + self.FILE_EXTENSION)

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
                    num_recordings - len(failed_mbids),
                    self._tmp_dir_path())
        if failed_mbids:
            logger.warning(
                "Failed to download %d recordings", len(failed_mbids))

        return failed_mbids

    def _mlflow_tags(self) -> Dict:
        """returns tags to log onto a mlflow run

        Returns
        -------
        Dict
            tags to log, namely, source of audio files (Dunya website url)
        """
        tags = {"audio_source": self.AUDIO_SOURCE}

        return tags
