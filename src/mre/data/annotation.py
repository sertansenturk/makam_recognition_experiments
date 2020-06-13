import logging

import pandas as pd
from .. config import config

logger = logging.Logger(  # pylint: disable-msg=C0103
    __name__, level=logging.INFO)


class Annotation:
    """class to read and parse makam recognition annotations"""
    def __init__(self):
        """instantiates an Annotation object
        """
        cfg = config.read()
        self.annotation_url = cfg["dataset"]["annotation_file"]
        self.num_recordings = cfg.getint("dataset", "num_recordings")
        self.num_recordings_per_makam = cfg.getint(
            "dataset", "num_recordings_per_makam")
        logger.info(f"Reading annotations from: {self.annotation_url}")

        self.annotations = self._read_from_github()
        self._validate()
        self._parse_mbids()
        self._patch_uids()

    def head(self) -> pd.DataFrame:
        """returns the first five annotations

        Returns
        -------
        pd.DataFrame
            first five annotations
        """
        return self.annotations.head()

    def _read_from_github(self):
        """reads the annotation file from github and validates
        """
        return pd.read_json(self.annotation_url)
        
    def _validate(self):
        """validates the annotations
        
        Verifies that the:
            1) number of recordings
            2) number of makams
            3) number of recordings per makam
        have the expected values.
        """
        if len(self.annotations.mbid) != self.num_recordings:
            raise ValueError(
                f"There are {len(self.annotations.mbid)} recordings. "
                f"Expected: {self.num_recordings}")
        if len(self.annotations.mbid.unique()) != self.num_recordings:
            raise ValueError("MusicBrainz ID (MBIDs) are not unique")
        logger.info(f"{self.num_recordings} annotated recordings.")

        num_makams = self.num_recordings / self.num_recordings_per_makam
        makam_counts = self.annotations.makam.value_counts()
        if len(makam_counts) != num_makams:
            raise ValueError(
                f"There are less than {num_makams} makams")
        logger.info(f"{num_makams} makams.")

        num_recs_per_makam = makam_counts.unique()
        is_rec_makam_valid = (
            (len(num_recs_per_makam) == 1) and
            (num_recs_per_makam[0] == self.num_recordings_per_makam))
        if not is_rec_makam_valid:
            raise ValueError("The number of recorings per makam should have "
                             f"been {self.num_recordings_per_makam}")
        logger.info(f"{self.num_recordings_per_makam} recordings per makam.")

    def _parse_mbids(self):
        """
        the mbid's in the "otmm_makam_recognition_dataset" are stored as 
        URL's pointing to the MusicBrainz website.

        This method moves the URL to a new mb_url field and trims mbid's
        """
        self.annotations["mb_url"] = self.annotations["mbid"]
        self.annotations["mbid"] = self.annotations["mbid"].str.split(
            pat = "/").apply(lambda a: a[-1])
    
    def _patch_uids(self):
        """Patches the mbid and dunya_uid's
        
        Some recordings in "CompMusic makam music corpus" may not point to the
        master MBID in MusicBrainz due to database merges. 
        
        "otmm_makam_recognition_dataset" keeps track of such discrepancies in
        in the dunya_uid field. This field is not populated if the dunya_uid
        & mbid are the same.

        This method merges the mbid and dunya_uid to simplify sending 
        requests to Dunya API later
        """
        self.annotations.loc[
            self.annotations["dunya_uid"].isna(), "dunya_uid"] = (
                self.annotations.loc[
                    self.annotations["dunya_uid"].isna(), "mbid"])
