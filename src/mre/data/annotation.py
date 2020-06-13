import logging

import pandas as pd
from .. config import config

logger = logging.Logger(  # pylint: disable-msg=C0103
    __name__, level=logging.INFO)


class Annotation:
    """class to read and process makam recognition annotations"""
    def __init__(self):
        """instantiates an Annotation object
        """
        self.annotation_url = config.read()["dataset"]["annotation_file"]
        
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

    def _get_num_recs_per_makam(self) -> int:
        """[summary]

        Returns
        -------
        int
            [description]
        """
        num_recordings_per_makam = list(
            self.annotations.makam.value_counts())

        # balanced dataset; all classes have the same number of instances
        if num_recordings_per_makam == num_recordings_per_makam[0]:
            num_recordings_per_makam = num_recordings_per_makam[0]
        return num_recordings_per_makam

    def _validate(self):
        """validates the annotations
        
        Read the expected values mre config and verifies that:
            1) number of recordings
            2) number of makams
            3) number of recordings per makam
        have the expected values.
        """
        cfg = config.read()

        num_recordings = len(self.annotations.mbid)
        expected_num_recs = cfg.getint("dataset", "num_recordings")
        if num_recordings != expected_num_recs:
            raise ValueError(
                f"There are {num_recordings} recordings. "
                f"Expected: {expected_num_recs}.")

        unique_num_recordings = self.annotations.mbid.nunique()
        if unique_num_recordings != num_recordings:
            raise ValueError("MusicBrainz ID (MBIDs) are not unique.")
        logger.info(f"{num_recordings} annotated recordings.")

        num_recordings_per_makam = self._get_num_recs_per_makam()
        expected_num_recs_per_makam = cfg.getint(
            "dataset", "num_recordings_per_makam")
        if num_recordings_per_makam == expected_num_recs_per_makam:
            raise ValueError(
                f"{num_recordings_per_makam} number of recordings per makam. "
                f"Expected: {expected_num_recs_per_makam}")
        logger.info(f"{num_recordings_per_makam} recordings per makam.")

        num_makams = len(self.annotations.makam.value_counts())
        expected_num_makams = expected_num_recs / expected_num_recs_per_makam
        if num_makams != expected_num_makams:
            raise ValueError(
                f"There are {num_makams} makams. "
                f"Expected: {expected_num_makams}.")
        logger.info(f"{num_makams} makams.")

    def _parse_mbids(self):
        """Parses mbid field

        The mbid's in the "otmm_makam_recognition_dataset" are stored as 
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

        This method populates all dunya_uid's by merging the mbid and dunya_uid
        fields.
        """
        self.annotations.loc[
            self.annotations["dunya_uid"].isna(), "dunya_uid"] = (
                self.annotations.loc[
                    self.annotations["dunya_uid"].isna(), "mbid"])
