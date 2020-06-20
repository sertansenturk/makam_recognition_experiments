import logging

import numpy as np
import pandas as pd

from ..config import config

logger = logging.Logger(  # pylint: disable-msg=C0103
    __name__, level=logging.INFO)
cfg = config.read()


class Annotation:
    """class to read and process makam recognition annotations"""
    MUSICBRAINZ_RECORDING_URL = "http://musicbrainz.org/recording/"
    EXPECTED_NUM_RECORDINGS = cfg.getint("dataset", "num_recordings")
    EXPECTED_NUM_RECORDINGS_PER_MAKAM = cfg.getint(
        "dataset", "num_recordings_per_makam")
    EXPECTED_NUM_MAKAMS = (
        EXPECTED_NUM_RECORDINGS / EXPECTED_NUM_RECORDINGS_PER_MAKAM)

    URL = cfg["dataset"]["annotation_file"]

    def __init__(self):
        """instantiates an Annotation object
        """
        logger.info("Reading annotations from: %s", self.URL)

        self.data = self._read_from_github()
        self._validate()

        self._parse_mbid_urls()
        self._patch_dunya_uids()

    def head(self) -> pd.DataFrame:
        """returns the first five annotations

        Returns
        -------
        pd.DataFrame
            first five annotations
        """
        return self.data.head()

    def _read_from_github(self) -> pd.DataFrame:
        """reads the annotation file from github and validates

        Returns
        -------
        pd.DataFrame
            annotations
        """
        return pd.read_json(self.URL)

    def _validate(self):
        """runs all validations
        """
        self._validate_num_recordings()
        self._validate_mbids()
        self._validate_num_recordings_per_makam()
        self._validate_num_makams()

    def _validate_num_recordings(self):
        """validates the number of recordings

        Raises
        ------
        ValueError
            if the number of recordings is not equal to the expected
        """
        num_recordings = len(self.data)
        if num_recordings != self.EXPECTED_NUM_RECORDINGS:
            raise ValueError(
                f"There are {num_recordings} recordings. "
                f"Expected: {self.EXPECTED_NUM_RECORDINGS}.")
        logger.info("%d annotated recordings.", num_recordings)

    def _validate_mbids(self):
        """validates that all of the MBIDs are non-empty and unique

        Raises
        ------
        ValueError
            raises if MBIDs are not unique
        """
        missing_mbid = self.data.mbid.replace("", np.nan).isna()
        if any(missing_mbid):
            raise ValueError("Missing MBIDs.")

        unique_num_recordings = self.data.mbid.nunique()
        num_recordings = len(self.data)
        if unique_num_recordings != num_recordings:
            raise ValueError("MBIDs are not unique.")
        logger.info("%d annotated recordings.", num_recordings)

    def _validate_num_recordings_per_makam(self) -> int:
        """validates that the number of recordings per makam is equal to
        expected

        Raises
        ------
        ValueError
            if the dataset is unbalanced (not supported)
        ValueError
            if the number of recordings per makams is not equal to the
            expected
        """
        num_recordings_per_makam = self.data.makam.value_counts()

        # balanced dataset; all classes have the same number of instances
        if num_recordings_per_makam.nunique() == 1:
            num_recordings_per_makam = num_recordings_per_makam[0]
        else:
            raise ValueError("Unbalanced datasets are not supported.")

        if num_recordings_per_makam != self.EXPECTED_NUM_RECORDINGS_PER_MAKAM:
            raise ValueError(
                f"{num_recordings_per_makam} number of recordings per makam. "
                f"Expected: {self.EXPECTED_NUM_RECORDINGS_PER_MAKAM}")
        logger.info("%d recordings per makam.", num_recordings_per_makam)

    def _validate_num_makams(self):
        """validates that the number of makams is equal to expected

        Raises
        ------
        ValueError
            if the number of makams is not equal to the expected
        """
        num_makams = len(self.data.makam.value_counts())
        if num_makams != self.EXPECTED_NUM_MAKAMS:
            raise ValueError(
                f"There are {num_makams} makams. "
                f"Expected: {self.EXPECTED_NUM_MAKAMS}.")
        logger.info("%d makams.", num_makams)

    def _parse_mbid_urls(self):
        """Parses the urls in the MBID field

        The MBIDs in the "otmm_makam_recognition_dataset" are stored as
        URLs pointing to the MusicBrainz website.

        This method trims MBIDs while moving the URL to a new field called
        mb_url.
        """
        self.data["mb_url"] = self.data["mbid"]

        invalid_url_bool = ~self.data["mb_url"].str.startswith(
            self.MUSICBRAINZ_RECORDING_URL, na=False)
        if any(invalid_url_bool):
            raise ValueError('Invalid urls:\n{}'.format(
                self.data.to_string()))

        self.data["mbid"] = self.data["mb_url"].str.split(pat="/").apply(
            lambda a: a[-1])

    def _patch_dunya_uids(self):
        """Fills missing dunya UIDs with MBIDs

        Some recordings in "CompMusic makam music corpus" may not point to the
        master MBID in MusicBrainz due to database merges.

        "otmm_makam_recognition_dataset" keeps track of such discrepancies in
        in the dunya_uid field. This field is not populated if the dunya_uid
        & MBID are the same.

        This method fills missing dunya_uid's by the corresponding MBID
        """
        self.data.loc[self.data["dunya_uid"].isna(), "dunya_uid"] = (
            self.data.loc[self.data["dunya_uid"].isna(), "mbid"])
        self.data.loc[self.data["dunya_uid"].isna(), "dunya_uid"] = 5