import logging

import pandas as pd

from ..config import config

logger = logging.Logger(  # pylint: disable-msg=C0103
    __name__, level=logging.INFO)


class Annotation:
    """class to read and process makam recognition annotations"""
    MUSICBRAINZ_RECORDING_URL = "http://musicbrainz.org/recording/"

    def __init__(self):
        """instantiates an Annotation object
        """
        self.annotation_url = config.read()["dataset"]["annotation_file"]

        logger.info("Reading annotations from: %s", self.annotation_url)

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

    def _read_from_github(self):
        """reads the annotation file from github and validates
        """
        return pd.read_json(self.annotation_url)

    def _get_num_recs_per_makam(self) -> int:
        """returns number of recordings per makam

        Returns
        -------
        int
            number of recordings per makam
        """
        num_recordings_per_makam = self.data.makam.value_counts()

        # balanced dataset; all classes have the same number of instances
        if num_recordings_per_makam.nunique() == 1:
            return num_recordings_per_makam[0]

        raise ValueError("Inbalanced datasets are not supported")

    def _validate(self):
        """validates the annotations

        Read the expected values mre config and verifies that:
            1) number of recordings
            2) number of makams
            3) number of recordings per makam
        have the expected values.
        """
        cfg = config.read()

        num_recordings = len(self.data.mbid)
        expected_num_recs = cfg.getint("dataset", "num_recordings")
        if num_recordings != expected_num_recs:
            raise ValueError(
                f"There are {num_recordings} recordings. "
                f"Expected: {expected_num_recs}.")

        unique_num_recordings = self.data.mbid.nunique()
        if unique_num_recordings != num_recordings:
            raise ValueError("MusicBrainz ID (MBIDs) are not unique.")
        logger.info("%d annotated recordings.", num_recordings)

        num_recordings_per_makam = self._get_num_recs_per_makam()
        expected_num_recs_per_makam = cfg.getint(
            "dataset", "num_recordings_per_makam")
        if num_recordings_per_makam == expected_num_recs_per_makam:
            raise ValueError(
                f"{num_recordings_per_makam} number of recordings per makam. "
                f"Expected: {expected_num_recs_per_makam}")
        logger.info("%d recordings per makam.", num_recordings_per_makam)

        num_makams = len(self.data.makam.value_counts())
        expected_num_makams = expected_num_recs / expected_num_recs_per_makam
        if num_makams != expected_num_makams:
            raise ValueError(
                f"There are {num_makams} makams. "
                f"Expected: {expected_num_makams}.")
        logger.info("%d makams.", num_makams)

    def _parse_mbid_urls(self):
        """Parses the urls in the mbid field

        The MBIDs in the "otmm_makam_recognition_dataset" are stored as
        URLs pointing to the MusicBrainz website.

        This method trims mbid's while moving the URL to a new field called
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
        """Patches missing dunya uid's with mbid's

        Some recordings in "CompMusic makam music corpus" may not point to the
        master MBID in MusicBrainz due to database merges.

        "otmm_makam_recognition_dataset" keeps track of such discrepancies in
        in the dunya_uid field. This field is not populated if the dunya_uid
        & mbid are the same.

        This method fills missing dunya_uid's by the corresponding mbid
        """
        self.data.loc[self.data["dunya_uid"].isna(), "dunya_uid"] = (
            self.data.loc[self.data["dunya_uid"].isna(), "mbid"])
        self.data.loc[self.data["dunya_uid"].isna(), "dunya_uid"] = 5
