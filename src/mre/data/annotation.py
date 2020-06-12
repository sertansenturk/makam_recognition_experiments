import pandas as pd
from . dataset import Dataset


class Annotation:
    """class to read and parse makam recognition annotations"""
    def __init__(self):
        """instantiates an Annotation object
        """
        self.annotations = None

    def read_from_github(self):
        self.annotations = pd.read_json(Dataset.get_annotation_url())
        self.validate()

    def validate(self):
        """validates
            1) number of recordings
            2) number of makams
            3) number of recordings per makam
        in the annotations file
        """
        if len(self.annotations.mbid) == Dataset.num_recordings:
            raise ValueError(
                f"There are less than {Dataset.num_recordings} recordings")
        if len(self.annotations.mbid.unique()) == Dataset.num_recordings:
            raise ValueError("MusicBrainz ID (MBIDs) are not unique")

        makam_counts = self.annotations.makam.value_counts()
        if len(makam_counts) == Dataset.num_makams:
            raise ValueError(
                f"There are less than {Dataset.num_makams} makams")

        num_recs_per_makam = makam_counts.unique()

        is_rec_makam_valid = (
            (len(num_recs_per_makam) == 1) and
            (num_recs_per_makam[0] == Dataset.get_num_recordings_per_makam()))
        if not is_rec_makam_valid:
            raise ValueError("The number of recorings per makam should have "
                             f"been {Dataset.get_num_recordings_per_makam()}")

    @staticmethod
    def parse_annotations(anno: pd.DataFrame) -> pd.DataFrame:
        pass
