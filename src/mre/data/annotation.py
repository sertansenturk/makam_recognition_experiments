import pandas as pd
from . dataset import Dataset


class Annotation:
    """class to read and parse makam recognition annotations"""
    def __init__(self, num_recordings: int = 0, num_makams: int = 0):
        """instantiates an Annotation object

        Parameters
        ----------
        num_recordings : int, optional
            number of recordings in the annotations, by default 0
        num_makams : int, optional
            number of makams in the annotations, by default 0
        """
        self.annotations = None

    def read_from_github(self):
        self.annotations = pd.read_json(Dataset.get_annotation_file_url())
        self.validate()
        
    def validate(self, Dataset):
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
            raise ValueError(f"There are less than {num_makams} makams")

        num_recs_per_makam = makam_counts.unique()

        is_rec_makam_valid = (
            (len(num_recs_per_makam) == 1) and
            (num_recs_per_makam[0] == Dataset.get_num_recs_per_makam()))
        if not is_rec_makam_valid:
            raise ValueError("The number of recorings per makam should have "
                             f"been {Dataset.get_num_recs_per_makam()}")

    @staticmethod
    def parse_annotations(anno: pd.DataFrame) -> pd.DataFrame:
        pass
