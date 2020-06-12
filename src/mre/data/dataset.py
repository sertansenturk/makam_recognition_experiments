
class Dataset:
    name = "otmm_makam_recognition_dataset"
    git_release_tag = "dlfm2016-fix1"
    num_recordings = 1000
    num_makams = 20
    num_recordings_per_makam = 50

    @classmethod
    def get_num_recordings_per_makam(cls):
        return cls.num_recordings / cls.num_recordings

    @classmethod
    def get_url(cls):
        return f"https://raw.githubusercontent.com/sertansenturk/{cls.name}"

    @classmethod
    def get_annotation_url(cls):
        return f"{cls.get_url}/{cls.git_release_tag}/annotations.json"
