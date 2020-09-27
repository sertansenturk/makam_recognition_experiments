from unittest import mock

import mlflow
import numpy as np
import pandas as pd
import pytest

from mre.data import Annotation
from pandas.testing import assert_frame_equal


@pytest.fixture
def mock_experiment(scope="session") -> mock.MagicMock:
    experiment = mock.MagicMock()
    experiment.experiment_id = "mock_id"
    return experiment


class TestAnnotation:
    @mock.patch("mre.data.annotation.get_run_by_name", return_value=None)
    def test_from_mlflow_no_run(self, mock_run):
        # GIVEN
        annotation = Annotation()

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotation.from_mlflow()
        mock_run.assert_called_once()

    def test_from_mlflow(self):
        # GIVEN
        annotation = Annotation()
        mock_run = pd.Series({"run_id": "rid1"})
        annotation_filepath = "annotation_path.json"

        # WHEN; THEN
        with mock.patch("mre.data.annotation.get_run_by_name",
                        return_value=mock_run):
            with mock.patch('mlflow.tracking.MlflowClient.__init__',
                            autospec=True,
                            return_value=None):
                with mock.patch.object(mlflow.tracking.MlflowClient,
                                       "download_artifacts",
                                       autospec=True,
                                       return_value=annotation_filepath
                                       ) as mock_download_artifacts:
                    with mock.patch('pandas.read_json',
                                    autospec=True
                                    ) as mock_read_json:
                        annotation.from_mlflow()

                        mock_download_artifacts.assert_called_once()
                        mock_read_json.assert_called_once_with(
                            annotation_filepath, orient="records")

    @mock.patch('mre.data.Annotation._validate')
    @mock.patch('pandas.read_json', autospec=True)
    def test_from_github(self, mock_read_json, mock_validate):
        # GIVEN
        url = "mock_url"
        annotation = Annotation()

        # WHEN
        with mock.patch.object(annotation, 'URL', url):
            _ = annotation.from_github()

        # THEN
        mock_read_json.assert_called_once_with(url)
        mock_validate.assert_called_once_with()

    @mock.patch('mre.data.Annotation._validate_num_makams')
    @mock.patch('mre.data.Annotation._validate_num_recordings_per_makam')
    @mock.patch('mre.data.Annotation._validate_mbids')
    @mock.patch('mre.data.Annotation._validate_num_recordings')
    def test_validate(self,
                      mock_validate_num_recordings,
                      mock_validate_mbids,
                      mock_validate_num_recordings_per_makam,
                      mock_validate_num_makams):
        # GIVEN
        annotation = Annotation()

        # WHEN
        annotation._validate()

        # THEN
        mock_validate_num_recordings.assert_called_once_with()
        mock_validate_mbids.assert_called_once_with()
        mock_validate_num_recordings_per_makam.assert_called_once_with()
        mock_validate_num_makams.assert_called_once_with()

    def test_validate_num_recordings(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])

        # WHEN; THEN
        with mock.patch.object(annotation,
                               'EXPECTED_NUM_RECORDINGS',
                               2):
            annotation._validate_num_recordings()

    def test_validate_unexpected_num_recordings(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])

        # WHEN; THEN
        with mock.patch.object(annotation,
                               'EXPECTED_NUM_RECORDINGS',
                               3):  # not 2
            with pytest.raises(ValueError):
                annotation._validate_num_recordings()

    def test_validate_mbids(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"mbid": "id1"}, {"mbid": "id2"}])

        # WHEN; THEN
        annotation._validate_mbids()

    def test_validate_nonunique_mbids(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"mbid": "id1"}, {"mbid": "id1"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotation._validate_mbids()

    @pytest.mark.parametrize("data", [
        pd.DataFrame([{"mbid": ""}]),
        pd.DataFrame([{"mbid": None}]),
        pd.DataFrame([{"mbid": np.nan}])])
    def test_validate_missing_mbid(self, data):
        # GIVEN
        annotation = Annotation()
        annotation.data = data

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotation._validate_mbids()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam1"},
                       {"makam": "makam2"}, {"makam": "makam2"}]), 2)])
    def test_val_num_recs_per_makam(self, data, expected):
        # GIVEN
        annotation = Annotation()
        annotation.data = data

        # WHEN; THEN
        with mock.patch.object(annotation,
                               'EXPECTED_NUM_RECORDINGS_PER_MAKAM',
                               expected):
            annotation._validate_num_recordings_per_makam()

    def test_val_num_recs_per_makam_unbalanced(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame([  # unbalanced
            {"makam": "makam1"}, {"makam": "makam1"}, {"makam": "makam2"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotation._validate_num_recordings_per_makam()

    def test_val_num_recs_per_makam_unexpected(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame([  # 2 recordings per makams
            {"makam": "makam1"}, {"makam": "makam2"}])

        # WHEN; THEN
        with mock.patch.object(annotation,
                               'EXPECTED_NUM_RECORDINGS_PER_MAKAM',
                               5):  # not 2
            with pytest.raises(ValueError):
                annotation._validate_num_recordings_per_makam()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 2)])
    def test_validate_num_makams(self, data, expected):
        # GIVEN
        annotation = Annotation()
        annotation.data = data

        # WHEN; THEN
        with mock.patch.object(annotation,
                               'EXPECTED_NUM_MAKAMS',
                               expected):
            annotation._validate_num_makams()

    def test_validate_num_makams_incorrect(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame([  # 3 makams
            {"makam": "makam1"}, {"makam": "makam2"}, {"makam": "makam3"}])

        # WHEN; THEN
        with mock.patch.object(annotation,
                               'EXPECTED_NUM_MAKAMS',
                               5):  # not 3
            with pytest.raises(ValueError):
                annotation._validate_num_makams()

    @mock.patch('mre.data.Annotation._parse_mbid_urls')
    @mock.patch('mre.data.Annotation._patch_dunya_uids')
    def test_parse(self, mock_patch_dunya_uids, mock_parse_mbid_urls):
        # WHEN
        annotation = Annotation()
        annotation.parse()

        # THEN
        mock_patch_dunya_uids.assert_called_once_with()
        mock_parse_mbid_urls.assert_called_once_with()

    def test_parse_mbid_urls(self):
        # GIVEN
        annotation = Annotation()
        mb_url = "http://musicbrainz.org/recording/some_mbid"
        annotation.data = pd.DataFrame([{"mbid": mb_url}])

        # WHEN
        annotation._parse_mbid_urls()

        # THEN
        expected = pd.DataFrame([{"mb_url": mb_url, "mbid": "some_mbid"}])

        assert_frame_equal(annotation.data, expected, check_like=True)

    @pytest.mark.parametrize("invalid_url", [
        None,
        "",
        "string",
        "string/some_mbid",
        "http://wrongdomain.com/some_mbid",
        "http://musicbrainz.org/work/some_mbid",  # not recording
    ])
    def test_parse_mbid_urls_invalid_url(self, invalid_url):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame([{"mbid": invalid_url}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            annotation._parse_mbid_urls()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]))])
    def test_patch_dunya_uids(self, data, expected):
        # GIVEN
        annotation = Annotation()
        annotation.data = data

        # WHEN
        annotation._patch_dunya_uids()

        # THEN
        assert_frame_equal(annotation.data, expected, check_like=True)

    def test_log(self):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])
        tmp_dir = "/tmp/dir"

        # WHEN; THEN
        with mock.patch("mre.data.annotation.log") as mock_log:
            with mock.patch.object(annotation.data, 'to_json'):
                with mock.patch('tempfile.TemporaryDirectory',
                                autospec=True) as tmp_dir_cont:
                    tmp_dir_cont.return_value.__enter__.return_value = tmp_dir
                    annotation.log()

                    mock_log.assert_called_once_with(
                        experiment_name=annotation.EXPERIMENT_NAME,
                        run_name=annotation.RUN_NAME,
                        artifact_dir=tmp_dir,
                        tags=annotation._mlflow_tags())
