from unittest import mock

import pytest

import numpy as np
import pandas as pd
from mre.data import Annotation
from pandas.testing import assert_frame_equal


@pytest.fixture
def mock_annotation(scope="session") -> Annotation:
    with mock.patch("mre.data.Annotation.__init__",
                    return_value=None,
                    autospec=True):
        return Annotation()


@pytest.fixture
def mock_experiment(scope="session") -> mock.MagicMock:
    experiment = mock.MagicMock()
    experiment.experiment_id = "mock_id"
    return experiment


class TestAnnotation:
    @mock.patch('mre.data.Annotation._validate')
    @mock.patch('mre.data.Annotation._read_from_github')
    def test_annotation_init(self, mock_read, mock_validate):
        # WHEN
        _ = Annotation()

        # THEN
        mock_read.assert_called_once_with()
        mock_validate.assert_called_once_with()

    @mock.patch('pandas.read_json', autospec=True)
    def test_read_from_github(self, mock_read_json, mock_annotation):
        # GIVEN
        url = "mock_url"

        # WHEN
        with mock.patch.object(mock_annotation, 'URL', url):
            _ = mock_annotation._read_from_github()

        # THEN
        mock_read_json.assert_called_once_with(url)

    @mock.patch('mre.data.Annotation._validate_num_makams')
    @mock.patch('mre.data.Annotation._validate_num_recordings_per_makam')
    @mock.patch('mre.data.Annotation._validate_mbids')
    @mock.patch('mre.data.Annotation._validate_num_recordings')
    def test_validate(self,
                      mock_validate_num_recordings,
                      mock_validate_mbids,
                      mock_validate_num_recordings_per_makam,
                      mock_validate_num_makams,
                      mock_annotation):
        # WHEN
        mock_annotation._validate()

        # THEN
        mock_validate_num_recordings.assert_called_once_with()
        mock_validate_mbids.assert_called_once_with()
        mock_validate_num_recordings_per_makam.assert_called_once_with()
        mock_validate_num_makams.assert_called_once_with()

    def test_validate_num_recordings(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])

        # WHEN; THEN
        with mock.patch.object(mock_annotation,
                               'EXPECTED_NUM_RECORDINGS',
                               2):
            mock_annotation._validate_num_recordings()

    def test_validate_unexpected_num_recordings(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])

        # WHEN; THEN
        with mock.patch.object(mock_annotation,
                               'EXPECTED_NUM_RECORDINGS',
                               3):  # not 2
            with pytest.raises(ValueError):
                mock_annotation._validate_num_recordings()

    def test_validate_mbids(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"mbid": "id1"}, {"mbid": "id2"}])

        # WHEN; THEN
        mock_annotation._validate_mbids()

    def test_validate_nonunique_mbids(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"mbid": "id1"}, {"mbid": "id1"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_mbids()

    @pytest.mark.parametrize("data", [
        pd.DataFrame([{"mbid": ""}]),
        pd.DataFrame([{"mbid": None}]),
        pd.DataFrame([{"mbid": np.nan}])])
    def test_validate_missing_mbid(self, mock_annotation, data):
        # GIVEN
        mock_annotation.data = data

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_mbids()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam1"},
                       {"makam": "makam2"}, {"makam": "makam2"}]), 2)])
    def test_val_num_recs_per_makam(self, mock_annotation, data, expected):
        # GIVEN
        mock_annotation.data = data

        # WHEN; THEN
        with mock.patch.object(mock_annotation,
                               'EXPECTED_NUM_RECORDINGS_PER_MAKAM',
                               expected):
            mock_annotation._validate_num_recordings_per_makam()

    def test_val_num_recs_per_makam_unbalanced(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame([  # unbalanced
            {"makam": "makam1"}, {"makam": "makam1"}, {"makam": "makam2"}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._validate_num_recordings_per_makam()

    def test_val_num_recs_per_makam_unexpected(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame([  # 2 recordings per makams
            {"makam": "makam1"}, {"makam": "makam2"}])

        # WHEN; THEN
        with mock.patch.object(mock_annotation,
                               'EXPECTED_NUM_RECORDINGS_PER_MAKAM',
                               5):  # not 2
            with pytest.raises(ValueError):
                mock_annotation._validate_num_recordings_per_makam()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"makam": "makam1"}]), 1),
        (pd.DataFrame([{"makam": "makam1"}, {"makam": "makam2"}]), 2)])
    def test_validate_num_makams(self, mock_annotation, data, expected):
        # GIVEN
        mock_annotation.data = data

        # WHEN; THEN
        with mock.patch.object(mock_annotation,
                               'EXPECTED_NUM_MAKAMS',
                               expected):
            mock_annotation._validate_num_makams()

    def test_validate_num_makams_incorrect(self, mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame([  # 3 makams
            {"makam": "makam1"}, {"makam": "makam2"}, {"makam": "makam3"}])

        # WHEN; THEN
        with mock.patch.object(mock_annotation,
                               'EXPECTED_NUM_MAKAMS',
                               5):  # not 3
            with pytest.raises(ValueError):
                mock_annotation._validate_num_makams()

    @mock.patch('mre.data.Annotation._parse_mbid_urls')
    @mock.patch('mre.data.Annotation._patch_dunya_uids')
    def test_parse(self,
                   mock_patch_dunya_uids,
                   mock_parse_mbid_urls,
                   mock_annotation):
        # WHEN
        mock_annotation.parse()

        # THEN
        mock_patch_dunya_uids.assert_called_once_with()
        mock_parse_mbid_urls.assert_called_once_with()

    def test_parse_mbid_urls(self, mock_annotation):
        # GIVEN
        mb_url = "http://musicbrainz.org/recording/some_mbid"
        mock_annotation.data = pd.DataFrame([{"mbid": mb_url}])

        # WHEN
        mock_annotation._parse_mbid_urls()

        # THEN
        expected = pd.DataFrame([{"mb_url": mb_url, "mbid": "some_mbid"}])

        assert_frame_equal(mock_annotation.data, expected, check_like=True)

    @pytest.mark.parametrize("invalid_url", [
        None,
        "",
        "string",
        "string/some_mbid",
        "http://wrongdomain.com/some_mbid",
        "http://musicbrainz.org/work/some_mbid",  # not recording
    ])
    def test_parse_mbid_urls_invalid_url(self, mock_annotation, invalid_url):
        # GIVEN
        mock_annotation.data = pd.DataFrame([{"mbid": invalid_url}])

        # WHEN; THEN
        with pytest.raises(ValueError):
            mock_annotation._parse_mbid_urls()

    @pytest.mark.parametrize("data,expected", [
        (pd.DataFrame([{"dunya_uid": None, "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_mbid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": "some_mbid"}])),
        (pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]),
         pd.DataFrame([{"dunya_uid": "some_uid", "mbid": None}]))])
    def test_patch_dunya_uids(self, mock_annotation, data, expected):
        # GIVEN
        mock_annotation.data = data

        # WHEN
        mock_annotation._patch_dunya_uids()

        # THEN
        assert_frame_equal(mock_annotation.data, expected, check_like=True)

    @mock.patch("mre.data.annotation.logger.warning")
    @mock.patch("mlflow.start_run")
    def test_log_existing(self,
                          mock_mlflow_start_run,
                          mock_logger_warning,
                          mock_annotation,
                          mock_experiment):
        # GIVEN
        mock_annotation.data = "dummy_data"

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=pd.DataFrame([{"run_id": "rid"}])):
                mock_annotation.log()

                mock_mlflow_start_run.assert_not_called()
                mock_logger_warning.assert_called_once()

    @mock.patch("mlflow.log_artifacts")
    @mock.patch("mlflow.set_tags")
    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.set_experiment")
    @mock.patch("mre.data.annotation.logger.warning")
    def test_log_no_experiment(self,
                               mock_logger_warning,
                               mock_mlflow_set_experiment,
                               mock_mlflow_start_run,
                               mock_mlflow_set_tags,
                               mock_mlflow_log_artifacts,
                               mock_annotation):
        # GIVEN
        mock_annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])
        tmp_dir = "/tmp/dir"

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        return_value=None):  # no experiment
            with mock.patch.object(mock_annotation.data, 'to_json'):
                with mock.patch('tempfile.TemporaryDirectory',
                                autospec=True) as tmp_dir_context:
                    tmp_dir_context.return_value.__enter__.return_value = \
                        tmp_dir
                    mock_annotation.log()

                    mock_logger_warning.assert_not_called()

                    mock_mlflow_set_experiment.assert_called_once()
                    mock_mlflow_start_run.assert_called_once()

                    mock_mlflow_set_tags.assert_called_once()
                    mock_mlflow_log_artifacts.assert_called_once_with(
                        tmp_dir)
