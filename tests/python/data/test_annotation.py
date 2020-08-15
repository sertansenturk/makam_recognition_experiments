from unittest import mock

import pytest

import numpy as np
import pandas as pd
from mre.data import Annotation
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.fixture
def mock_experiment(scope="session") -> mock.MagicMock:
    experiment = mock.MagicMock()
    experiment.experiment_id = "mock_id"
    return experiment


class TestAnnotation:
    # def test_from_mlflow(self):
    #     # GIVEN
    #     annotations = Annotation()

    #     # WHEN
    #     annotations.read()

    #     # THEN
    #     assert False

    @mock.patch("mre.data.annotation.logger.warning")
    def test_get_mlflow_run_no_experiment(self, mock_warning):
        # WHEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=None):
            result = Annotation.get_mlflow_run()

        # THEN
        mock_warning.assert_called_once()
        assert result is None

    @mock.patch("mre.data.annotation.logger.warning")
    def test_get_mlflow_run_no_run(self, mock_warning, mock_experiment):
        # GIVEN
        mock_runs = pd.DataFrame(columns=["run_id"])  # empty

        # WHEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=mock_runs):
                result = Annotation.get_mlflow_run()

        # THEN
        mock_warning.assert_called_once()
        assert result is None

    @mock.patch("mre.data.annotation.logger.warning")
    def test_get_mlflow_run_multi_runs(self, mock_warning, mock_experiment):
        # GIVEN
        mock_runs = pd.DataFrame([{"run_id": "rid1"}, {"run_id": "rid2"}])

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=mock_runs):
                with pytest.raises(ValueError):
                    Annotation.get_mlflow_run()

    def test_get_mlflow_run_single_run(self, mock_experiment):
        # GIVEN
        mock_run_dict = {"run_id": "rid1"}
        mock_runs = pd.DataFrame([mock_run_dict])

        # WHEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=mock_runs):
                result = Annotation.get_mlflow_run()

        # THEN
        expected = pd.Series(mock_run_dict)
        assert_series_equal(result, expected, check_names=False)

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

    @mock.patch("mlflow.start_run")
    def test_log_existing_run(self,
                              mock_mlflow_start_run,
                              mock_experiment):
        # GIVEN
        annotation = Annotation()
        annotation.data = "dummy_data"
        mock_run = pd.DataFrame([{"run_id": "rid"}])

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=mock_run):
                with pytest.raises(ValueError):
                    annotation.log()
                mock_mlflow_start_run.assert_not_called()

    @mock.patch("mlflow.log_artifacts")
    @mock.patch("mlflow.set_tags")
    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.set_experiment")
    def test_log_no_run(self,
                        mock_mlflow_set_experiment,
                        mock_mlflow_start_run,
                        mock_mlflow_set_tags,
                        mock_mlflow_log_artifacts,
                        mock_experiment):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])
        tmp_dir = "/tmp/dir"
        mock_run = pd.DataFrame(columns=["run_id"])  # empty

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=mock_run):
                with mock.patch.object(annotation.data, 'to_json'):
                    with mock.patch('tempfile.TemporaryDirectory',
                                    autospec=True) as tmp_dir_cont:
                        tmp_dir_cont.return_value.__enter__.return_value = \
                            tmp_dir
                        annotation.log()

                        mock_mlflow_set_experiment.assert_called_once()
                        mock_mlflow_start_run.assert_called_once()

                        mock_mlflow_set_tags.assert_called_once()
                        mock_mlflow_log_artifacts.assert_called_once_with(
                            tmp_dir)

    @mock.patch("mlflow.log_artifacts")
    @mock.patch("mlflow.set_tags")
    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.set_experiment")
    def test_log_no_experiment(self,
                               mock_mlflow_set_experiment,
                               mock_mlflow_start_run,
                               mock_mlflow_set_tags,
                               mock_mlflow_log_artifacts):
        # GIVEN
        annotation = Annotation()
        annotation.data = pd.DataFrame(
            [{"col1": "val1"}, {"col1": "val2"}])
        tmp_dir = "/tmp/dir"

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        return_value=None):  # no experiment
            with mock.patch.object(annotation.data, 'to_json'):
                with mock.patch('tempfile.TemporaryDirectory',
                                autospec=True) as tmp_dir_cont:
                    tmp_dir_cont.return_value.__enter__.return_value = tmp_dir
                    annotation.log()

                    mock_mlflow_set_experiment.assert_called_once()
                    mock_mlflow_start_run.assert_called_once()

                    mock_mlflow_set_tags.assert_called_once()
                    mock_mlflow_log_artifacts.assert_called_once_with(
                        tmp_dir)
