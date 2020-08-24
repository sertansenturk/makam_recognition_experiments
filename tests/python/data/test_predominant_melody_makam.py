from unittest import mock

import pytest

import pandas as pd
from mre.data import PredominantMelodyMakam


@pytest.fixture
def mock_tmp_dir(scope="session") -> mock.MagicMock:
    tmp_dir = mock.MagicMock()
    tmp_dir.name = "/tmp/dir_path"

    return tmp_dir


@pytest.fixture
def mock_experiment(scope="session") -> mock.MagicMock:
    experiment = mock.MagicMock()
    experiment.experiment_id = "mock_id"
    return experiment


class TestPredominantMelodyMakam():
    def test_extract_empty_paths(self):
        # GIVEN
        audio_paths = []

        # WHEN; THEN
        pmm = PredominantMelodyMakam()
        with pytest.raises(ValueError):
            pmm.extract(audio_paths)

    def test_extract(self, mock_tmp_dir):
        # GIVEN
        audio_paths = ["./path/audio1.mp3", "./path/audio2.mp3"]
        mock_pitch = [[0, 1], [2, 0], [1, 2]]

        # WHEN
        pmm = PredominantMelodyMakam()
        with mock.patch("tempfile.TemporaryDirectory",
                        autospec=True,
                        return_value=mock_tmp_dir):
            with mock.patch.object(pmm.extractor,
                                   'extract',
                                   autospec=True,
                                   return_value={"pitch": mock_pitch}
                                   ) as mock_extract:
                with mock.patch('builtins.open',
                                mock.mock_open()
                                ) as mock_open:
                    pmm.extract(audio_paths)

        # THEN
        expected_num_writes = len(audio_paths) * len(mock_pitch)
        expected_extract_calls = [mock.call(ap) for ap in audio_paths]

        mock_extract.assert_has_calls(expected_extract_calls)
        assert mock_open().write.call_count == expected_num_writes

    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.set_experiment")
    def test_log_existing_run(self,
                              mock_mlflow_set_experiment,
                              mock_mlflow_start_run):
        # GIVEN
        pmm = PredominantMelodyMakam()
        mock_run = pd.DataFrame([{"run_id": "rid"}])

        # WHEN; THEN
        with mock.patch("mre.data.predominant_melody_makam.get_run_by_name",
                        return_value=mock_run):
            with pytest.raises(ValueError):
                pmm.log()

            mock_mlflow_set_experiment.assert_not_called()
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
                        mock_tmp_dir,
                        mock_experiment):
        # GIVEN
        pmm = PredominantMelodyMakam()
        mock_run = pd.DataFrame(columns=["run_id"])  # empty

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=mock_experiment):
            with mock.patch('mlflow.search_runs',
                            autospec=True,
                            return_value=mock_run):
                with mock.patch.object(pmm,
                                       "tmp_dir",
                                       mock_tmp_dir):
                    with mock.patch.object(pmm,
                                           "_cleanup"
                                           ) as mock_cleanup:
                        pmm.log()

                        mock_mlflow_set_experiment.assert_called_once()
                        mock_mlflow_start_run.assert_called_once()

                        mock_mlflow_set_tags.assert_called_once()
                        mock_mlflow_log_artifacts.assert_called_once_with(
                            pmm._tmp_dir_path())
                        mock_cleanup.assert_called_once_with()

    @mock.patch("mlflow.log_artifacts")
    @mock.patch("mlflow.set_tags")
    @mock.patch("mlflow.start_run")
    @mock.patch("mlflow.set_experiment")
    def test_log_no_experiment(self,
                               mock_mlflow_set_experiment,
                               mock_mlflow_start_run,
                               mock_mlflow_set_tags,
                               mock_mlflow_log_artifacts,
                               mock_tmp_dir):
        # GIVEN
        pmm = PredominantMelodyMakam()

        # WHEN; THEN
        with mock.patch('mlflow.get_experiment_by_name',
                        autospec=True,
                        return_value=None):
            with mock.patch.object(pmm,
                                   "tmp_dir",
                                   mock_tmp_dir):
                with mock.patch.object(pmm,
                                       "_cleanup"
                                       ) as mock_cleanup:
                    pmm.log()

                    mock_mlflow_set_experiment.assert_called_once()
                    mock_mlflow_start_run.assert_called_once()

                    mock_mlflow_set_tags.assert_called_once()
                    mock_mlflow_log_artifacts.assert_called_once_with(
                        pmm._tmp_dir_path())
                    mock_cleanup.assert_called_once_with()
