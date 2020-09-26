from unittest import mock

import pytest

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

    def test_log(self,
                 mock_tmp_dir):
        # GIVEN
        pmm = PredominantMelodyMakam()

        # WHEN; THEN
        with mock.patch("mre.data.predominant_melody_makam.get_run_by_name",
                        return_value="mock_audio_run_id"):
            with mock.patch("mre.data.predominant_melody_makam.log"
                            ) as mock_log:
                with mock.patch.object(pmm,
                                       "tmp_dir",
                                       mock_tmp_dir):
                    with mock.patch.object(pmm,
                                           "_cleanup"
                                           ) as mock_cleanup:
                        pmm.log()

                        mock_log.assert_called_once_with(
                            experiment_name=pmm.EXPERIMENT_NAME,
                            run_name=pmm.RUN_NAME,
                            artifact_dir=pmm._tmp_dir_path(),
                            tags=pmm._mlflow_tags()
                        )
                        mock_cleanup.assert_called_once_with()

    def test_mlflow_tags(self):
        # GIVEN
        pmm = PredominantMelodyMakam()
        mock_extractor_settings = {"setting1": "value1"}
        mock_audio_run_id = "mock_audio_run_id"

        # WHEN
        with mock.patch("mre.data.predominant_melody_makam.get_run_by_name",
                        return_value=mock_audio_run_id):
            with mock.patch.object(pmm.extractor,
                                   "get_settings",
                                   return_value=mock_extractor_settings):
                result = pmm._mlflow_tags()

        # THEN
        expected = {**mock_extractor_settings,
                    "source_run_id": mock_audio_run_id}

        assert result == expected
