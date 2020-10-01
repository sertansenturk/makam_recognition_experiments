from unittest import mock
import pytest
import pathlib

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
                with mock.patch('numpy.save',
                                autospec=True,
                                ) as mock_save:
                    pmm.extract(audio_paths)

        # THEN
        expected_extract_calls = [mock.call(ap) for ap in audio_paths]
        expected_save_calls = [
            mock.call(pathlib.Path(mock_tmp_dir.name, "audio1.npy"),
                      mock_pitch),
            mock.call(pathlib.Path(mock_tmp_dir.name, "audio2.npy"),
                      mock_pitch)]

        mock_extract.assert_has_calls(expected_extract_calls)
        mock_save.assert_has_calls(expected_save_calls)

    def test_extract_existing_tmp_dir(self, mock_tmp_dir):
        # GIVEN
        audio_paths = ["./path/audio1.mp3", "./path/audio2.mp3"]
        mock_pitch = [[0, 1], [2, 0], [1, 2]]
        pmm = PredominantMelodyMakam()
        pmm.tmp_dir = mock_tmp_dir  # extract called before

        # WHEN
        with mock.patch.object(pmm,
                               '_cleanup',
                               autospec=True) as mock_cleanup:
            with mock.patch("tempfile.TemporaryDirectory",
                            autospec=True,
                            return_value=mock_tmp_dir):
                with mock.patch.object(pmm.extractor,
                                       'extract',
                                       autospec=True,
                                       return_value={"pitch": mock_pitch}
                                       ) as mock_extract:
                    with mock.patch('numpy.save',
                                    autospec=True,
                                    ) as mock_save:
                        pmm.extract(audio_paths)

        # THEN
        expected_extract_calls = [mock.call(ap) for ap in audio_paths]
        expected_save_calls = [
            mock.call(pathlib.Path(mock_tmp_dir.name, "audio1.npy"),
                      mock_pitch),
            mock.call(pathlib.Path(mock_tmp_dir.name, "audio2.npy"),
                      mock_pitch)]

        mock_cleanup.assert_called_once_with()
        mock_extract.assert_has_calls(expected_extract_calls)
        mock_save.assert_has_calls(expected_save_calls)

    def test_mlflow_tags(self):
        # GIVEN
        pmm = PredominantMelodyMakam()
        mock_extractor_settings = {"setting1": "value1"}
        mock_audio_run = pd.DataFrame([{"run_id": "mock_audio_run_id"}])

        # WHEN
        with mock.patch("mre.data.predominant_melody_makam.get_run_by_name",
                        return_value=mock_audio_run):
            with mock.patch.object(pmm.extractor,
                                   "get_settings",
                                   return_value=mock_extractor_settings):
                result = pmm._mlflow_tags()

        # THEN
        expected = {**mock_extractor_settings,
                    "source_run_id": mock_audio_run["run_id"]}

        assert result == expected
