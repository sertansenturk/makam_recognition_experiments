from pathlib import Path
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


class TestPredominantMelodyMakam:
    def test_transform_empty_paths(self):
        # GIVEN
        audio_paths = []

        # WHEN; THEN
        pmm = PredominantMelodyMakam()
        with pytest.raises(ValueError):
            pmm.transform(audio_paths)

    def test_transform(self, mock_tmp_dir):
        # GIVEN
        audio_paths = ["./path/file1.mp3", "./path/file2.mp3"]
        mock_pitch = [[0, 1], [2, 0], [1, 2]]

        # WHEN
        pmm = PredominantMelodyMakam()
        with mock.patch(
            "tempfile.TemporaryDirectory", autospec=True, return_value=mock_tmp_dir
        ):
            with mock.patch.object(
                pmm, "transform_func", autospec=True, return_value={"pitch": mock_pitch}
            ) as mock_extract:
                with mock.patch(
                    "numpy.save",
                    autospec=True,
                ) as mock_save:
                    pmm.transform(audio_paths)

        # THEN
        expected_extract_calls = [mock.call(ap) for ap in audio_paths]
        expected_save_calls = [
            mock.call(Path(mock_tmp_dir.name, "file1.npy"), mock_pitch),
            mock.call(Path(mock_tmp_dir.name, "file2.npy"), mock_pitch),
        ]

        mock_extract.assert_has_calls(expected_extract_calls)
        mock_save.assert_has_calls(expected_save_calls)

    def test_transform_existing_tmp_dir(self, mock_tmp_dir):
        # GIVEN
        audio_paths = ["./path/file1.mp3"]
        mock_pitch = [[0, 1], [2, 0], [1, 2]]
        pmm = PredominantMelodyMakam()
        pmm.tmp_dir = mock_tmp_dir  # transform called before

        # WHEN
        with mock.patch.object(pmm, "_cleanup", autospec=True) as mock_cleanup:
            with mock.patch(
                "tempfile.TemporaryDirectory", autospec=True, return_value=mock_tmp_dir
            ):
                with mock.patch.object(
                    pmm,
                    "transform_func",
                    autospec=True,
                    return_value={"pitch": mock_pitch},
                ):
                    with mock.patch("numpy.save", autospec=True):
                        pmm.transform(audio_paths)

        # THEN
        mock_cleanup.assert_called_once_with()

    def test_mlflow_tags(self):
        # GIVEN
        pmm = PredominantMelodyMakam()
        mock_transformer_settings = {"setting1": "value1"}

        # WHEN
        with mock.patch.object(
            pmm.transformer, "get_settings", return_value=mock_transformer_settings
        ):
            result = pmm._mlflow_tags()

        # THEN
        expected = mock_transformer_settings

        assert result == expected
