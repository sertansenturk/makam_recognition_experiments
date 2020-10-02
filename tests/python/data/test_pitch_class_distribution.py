from unittest import mock
from pathlib import Path
import pytest

import pandas as pd

from mre.data import PitchClassDistribution
from mre.data.pitch_class_distribution import PitchDistribution


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
        pitch_paths = []

        # WHEN; THEN
        pcd = PitchClassDistribution()
        with pytest.raises(ValueError):
            pcd.extract(pitch_paths)

    def test_extract(self, mock_tmp_dir):
        # GIVEN
        pitch_paths = ["./path/file1.npy", "./path/file2.npy"]
        mock_distribution = PitchDistribution(
            [100, 500, 1300],  # span over an octave (1200 cents)
            [0, 10, 8])
        pcd = PitchClassDistribution()

        # WHEN
        with mock.patch("tempfile.TemporaryDirectory",
                        autospec=True,
                        return_value=mock_tmp_dir):
            with mock.patch("numpy.load",
                            autospec=True
                            ) as mock_load:
                with mock.patch.object(PitchDistribution,
                                       attribute='from_hz_pitch',
                                       autospec=True,
                                       return_value=mock_distribution):
                    with mock.patch.object(mock_distribution,
                                           'to_pcd',
                                           autospec=True,
                                           ) as mock_to_pcd:
                        with mock.patch.object(mock_distribution,
                                               'to_json',
                                               autospec=True,
                                               ) as mock_to_json:
                            pcd.extract(pitch_paths)

        # THEN
        expected_to_json_calls = [
            mock.call(Path(mock_tmp_dir.name, "file1.json")),
            mock.call(Path(mock_tmp_dir.name, "file2.json"))]

        assert mock_load.call_count == len(pitch_paths)
        assert mock_to_pcd.call_count == len(pitch_paths)
        mock_to_json.assert_has_calls(expected_to_json_calls)

    def test_extract_existing_tmp_dir(self, mock_tmp_dir):
        # GIVEN
        pitch_paths = ["./path/file1.npy", "./path/file2.npy"]
        mock_distribution = PitchDistribution(
            [100, 500, 1300],  # span over an octave (1200 cents)
            [0, 10, 8])
        pcd = PitchClassDistribution()
        pcd.tmp_dir = mock_tmp_dir  # extract called before

        # WHEN
        with mock.patch.object(pcd,
                               '_cleanup',
                               autospec=True) as mock_cleanup:
            with mock.patch("tempfile.TemporaryDirectory",
                            autospec=True,
                            return_value=mock_tmp_dir):
                with mock.patch("numpy.load",
                                autospec=True
                                ) as mock_load:
                    with mock.patch.object(PitchDistribution,
                                           attribute='from_hz_pitch',
                                           autospec=True,
                                           return_value=mock_distribution):
                        with mock.patch.object(mock_distribution,
                                               'to_pcd',
                                               autospec=True,
                                               ) as mock_to_pcd:
                            with mock.patch.object(mock_distribution,
                                                   'to_json',
                                                   autospec=True,
                                                   ) as mock_to_json:
                                pcd.extract(pitch_paths)

        # THEN
        expected_to_json_calls = [
            mock.call(Path(mock_tmp_dir.name, "file1.json")),
            mock.call(Path(mock_tmp_dir.name, "file2.json"))]

        mock_cleanup.assert_called_once_with()
        assert mock_load.call_count == len(pitch_paths)
        assert mock_to_pcd.call_count == len(pitch_paths)
        mock_to_json.assert_has_calls(expected_to_json_calls)

    def test_mlflow_tags(self):
        # GIVEN
        pcd = PitchClassDistribution()
        mock_run = pd.Series({"run_id": "mock_run_id"})

        # WHEN
        with mock.patch("mre.data.pitch_class_distribution.get_run_by_name",
                        return_value=mock_run):
            result = pcd._mlflow_tags()

        # THEN
        assert result["source_run_id"] == mock_run["run_id"]
        # assert set(result.keys())
