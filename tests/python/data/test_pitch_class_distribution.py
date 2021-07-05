from unittest import mock
from pathlib import Path
import pytest

import pandas as pd
import numpy as np
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


class TestPitchClassDistribution:
    def test_transform_empty_melody_paths(self):
        # GIVEN
        melody_paths = []
        tonic_freqs = pd.Series({"id1": 400, "id2": 300})

        # WHEN; THEN
        pcd = PitchClassDistribution()
        with pytest.raises(ValueError):
            pcd.transform(melody_paths, tonic_freqs)

    def test_transform_empty_tonic_freqs(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series({}, dtype=np.float)

        # WHEN; THEN
        pcd = PitchClassDistribution()
        with pytest.raises(ValueError):
            pcd.transform(melody_paths, tonic_freqs)

    def test_transform_duplicate_melody_paths(self):
        # GIVEN
        melody_paths = ["id1.npy", "id1.npy"]
        tonic_freqs = pd.Series({"id1": 400, "id2": 300})

        # WHEN; THEN
        pcd = PitchClassDistribution()
        with pytest.raises(ValueError):
            pcd.transform(melody_paths, tonic_freqs)

    def test_transform_duplicate_tonic_ids(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series([400, 300], index=["id1", "id1"])

        # WHEN; THEN
        pcd = PitchClassDistribution()
        with pytest.raises(ValueError):
            pcd.transform(melody_paths, tonic_freqs)

    def test_transform_mbid_mismatch(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series([400, 300], index=["id1", "id3"])

        # WHEN; THEN
        pcd = PitchClassDistribution()
        with pytest.raises(ValueError):
            pcd.transform(melody_paths, tonic_freqs)

    def test_transform(self, mock_tmp_dir):
        # GIVEN
        melody_paths = ["./path/id1.npy", "./path/id2.npy"]
        tonic_freqs = pd.Series([400, 100], index=["id1", "id2"])
        mock_distribution = PitchDistribution(
            [100, 500, 1100], [0, 10, 8]  # do not span over an octave (1200 cents)
        )
        pcd = PitchClassDistribution()

        # WHEN
        with mock.patch(
            "tempfile.TemporaryDirectory", autospec=True, return_value=mock_tmp_dir
        ):
            with mock.patch("numpy.load", autospec=True) as mock_load:
                with mock.patch.object(
                    pcd, "transform_func", autospec=True, return_value=mock_distribution
                ):
                    with mock.patch.object(
                        mock_distribution,
                        "to_pcd",
                        autospec=True,
                    ) as mock_to_pcd:
                        with mock.patch.object(
                            mock_distribution,
                            "to_json",
                            autospec=True,
                        ) as mock_to_json:
                            pcd.transform(melody_paths, tonic_freqs)

        # THEN
        expected_to_json_calls = [
            mock.call(Path(mock_tmp_dir.name, "id1.json")),
            mock.call(Path(mock_tmp_dir.name, "id2.json")),
        ]

        assert mock_load.call_count == len(melody_paths)
        assert mock_to_pcd.call_count == len(melody_paths)
        mock_to_json.assert_has_calls(expected_to_json_calls)

    def test_transform_existing_tmp_dir(self, mock_tmp_dir):
        # GIVEN
        melody_paths = ["./path/id1.npy"]
        tonic_freqs = pd.Series([400], index=["id1"])
        mock_distribution = PitchDistribution(
            [100, 500, 1100], [0, 10, 8]  # do not span over an octave (1200 cents)
        )
        pcd = PitchClassDistribution()
        pcd.tmp_dir = mock_tmp_dir  # transform called before

        # WHEN
        with mock.patch.object(pcd, "_cleanup", autospec=True) as mock_cleanup:
            with mock.patch(
                "tempfile.TemporaryDirectory", autospec=True, return_value=mock_tmp_dir
            ):
                with mock.patch("numpy.load", autospec=True):
                    with mock.patch.object(
                        pcd,
                        "transform_func",
                        autospec=True,
                        return_value=mock_distribution,
                    ):
                        with mock.patch.object(
                            mock_distribution, "to_pcd", autospec=True
                        ):
                            with mock.patch.object(
                                mock_distribution, "to_json", autospec=True
                            ):
                                pcd.transform(melody_paths, tonic_freqs)

        # THEN
        mock_cleanup.assert_called_once_with()

    def test_mlflow_tags(self):
        # GIVEN
        pcd = PitchClassDistribution()

        # WHEN
        result = pcd._mlflow_tags()

        # THEN
        expected = {
            "kernel_width": pcd.KERNEL_WIDTH,
            "norm_type": pcd.NORM_TYPE,
            "step_size": pcd.STEP_SIZE,
        }

        assert result == expected
