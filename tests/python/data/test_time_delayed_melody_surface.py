from unittest import mock
from pathlib import Path
import pytest

import pandas as pd
import numpy as np

from mre.data.time_delayed_melody_surface import TimeDelayedMelodySurface
from mre.data.tdms_feature import TDMSFeature
from .test_tdms_feature import DEFAULT_EMBEDDING, DEFAULT_PITCH_BINS


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


class TestTimeDelayedMelodySurface:
    def test_transform_empty_melody_paths(self):
        # GIVEN
        melody_paths = []
        tonic_freqs = pd.Series({"id1": 400, "id2": 300})

        # WHEN; THEN
        tdms = TimeDelayedMelodySurface()
        with pytest.raises(ValueError, match="melody_paths is empty"):
            tdms.transform(melody_paths, tonic_freqs)

    def test_transform_empty_tonic_freqs(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series({}, dtype=np.float)

        # WHEN; THEN
        tdms = TimeDelayedMelodySurface()
        with pytest.raises(ValueError, match="tonic_frequencies is empty"):
            tdms.transform(melody_paths, tonic_freqs)

    def test_transform_duplicate_melody_paths(self):
        # GIVEN
        melody_paths = ["id1.npy", "id1.npy"]
        tonic_freqs = pd.Series({"id1": 400, "id2": 300})

        # WHEN; THEN
        tdms = TimeDelayedMelodySurface()
        with pytest.raises(ValueError, match="melody_paths has a duplicate path"):
            tdms.transform(melody_paths, tonic_freqs)

    def test_transform_duplicate_tonic_ids(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series([400, 300], index=["id1", "id1"])

        # WHEN; THEN
        tdms = TimeDelayedMelodySurface()
        with pytest.raises(ValueError, match="tonic_mbids has a duplicate index"):
            tdms.transform(melody_paths, tonic_freqs)

    def test_transform_mbid_mismatch(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series([400, 300], index=["id1", "id3"])

        # WHEN; THEN
        tdms = TimeDelayedMelodySurface()
        with pytest.raises(
            ValueError, match="MBIDs of melody_paths and tonic_mbids do not match"
        ):
            tdms.transform(melody_paths, tonic_freqs)

    def test_transform(self, mock_tmp_dir):
        # GIVEN
        melody_paths = ["./path/id1.npy", "./path/id2.npy"]
        tonic_freqs = pd.Series([400, 100], index=["id1", "id2"])
        mock_tdms_feature = TDMSFeature(
            embedding=DEFAULT_EMBEDDING, pitch_bins=DEFAULT_PITCH_BINS
        )
        tdms = TimeDelayedMelodySurface()

        # WHEN
        with mock.patch(
            "tempfile.TemporaryDirectory", autospec=True, return_value=mock_tmp_dir
        ):
            with mock.patch("numpy.load", autospec=True) as mock_load:
                with mock.patch.object(
                    tdms,
                    "transform_func",
                    autospec=True,
                    return_value=mock_tdms_feature,
                ):
                    with mock.patch.object(
                        mock_tdms_feature,
                        "to_json",
                        autospec=True,
                    ) as mock_to_json:
                        tdms.transform(melody_paths, tonic_freqs)

        # THEN
        expected_to_json_calls = [
            mock.call(Path(mock_tmp_dir.name, "id1.json")),
            mock.call(Path(mock_tmp_dir.name, "id2.json")),
        ]

        assert mock_load.call_count == len(melody_paths)
        mock_to_json.assert_has_calls(expected_to_json_calls)

    def test_transform_existing_tmp_dir(self, mock_tmp_dir):
        # GIVEN
        melody_paths = ["./path/id1.npy"]
        tonic_freqs = pd.Series([400], index=["id1"])
        mock_tdms_feature = TDMSFeature(
            embedding=DEFAULT_EMBEDDING, pitch_bins=DEFAULT_PITCH_BINS
        )
        tdms = TimeDelayedMelodySurface()
        tdms.tmp_dir = mock_tmp_dir  # transform called before

        # WHEN
        with mock.patch.object(tdms, "_cleanup", autospec=True) as mock_cleanup:
            with mock.patch(
                "tempfile.TemporaryDirectory", autospec=True, return_value=mock_tmp_dir
            ):
                with mock.patch("numpy.load", autospec=True):
                    with mock.patch.object(
                        tdms,
                        "transform_func",
                        autospec=True,
                        return_value=mock_tdms_feature,
                    ):
                        with mock.patch.object(
                            mock_tdms_feature, "to_json", autospec=True
                        ):
                            tdms.transform(melody_paths, tonic_freqs)

        # THEN
        mock_cleanup.assert_called_once_with()

    def test_mlflow_tags(self):
        # GIVEN
        tdms = TimeDelayedMelodySurface()  # use default values

        # WHEN
        result = tdms._mlflow_tags()

        # THEN
        expected = {  # defaults hardcoded
            "step_size": 25,
            "time_delay_index": 0.3,
            "compression_exponent": 0.25,
            "kernel_width": 25,
        }

        assert result == expected
