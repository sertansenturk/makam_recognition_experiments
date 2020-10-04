from pathlib import Path
from unittest import mock
import pytest

import numpy as np
import pandas as pd
from mre.data import PredominantMelodyNormalized


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


class TestPredominantMelodyNormalized():
    def test_transform_empty_melody_paths(self):
        # GIVEN
        melody_paths = []
        tonic_freqs = pd.Series({"id1": 400, "id2": 300})

        # WHEN; THEN
        pmn = PredominantMelodyNormalized()
        with pytest.raises(ValueError):
            pmn.transform(melody_paths, tonic_freqs)

    def test_transform_empty_tonic_freqs(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series({}, dtype=np.float)

        # WHEN; THEN
        pmn = PredominantMelodyNormalized()
        with pytest.raises(ValueError):
            pmn.transform(melody_paths,
                          tonic_freqs)

    def test_transform_duplicate_melody_paths(self):
        # GIVEN
        melody_paths = ["id1.npy", "id1.npy"]
        tonic_freqs = pd.Series({"id1": 400, "id2": 300})

        # WHEN; THEN
        pmn = PredominantMelodyNormalized()
        with pytest.raises(ValueError):
            pmn.transform(melody_paths, tonic_freqs)

    def test_transform_duplicate_tonic_ids(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series([400, 300], index=["id1", "id1"])

        # WHEN; THEN
        pmn = PredominantMelodyNormalized()
        with pytest.raises(ValueError):
            pmn.transform(melody_paths, tonic_freqs)

    def test_transform_mbid_mismatch(self):
        # GIVEN
        melody_paths = ["id1.npy", "id2.npy"]
        tonic_freqs = pd.Series([400, 300], index=["id1", "id3"])

        # WHEN; THEN
        pmn = PredominantMelodyNormalized()
        with pytest.raises(ValueError):
            pmn.transform(melody_paths, tonic_freqs)

    def test_transform(self, mock_tmp_dir):
        # GIVEN
        melody_paths = ["./path_to/id1.npy", "./path_to/id2.npy"]
        tonic_freqs = pd.Series([400, 100], index=["id1", "id2"])
        mock_melody = [np.array([[0, 100], [1, 200], [3, 50]])
                       for _ in melody_paths]

        # WHEN
        pmn = PredominantMelodyNormalized()
        with mock.patch("tempfile.TemporaryDirectory",
                        autospec=True,
                        return_value=mock_tmp_dir):
            with mock.patch('numpy.load',
                            autospec=True,
                            side_effect=mock_melody):
                with mock.patch('numpy.save',
                                autospec=True,
                                ) as mock_save:
                    pmn.transform(melody_paths, tonic_freqs)

        # THEN
        expected_melodies = [  # converted to cent scale by hand
            np.array([[0, -2400], [1, -1200], [3, -3600]]),
            np.array([[0, 0], [1, 1200], [3, -1200]])]
        expected_save_calls = [
            (Path(mock_tmp_dir.name, Path(pp).name), ep)
            for pp, ep in zip(melody_paths, expected_melodies)]

        for args, exp in zip(mock_save.call_args_list, expected_save_calls):
            assert args[0][0] == exp[0]
            np.testing.assert_array_equal(args[0][1], exp[1])

    def test_transform_existing_tmp_dir(self, mock_tmp_dir):
        # GIVEN
        melody_paths = ["./path_to/id1.npy"]
        tonic_freqs = pd.Series([400], index=["id1"])
        mock_melody = [np.array([[0, 100], [1, 200], [3, 50]])]
        pmn = PredominantMelodyNormalized()
        pmn.tmp_dir = mock_tmp_dir  # transform called before

        # WHEN
        with mock.patch.object(pmn,
                               '_cleanup',
                               autospec=True) as mock_cleanup:
            with mock.patch("tempfile.TemporaryDirectory",
                            autospec=True,
                            return_value=mock_tmp_dir):
                with mock.patch('numpy.load',
                                autospec=True,
                                side_effect=mock_melody):
                    with mock.patch('numpy.save',
                                    autospec=True):
                        pmn.transform(melody_paths, tonic_freqs)

        # THEN
        mock_cleanup.assert_called_once_with()

    def test_mlflow_tags(self):
        # GIVEN
        pmn = PredominantMelodyNormalized()

        # WHEN
        result = pmn._mlflow_tags()

        # THEN
        expected = {"min_freq": pmn.MIN_FREQ}

        assert result == expected
