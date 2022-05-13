import copy

import pytest

import numpy as np
from mre.data.tdms_processor import TDMSProcessor


DEFAULT_TIME_PITCH_CONFIDENCE = [
    [0.0, 400.0, 0.9],
    [1.0, 500.0, 0.5],
    [2.0, 300.0, 0.7],
    [3.0, 203.0, 0.6],
    [4.0, 249.0, 0.6],
    [5.0, 501.0, 0.9],
]


class TestTDMLProcessor:
    def test_init(self):
        pass

    def test_compress(self):
        embedding = np.array(
            [
                [0, 16, 0],
                [1, 0, 0],
                [0, 0, 4],
            ]
        )
        dummy_pitch_bins = np.array([0.0, 400.0, 800.0])
        compression_exponent = 0.5

        tdms = TDMSProcessor(
            embedding=embedding,
            pitch_bins=dummy_pitch_bins,
            compression_exponent=compression_exponent,
        )
        tdms.compress()

        expected = np.array(
            [
                [0.0, 4.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        )
        np.testing.assert_array_equal(tdms.embedding, expected)

    @pytest.mark.parametrize(
        "kernel_width",
        [1, 2, 2.5],
    )
    def test_smoothen(self, kernel_width):
        embedding = np.array(
            [
                [0.0, 2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, 14.0, 16.0, 18.0],
                [20.0, 22.0, 24.0, 26.0, 28.0],
                [30.0, 32.0, 34.0, 36.0, 38.0],
                [40.0, 42.0, 44.0, 46.0, 48.0],
            ]
        )
        dummy_pitch_bins = np.arange(0.0, 1200.0, 240.0)

        tdms = TDMSProcessor(
            embedding=embedding,
            pitch_bins=dummy_pitch_bins,
            kernel_width=kernel_width,
        )
        tdms.smoothen()

        # the actual implementation is tested thoroughly in scipy,
        # here we are just checking if the embedding is "smoothened"
        assert np.sum(embedding) == pytest.approx(np.sum(tdms.embedding), 10e-6)
        assert np.min(embedding) < np.min(tdms.embedding)
        assert np.max(embedding) > np.max(tdms.embedding)

    def test_normalize(self):
        embedding = np.array([[0.0, 5.0, 10.0], [15.0, 20.0, 25.0], [30.0, 35.0, 40.0]])
        dummy_pitch_bins = np.array([0.0, 400.0, 800.0])

        tdms = TDMSProcessor(embedding=embedding, pitch_bins=dummy_pitch_bins)
        tdms.normalize()

        expected = embedding / 180

        np.testing.assert_array_almost_equal(tdms.embedding, expected)

    @pytest.mark.parametrize(
        "hz_track",
        [
            DEFAULT_TIME_PITCH_CONFIDENCE,
            np.array(DEFAULT_TIME_PITCH_CONFIDENCE),
        ],
    )
    def test_from_hz_pitch(self, hz_track):
        pass

    @pytest.mark.parametrize(
        "hz_track",
        [
            DEFAULT_TIME_PITCH_CONFIDENCE,
            np.array(DEFAULT_TIME_PITCH_CONFIDENCE),
        ],
    )
    def test_parse_hz_track(self, hz_track):
        result = TDMSProcessor._parse_hz_track(hz_track)

        expected = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)

        np.testing.assert_array_equal(result, expected)

    def test_parse_hz_track_omit_confidence(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)[:, :1]

        result = TDMSProcessor._parse_hz_track(hz_track)
        expected = copy.deepcopy(np.array(DEFAULT_TIME_PITCH_CONFIDENCE)[:, :1])

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "hz_track",
        [
            np.array(DEFAULT_TIME_PITCH_CONFIDENCE)[:, 1],  # 1d
            np.array(DEFAULT_TIME_PITCH_CONFIDENCE)[..., np.newaxis],  # 3d
        ],
    )
    def test_parse_hz_track_wrong_dim(self, hz_track):
        with pytest.raises(
            ValueError, match=f"The hz_track has {hz_track.ndim} dimensions."
        ):
            TDMSProcessor._parse_hz_track(hz_track)

    def test_parse_hz_track_wrong_shape(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE).T
        with pytest.raises(ValueError, match=r"The hz_track's shape is \(3, 6\)."):
            TDMSProcessor._parse_hz_track(hz_track)

    def test_parse_hz_track_deviating_timestamps(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)
        hz_track[2, 0] = 2.5
        with pytest.raises(
            ValueError, match="The duration between the timestamps deviate too much"
        ):
            TDMSProcessor._parse_hz_track(hz_track)

    def test_process_pitch_non_silent(self):
        hz_track = np.array(
            [
                [0.0, None, 0.9],
                [1.0, np.nan, 0.5],
                [2.0, 0, 0.5],
                [4.0, 19.99, 0.7],
                [5.0, 20.0, 0.7],  # boundary condition; still np.nan
                [6.0, 20.1, 0.7],
            ]
        )
        ref_freq = 160  # hz
        step_size = 100  # cents

        result_idx, result_bins = TDMSProcessor._process_pitch(
            hz_track, ref_freq, step_size
        )

        expected_pitch_idx = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.0])
        expected_bins = np.arange(0, 1200, 100)

        np.testing.assert_array_equal(result_idx, expected_pitch_idx)
        np.testing.assert_array_equal(result_bins, expected_bins)

    def test_process_pitch(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)
        ref_freq = 250.0  # hz
        step_size = 100  # cents

        result_idx, result_bins = TDMSProcessor._process_pitch(
            hz_track, ref_freq, step_size
        )

        expected_pitch_idx = np.array([8, 0, 3, 8, 0, 0])
        expected_bins = np.arange(0, 1200, 100)

        np.testing.assert_array_equal(result_idx, expected_pitch_idx)
        np.testing.assert_array_equal(result_bins, expected_bins)

    @pytest.mark.parametrize(
        "time_delay_index,expected",
        [
            (1.0, 1),
            (1.9, 2),
            (2.0, 2),
            (3.2, 3),
            (3.6, 4),
        ],
    )
    def test_compute_sample_delay_index(self, time_delay_index, expected):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)

        result = TDMSProcessor._compute_sample_delay_index(hz_track, time_delay_index)

        assert result == expected

    @pytest.mark.parametrize(
        "delay",
        [2.1, "not_int", 4.0],
    )
    def test_compute_non_int_delay_coordinates(self, delay):
        vector = np.array([0, 1, 2, 3, 4, 5])

        with pytest.raises(
            ValueError, match="Provide delay per coordinate as an integer"
        ):
            _ = TDMSProcessor._compute_delay_coordinates(vector, delay)

    @pytest.mark.parametrize(
        "delay",
        [-1, 0],
    )
    def test_compute_non_positive_delay_coordinates(self, delay):
        vector = np.array([0, 1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="delay must be a positive integer"):
            _ = TDMSProcessor._compute_delay_coordinates(vector, delay)

    @pytest.mark.parametrize(
        "delay",
        [6, 7],
    )
    def test_compute_delay_coordinates_delay_longer_than_vector(self, delay):
        vector = np.array([0, 1, 2, 3, 4, 5])

        with pytest.raises(
            ValueError, match="delay cannot be more than the vector length - 1."
        ):
            _ = TDMSProcessor._compute_delay_coordinates(vector, delay)

    @pytest.mark.parametrize(
        "delay,expected",
        [
            # shift by 1
            (1, np.array([[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]])),
            (2, np.array([[2, 0], [3, 1], [4, 2], [5, 3]])),  # shift by 2
            (5, np.array([[5, 0]])),  # shift by 5
        ],
    )
    def test_compute_delay_coordinates(self, delay, expected):
        vector = np.array([0, 1, 2, 3, 4, 5])

        result = TDMSProcessor._compute_delay_coordinates(vector, delay)

        np.testing.assert_array_equal(result, expected)

    def test_remove_silent_delay_coordinates(self):
        delay_coord = np.array(
            [[1.0, 0.0], [2.0, np.nan], [np.nan, 2.0], [np.nan, np.nan], [5.0, 4.0]]
        )

        result = TDMSProcessor._remove_silent_delay_coordinates(delay_coord)

        expected = np.array([[1.0, 0.0], [5.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "delay_coord",
        [
            np.array([[1.0, 0.0], [2.0, 2.0], [2.0, 2.0], [2.0, 3.0]]),
            np.array([[1, 0], [2, 2], [2, 2], [2, 3]]),
        ],
    )
    def test_compute_phase_space_embedding(self, delay_coord):
        tdms_size = 5
        result = TDMSProcessor._compute_phase_space_embedding(delay_coord, tdms_size)

        expected = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize(
        "tdms_size",
        [2, 3],
    )
    def test_compute_phase_space_embedding_small_tdms_size(self, tdms_size):
        delay_coord = np.array([[1, 0], [2, 2], [2, 2], [2, 3]])
        with pytest.raises(
            ValueError,
            match=(
                "Largest index in delay coordinates could not be equal or "
                "more than the TDMS size."
            ),
        ):
            _ = TDMSProcessor._compute_phase_space_embedding(delay_coord, tdms_size)
