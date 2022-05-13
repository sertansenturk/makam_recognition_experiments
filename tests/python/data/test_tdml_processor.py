import copy

from unittest import mock
import pytest

import numpy as np
from mre.data.tdms_feature import TDMSFeature


DEFAULT_TIME_PITCH_CONFIDENCE = [
    [0.0, 400.0, 0.9],
    [1.0, 500.0, 0.5],
    [2.0, 300.0, 0.7],
    [3.0, 203.0, 0.6],
    [4.0, 249.0, 0.6],
    [5.0, 501.0, 0.9],
]
DEFAULT_EMBEDDING = np.array(
    [
        [0.0, 2.0, 4.0, 6.0, 8.0],
        [10.0, 12.0, 14.0, 16.0, 18.0],
        [20.0, 22.0, 24.0, 26.0, 28.0],
        [30.0, 32.0, 34.0, 36.0, 38.0],
        [40.0, 42.0, 44.0, 46.0, 48.0],
    ]
)
DEFAULT_PITCH_BINS = np.arange(0.0, 1200.0, 240.0)


class TestTDMLProcessor:
    def test_init(self):
        embedding = DEFAULT_EMBEDDING
        # should have 5 elements; same as embedding size in each dim
        pitch_bins = DEFAULT_PITCH_BINS
        ref_freq = 512  # hz
        step_size = 10  # cents
        time_delay_index = 0.5  # seconds
        compression_exponent = 0.5  # square root compression
        kernel_width = 20  # cents

        tdms = TDMSFeature(
            embedding,
            pitch_bins,
            ref_freq,
            step_size,
            time_delay_index,
            compression_exponent,
            kernel_width,
        )

        np.testing.assert_array_equal(tdms.embedding, embedding)
        np.testing.assert_array_equal(tdms.pitch_bins, pitch_bins)
        assert tdms.ref_freq == ref_freq
        assert tdms.step_size == step_size
        assert tdms.time_delay_index == time_delay_index
        assert tdms.compression_exponent == compression_exponent
        assert tdms.kernel_width == kernel_width

    def test_init_wrong_embedding_shape(self):
        embedding = np.array(
            [
                [0.0, 2.0, 4.0, 6.0, 8.0],
                [10.0, 12.0, 14.0, 16.0, 18.0],
            ]
        )
        dummy_pitch_bins = np.array([0, 400])

        with pytest.raises(
            ValueError,
            match=r"The embedding should be a square matrix. Shape: \(2, 5\)",
        ):
            _ = TDMSFeature(embedding, dummy_pitch_bins)

    @pytest.mark.parametrize(
        "embedding",
        [
            np.array([0.0, 2.0]),  # 1d
            np.array([[0.0, 2.0], [10.0, 12.0]])[..., np.newaxis],  # 3d
        ],
    )
    def test_init_wrong_embedding_ndim(self, embedding):
        dummy_pitch_bins = np.array([0, 400])

        with pytest.raises(
            ValueError,
            match=(
                "Only 2D phase space embeddings are supported. "
                f"# Dims: {embedding.ndim}"
            ),
        ):
            _ = TDMSFeature(embedding, dummy_pitch_bins)

    def test_init_pitch_bin_len_mismatch(self):
        embedding = DEFAULT_EMBEDDING
        # should have had 5 elements; same as embedding size in each dim
        pitch_bins = np.array([0, 400, 800])

        with pytest.raises(
            ValueError,
            match="The embedding size and pitch_bins length should be the same",
        ):
            _ = TDMSFeature(embedding, pitch_bins)

    @pytest.mark.parametrize(
        "pitch_bins",
        [
            np.array([0, 600])[..., np.newaxis],  # 2d
            np.array([0, 600])[..., np.newaxis][..., np.newaxis],  # 3d
        ],
    )
    def test_init_wrong_pitch_bin_ndim(self, pitch_bins):
        embedding = np.array([[0.0, 2.0], [10.0, 12.0]])

        with pytest.raises(
            ValueError,
            match=(
                r"pitch_bins should be a vector \(single dimension\). "
                f"# Dims: {pitch_bins.ndim}"
            ),
        ):
            _ = TDMSFeature(embedding, pitch_bins)

    def test_init_none_compression_exponent(self):
        embedding = DEFAULT_EMBEDDING
        pitch_bins = DEFAULT_PITCH_BINS

        tdms = TDMSFeature(embedding, pitch_bins, compression_exponent=None)

        assert tdms.compression_exponent == 1

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

        tdms = TDMSFeature(
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
        "compression_exponent",
        [0, None],
    )
    def test_compress_unit(self, compression_exponent):
        embedding = np.array(
            [
                [0, 16, 0],
                [1, 0, 0],
                [0, 0, 4],
            ]
        )
        dummy_pitch_bins = np.array([0.0, 400.0, 800.0])
        compression_exponent = None

        tdms = TDMSFeature(
            embedding=copy.deepcopy(embedding),
            pitch_bins=dummy_pitch_bins,
            compression_exponent=compression_exponent,
        )
        tdms.compress()

        expected = embedding
        np.testing.assert_array_equal(tdms.embedding, expected)

    def test_init_none_smoothing_exponent(self):
        embedding = DEFAULT_EMBEDDING
        pitch_bins = DEFAULT_PITCH_BINS

        tdms = TDMSFeature(embedding, pitch_bins, kernel_width=None)

        assert tdms.kernel_width == 0

    @pytest.mark.parametrize(
        "kernel_width",
        [1, 2, 2.5],
    )
    def test_smoothen(self, kernel_width):
        embedding = DEFAULT_EMBEDDING
        dummy_pitch_bins = DEFAULT_PITCH_BINS

        tdms = TDMSFeature(
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

    @pytest.mark.parametrize(
        "kernel_width",
        [0, None],
    )
    def test_smoothen_unit(self, kernel_width):
        embedding = DEFAULT_EMBEDDING
        dummy_pitch_bins = DEFAULT_PITCH_BINS

        tdms = TDMSFeature(
            embedding=copy.deepcopy(embedding),
            pitch_bins=dummy_pitch_bins,
            kernel_width=kernel_width,
        )
        tdms.smoothen()

        expected = embedding
        np.testing.assert_array_almost_equal(tdms.embedding, expected)

    def test_normalize(self):
        embedding = np.array([[0.0, 5.0, 10.0], [15.0, 20.0, 25.0], [30.0, 35.0, 40.0]])
        dummy_pitch_bins = np.array([0.0, 400.0, 800.0])

        tdms = TDMSFeature(embedding=embedding, pitch_bins=dummy_pitch_bins)
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
    @mock.patch("mre.data.tdms_feature.TDMSFeature.normalize")
    @mock.patch("mre.data.tdms_feature.TDMSFeature.smoothen")
    @mock.patch("mre.data.tdms_feature.TDMSFeature.compress")
    @mock.patch(
        "mre.data.tdms_feature.TDMSFeature._compute_phase_space_embedding",
        return_value=DEFAULT_EMBEDDING,
    )
    @mock.patch(
        "mre.data.tdms_feature.TDMSFeature._remove_silent_delay_coordinates",
        return_value="mock_nonsilent_delay_coord",
    )
    @mock.patch(
        "mre.data.tdms_feature.TDMSFeature._compute_delay_coordinates",
        return_value="mock_delay_coord",
    )
    @mock.patch(
        "mre.data.tdms_feature.TDMSFeature._compute_sample_delay_index",
        return_value="mock_sample_delay_index",
    )
    @mock.patch(
        "mre.data.tdms_feature.TDMSFeature._process_pitch",
        return_value=("mock_pitch_idx", DEFAULT_PITCH_BINS),
    )
    @mock.patch(
        "mre.data.tdms_feature.TDMSFeature._parse_hz_track",
        return_value="mock_hz_track",
    )
    def test_from_hz_pitch(
        self,
        mock_parse_hz_track,
        mock_process_pitch,
        mock_compute_sample_delay_index,
        mock_compute_delay_coordinates,
        mock_remove_silent_delay_coordinates,
        mock_compute_phase_space_embedding,
        mock_compress,
        mock_smoothen,
        mock_normalize,
        hz_track,
    ):
        manager = mock.Mock()
        manager.attach_mock(mock_parse_hz_track, "parse_hz_track")
        manager.attach_mock(mock_process_pitch, "process_pitch")
        manager.attach_mock(
            mock_compute_sample_delay_index, "compute_sample_delay_index"
        )
        manager.attach_mock(mock_compute_delay_coordinates, "compute_delay_coordinates")
        manager.attach_mock(
            mock_remove_silent_delay_coordinates, "remove_silent_delay_coordinates"
        )
        manager.attach_mock(
            mock_compute_phase_space_embedding, "compute_phase_space_embedding"
        )
        manager.attach_mock(mock_compress, "compress")
        manager.attach_mock(mock_smoothen, "smoothen")
        manager.attach_mock(mock_normalize, "normalize")

        ref_freq = 512  # hz
        step_size = 10  # cents
        time_delay_index = 2  # seconds
        compression_exponent = 0.5  # square root compression
        kernel_width = 20  # cents

        tdms = TDMSFeature.from_hz_pitch(
            hz_track,
            ref_freq,
            step_size,
            time_delay_index,
            compression_exponent,
            kernel_width,
        )

        # check obj variables, instead of mocking __init__
        assert tdms.ref_freq == ref_freq
        assert tdms.step_size == step_size
        assert tdms.time_delay_index == time_delay_index
        assert tdms.compression_exponent == compression_exponent
        assert tdms.kernel_width == kernel_width

        expected_calls = [
            mock.call.parse_hz_track(hz_track),
            mock.call.process_pitch("mock_hz_track", ref_freq, step_size),
            mock.call.compute_sample_delay_index("mock_hz_track", time_delay_index),
            mock.call.compute_delay_coordinates(
                "mock_pitch_idx", "mock_sample_delay_index"
            ),
            mock.call.remove_silent_delay_coordinates("mock_delay_coord"),
            mock.call.compute_phase_space_embedding(
                "mock_nonsilent_delay_coord", 5  # length of DEFAULT_PITCH_BINS
            ),
            mock.call.compress(),
            mock.call.smoothen(),
            mock.call.normalize(),
        ]
        assert manager.mock_calls == expected_calls

    @pytest.mark.parametrize(
        "hz_track",
        [
            DEFAULT_TIME_PITCH_CONFIDENCE,
            np.array(DEFAULT_TIME_PITCH_CONFIDENCE),
        ],
    )
    def test_parse_hz_track(self, hz_track):
        result = TDMSFeature._parse_hz_track(hz_track)

        expected = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)

        np.testing.assert_array_equal(result, expected)

    def test_parse_hz_track_omit_confidence(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)[:, :1]

        result = TDMSFeature._parse_hz_track(hz_track)
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
            TDMSFeature._parse_hz_track(hz_track)

    def test_parse_hz_track_wrong_shape(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE).T
        with pytest.raises(ValueError, match=r"The hz_track's shape is \(3, 6\)."):
            TDMSFeature._parse_hz_track(hz_track)

    def test_parse_hz_track_deviating_timestamps(self):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)
        hz_track[2, 0] = 2.5
        with pytest.raises(
            ValueError, match="The duration between the timestamps deviate too much"
        ):
            TDMSFeature._parse_hz_track(hz_track)

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

        result_idx, result_bins = TDMSFeature._process_pitch(
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

        result_idx, result_bins = TDMSFeature._process_pitch(
            hz_track, ref_freq, step_size
        )

        expected_pitch_idx = np.array([8, 0, 3, 8, 0, 0])
        expected_bins = np.arange(0, 1200, 100)

        np.testing.assert_array_equal(result_idx, expected_pitch_idx)
        np.testing.assert_array_equal(result_bins, expected_bins)

    @pytest.mark.parametrize(
        "time_delay_index,expected",
        [
            (0.51, 1),
            (1.0, 1),
            (1.9, 2),
            (2.0, 2),
            (3.2, 3),
            (3.6, 4),
        ],
    )
    def test_compute_sample_delay_index(self, time_delay_index, expected):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)

        result = TDMSFeature._compute_sample_delay_index(hz_track, time_delay_index)

        assert result == expected

    @pytest.mark.parametrize(
        "time_delay_index",
        [0.10, 0.499, 0.5],
    )
    def test_compute_time_delay_index_less_than_hop_size(self, time_delay_index):
        hz_track = np.array(DEFAULT_TIME_PITCH_CONFIDENCE)  # hop_size is 1 sec

        with pytest.raises(
            ValueError,
            match=("Cannot compute a positive sample delay index"),
        ):
            _ = TDMSFeature._compute_sample_delay_index(hz_track, time_delay_index)

    @pytest.mark.parametrize(
        "delay",
        [2.1, "not_int", 4.0],
    )
    def test_compute_non_int_delay_coordinates(self, delay):
        vector = np.array([0, 1, 2, 3, 4, 5])

        with pytest.raises(
            ValueError, match="Provide delay per coordinate as an integer"
        ):
            _ = TDMSFeature._compute_delay_coordinates(vector, delay)

    @pytest.mark.parametrize(
        "delay",
        [-1, 0],
    )
    def test_compute_non_positive_delay_coordinates(self, delay):
        vector = np.array([0, 1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="delay must be a positive integer"):
            _ = TDMSFeature._compute_delay_coordinates(vector, delay)

    @pytest.mark.parametrize(
        "delay",
        [6, 7],
    )
    def test_compute_delay_coordinates_delay_longer_than_vector(self, delay):
        vector = np.array([0, 1, 2, 3, 4, 5])

        with pytest.raises(
            ValueError, match="delay cannot be more than the vector length - 1."
        ):
            _ = TDMSFeature._compute_delay_coordinates(vector, delay)

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

        result = TDMSFeature._compute_delay_coordinates(vector, delay)

        np.testing.assert_array_equal(result, expected)

    def test_remove_silent_delay_coordinates(self):
        delay_coord = np.array(
            [[1.0, 0.0], [2.0, np.nan], [np.nan, 2.0], [np.nan, np.nan], [5.0, 4.0]]
        )

        result = TDMSFeature._remove_silent_delay_coordinates(delay_coord)

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
        result = TDMSFeature._compute_phase_space_embedding(delay_coord, tdms_size)

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
            _ = TDMSFeature._compute_phase_space_embedding(delay_coord, tdms_size)
