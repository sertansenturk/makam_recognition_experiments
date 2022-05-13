from typing import Union

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

from tomato.converter import Converter


class TDMSProcessor:
    NUM_CENTS_IN_OCTAVE = 1200
    MIN_FREQ = 20
    TIMESTAMP_DEVIATION_TOL = 10e-6  # 1 microsecond
    NUM_DELAY_COORDINATES = 2

    def __init__(
        self,
        embedding: np.ndarray,
        pitch_bins: np.ndarray,
        ref_freq: float = 440.0,
        step_size: float = 7.5,
        time_delay_index: float = 0.3,
        compression_exponent: float = 0.25,
        kernel_width: float = 7.5,
    ) -> None:
        self.embedding = np.array(embedding)  # force numpy array
        self.pitch_bins = np.array(pitch_bins)  # force numpy array

        # TODO: validate surface and bins

        self.ref_freq = ref_freq
        self.step_size = step_size
        self.time_delay_index = time_delay_index
        self.compression_exponent = compression_exponent
        self.kernel_width = kernel_width

    def compress(self):
        self.embedding = np.power(self.embedding, self.compression_exponent)

    def smoothen(self):
        self.embedding = ndimage.gaussian_filter(
            self.embedding, sigma=self.kernel_width
        )

    def normalize(self):
        self.embedding = self.embedding / self.embedding.sum()

    @classmethod
    def from_hz_pitch(
        cls,
        hz_track: np.ndarray,
        ref_freq=440.0,
        step_size=7.5,
        time_delay_index=0.3,
        compression_exponent=0.25,
        kernel_width=7.5,
    ):
        """Implements time delayed melody surface computation

        Original code: https://github.com/sankalpg/Library_PythonNew/blob/076b521c020c7d0afe7239f457dc9d67074b256d/melodyProcessing/phaseSpaceEmbedding.py # noqa: E501

        Args:
            melody (_type_): _description_
            ref_freq (_type_): _description_
            kernel_width (_type_): _description_
            step_size (_type_): _description_
            time_delay_index
        """
        hz_track = cls._parse_hz_track(hz_track)

        pitch_idx, pitch_bins = cls._process_pitch(hz_track, ref_freq, step_size)

        sample_delay_index = cls._compute_sample_delay_index(hz_track, time_delay_index)

        delay_coordinates = cls._compute_delay_coordinates(
            pitch_idx, sample_delay_index
        )

        non_silent_delay_coordinates = cls._remove_silent_delay_coordinates(
            delay_coordinates
        )

        phase_space_embedding = cls._compute_phase_space_embedding(
            non_silent_delay_coordinates, len(pitch_bins)
        )

        tdms = cls(
            phase_space_embedding,
            pitch_bins,
            ref_freq,
            step_size,
            time_delay_index,
            compression_exponent,
            kernel_width,
        )

        tdms.compress()
        tdms.smoothen()
        tdms.normalize()

        return tdms

    @classmethod
    def _parse_hz_track(cls, hz_track: Union[list, np.array]):
        hz_track = np.array(hz_track)  # force to numpy array

        if hz_track.ndim != 2:
            raise ValueError(
                f"The hz_track has {hz_track.ndim} dimensions. It should have "
                "2 dimensions of shape Nx2; [timestamp, pitch, optional:confidence]."
            )

        if hz_track.shape[1] > 3:  # maybe the data is flipped
            raise ValueError(
                f"The hz_track's shape is {hz_track.shape}. It should be "
                "of shape Nx2; [timestamp, pitch, optional:confidence]."
            )

        max_absolute_timestamp_deviation = np.max(
            np.abs(np.diff(np.diff(hz_track[:, 0])))
        )
        if max_absolute_timestamp_deviation > cls.TIMESTAMP_DEVIATION_TOL:
            raise ValueError(
                "The duration between the timestamps deviate too much from each other."
            )

        return hz_track

    @classmethod
    def _process_pitch(
        cls, hz_track: np.ndarray, ref_freq: float, step_size: float
    ) -> np.ndarray:
        """Converts pitch track in Hz scale to indices discretized in cent scale
        with respect to the reference frequency and the step size. Also converts any
        silent/non-audible value to np.nan

        Example:
            # GIVEN
            hz_track = np.array(440, 219, 500, 882)
            ref_freq = 440  # hz
            step_size = 100  # cents

            pitch_idx, pitch_bins = TDMSProcessor._process_pitch(
                hz_track, ref_freq, step_size)

            # THEN
            pitch_idx: np.array([0, 0, 5, 0])
            pitch_bins: np.array([0, 100, 200, ... 1100])

            pitch_idx[3] is mapped pitch_bins[pitch_idx[3]] => 400+1200N cents above the
            ref_freq, where N is any integer.

        Args:
            hz_track (np.array): Nx3 array, with timestamp, pitch and (optionally)
            confidence , only the pitch column is used
            ref_freq (float): Reference frequency to normalize the pitch values to
            cent scale
            step_size (float): Interval in cents to discretize the pitch values into.
            1st bin is centered around 0 cents

        Returns:
            np.ndarray: octave-wrapped, discretized pitch indices wrt ref_freq
            np.ndarray: mapping of each index to the discrete bin in cents wrt reference
            frequency
        """
        cent_pitch = Converter.hz_to_cent(
            hz_track[:, 1], ref_freq=ref_freq, min_freq=cls.MIN_FREQ
        )
        discretized_pitch = np.round(cent_pitch / step_size)

        # WARNING: don't convert to astype(int) as it will corrupt np.nan's
        octave_wrapped_pitch_idx = np.mod(
            discretized_pitch, cls.NUM_CENTS_IN_OCTAVE / step_size
        )
        pitch_bins = np.arange(0, cls.NUM_CENTS_IN_OCTAVE, step_size)

        return octave_wrapped_pitch_idx, pitch_bins

    @classmethod
    def _compute_sample_delay_index(cls, hz_track, time_delay_index):
        hop_size_sec = hz_track[1, 0] - hz_track[0, 0]
        return int(round(time_delay_index / hop_size_sec))

    @classmethod
    def _compute_delay_coordinates(cls, vector: np.ndarray, delay: int):
        """
        Extracts phase space embedding given an input vector and delay per
        coordinates (t) in samples for two delay coordinates
        """
        if not isinstance(delay, int):
            raise ValueError("Provide delay per coordinate as an integer (# samples).")

        if delay < 1:
            raise ValueError("delay must be a positive integer")

        if delay > len(vector) - 1:
            raise ValueError("delay cannot be more than the vector length - 1.")

        total_delay = (cls.NUM_DELAY_COORDINATES - 1) * delay
        n_samples = len(vector)
        delay_coordinates = np.zeros(
            (n_samples - total_delay, cls.NUM_DELAY_COORDINATES)
        )
        for ii in range(cls.NUM_DELAY_COORDINATES):
            delay_coordinates[:, ii] = vector[
                total_delay - (ii * delay): n_samples - (ii * delay)
            ]

        return delay_coordinates

    @staticmethod
    def _remove_silent_delay_coordinates(delay_coordinates):
        return delay_coordinates[~np.isnan(delay_coordinates).any(axis=1)]

    @staticmethod
    def _compute_phase_space_embedding(delay_coordinates, tdms_size):
        delay_coordinates = delay_coordinates.astype(int)
        if np.max(delay_coordinates) >= tdms_size:
            raise ValueError(
                "Largest index in delay coordinates could not be equal or "
                "more than the TDMS size."
            )

        tdms = np.zeros((tdms_size, tdms_size))
        for dc in delay_coordinates:
            tdms[dc[0], dc[1]] += 1

        return tdms

    def plot(self):
        ax = plt.imshow(self.embedding, cmap="rainbow")
        plt.colorbar()

        return ax

    def to_json(self):
        pass
