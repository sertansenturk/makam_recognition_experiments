from dataclasses import dataclass

from tomato.converter import Converter


@dataclass
class TDMSProcessor:
    pass

    def from_hz_pitch(
        melody,
        ref_freq,
        kernel_width,
        step_size,
    ):
        """Implements time delayed melody surface computation

        Original code: https://github.com/sankalpg/Library_PythonNew/blob/076b521c020c7d0afe7239f457dc9d67074b256d/melodyProcessing/phaseSpaceEmbedding.py # pylint: disable-msg=E501

        Args:
            melody (_type_): _description_
            ref_freq (_type_): _description_
            kernel_width (_type_): _description_
            step_size (_type_): _description_
        """
        melody._fetch_melody_convert_to_cent(ref_freq)

        pass

    def _fetch_melody_convert_to_cent(melody, ref_freq):
        # melody_cent = Converter.hz_to_cent(melody[:, 1], ref_freq)
        pass

    def to_json(self):
        pass
