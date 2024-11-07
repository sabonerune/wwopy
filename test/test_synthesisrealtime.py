from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import wwopy

if TYPE_CHECKING:
    import numpy.typing as npt


def test_multi_append(
    test_wave: tuple[npt.NDArray[np.double], int],
    dio_result: tuple[npt.NDArray[np.double], npt.NDArray[np.double], float],
    cheaptrick_result: tuple[npt.NDArray[np.double], int],
    d4c_result: npt.NDArray[np.double],
):
    x, fs = test_wave
    temporal_positions, f0, frame_period = dio_result
    spectrogram, fft_size = cheaptrick_result
    synthesizer = wwopy.RealtimeSynthesizer(fs, frame_period, fft_size, 64, 8)
    i = 0
    y = np.empty(0, np.double)
    while i < len(f0):
        if synthesizer.append(
            f0[i : i + 1], spectrogram[i : i + 1], d4c_result[i : i + 1]
        ):
            i += 1
        while (out := synthesizer.synthesis()) is not None:
            y = np.concatenate((y, out))
        if synthesizer.locked():
            break
