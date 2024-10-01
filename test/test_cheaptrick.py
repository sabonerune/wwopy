from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import wwopy

if TYPE_CHECKING:
    import numpy.typing as npt


def test_empty():
    empty_x = np.empty(0, np.double)
    empty_temporal_positions = np.empty(0, np.double)
    empty_f0 = np.empty(0, np.double)
    spectrogram, fft_size = wwopy.cheaptrick(
        empty_x, 44100, empty_temporal_positions, empty_f0
    )
    assert spectrogram.dtype == np.double
    assert spectrogram.shape == (0, fft_size // 2 + 1)


def test_fft_size(
    test_wave: tuple[npt.NDArray[np.double], int],
    dio_result: tuple[npt.NDArray[np.double], npt.NDArray[np.double], float],
):
    x, fs = test_wave
    fft_size = 4080
    temporal_positions, f0, frame_period = dio_result
    spectrogram, result_fft_size = wwopy.cheaptrick(
        x, fs, temporal_positions, f0, fft_size=fft_size
    )
    assert result_fft_size == fft_size


def test_warning(
    test_wave: tuple[npt.NDArray[np.double], int],
    dio_result: tuple[npt.NDArray[np.double], npt.NDArray[np.double], float],
):
    x, fs = test_wave
    temporal_positions, f0, frame_period = dio_result
    fft_size = 2048
    with pytest.warns(RuntimeWarning):
        spectrogram, result_fft_size = wwopy.cheaptrick(
            x, fs, temporal_positions, f0, f0_floor=72.0, fft_size=fft_size
        )
    assert result_fft_size == fft_size
