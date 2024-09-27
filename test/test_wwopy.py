from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

import wwopy


def test_cheaptrick(test_wave: tuple[npt.NDArray[np.double], int]):
    x, fs = test_wave
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    wwopy.cheaptrick(x, fs, temporal_positions, f0)


def test_d4c(test_wave: tuple[npt.NDArray[np.double], int]):
    x, fs = test_wave
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    wwopy.d4c(x, fs, temporal_positions, f0, fft_size)


def test_dio(test_wave: tuple[npt.NDArray[np.double], int]):
    x, fs = test_wave
    wwopy.dio(x, fs)


def test_stonemask(test_wave: tuple[npt.NDArray[np.double], int]):
    x, fs = test_wave
    temporal_positions, f0, frame_period = wwopy.dio(x, fs)
    wwopy.stonemask(x, fs, temporal_positions, f0)


def test_synthesis(test_wave: tuple[npt.NDArray[np.double], int]):
    x, fs = test_wave
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    aperiodicity = wwopy.d4c(x, fs, temporal_positions, f0, fft_size)
    wwopy.synthesis(f0, spectrogram, aperiodicity, fft_size, frame_period, fs)


def test_harvest(test_wave: tuple[npt.NDArray[np.double], int]):
    x, fs = test_wave
    wwopy.harvest(x, fs)
