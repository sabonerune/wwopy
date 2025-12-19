from __future__ import annotations

import numpy as np

import wwopy


def test_cheaptrick(test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]):
    x, fs = test_wave
    temporal_positions, f0, _frame_period = wwopy.harvest(x, fs)
    wwopy.cheaptrick(x, fs, temporal_positions, f0)


def test_d4c(test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]):
    x, fs = test_wave
    temporal_positions, f0, _frame_period = wwopy.harvest(x, fs)
    _spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    wwopy.d4c(x, fs, temporal_positions, f0, fft_size)


def test_dio(test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]):
    x, fs = test_wave
    wwopy.dio(x, fs)


def test_stonemask(test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]):
    x, fs = test_wave
    temporal_positions, f0, _frame_period = wwopy.dio(x, fs)
    wwopy.stonemask(x, fs, temporal_positions, f0)


def test_synthesis(test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]):
    x, fs = test_wave
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    aperiodicity = wwopy.d4c(x, fs, temporal_positions, f0, fft_size)
    wwopy.synthesis(f0, spectrogram, aperiodicity, frame_period, fs)


def test_harvest(test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]):
    x, fs = test_wave
    wwopy.harvest(x, fs)


def test_realtimesynthesizer(
    test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int],
):
    x, fs = test_wave
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    aperiodicity = wwopy.d4c(x, fs, temporal_positions, f0, fft_size)
    synthesizer = wwopy.RealtimeSynthesizer(fs, frame_period, fft_size, 64, 1)
    synthesizer.append(f0, spectrogram, aperiodicity)
    y = np.empty(0, np.double)
    while (out := synthesizer.synthesis()) is not None:
        y = np.concatenate((y, out))
