from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

import wwopy

if sys.version_info >= (3, 11):
    from typing import assert_never


_TEST_FILE = Path(__file__).parents[1] / "ext/World/test/vaiueo2d.wav"


@pytest.fixture
def test_wave() -> tuple[np.ndarray[tuple[int], np.dtype[np.double]], int]:
    with wave.open(str(_TEST_FILE), "rb") as f:
        f: wave.Wave_read
        nchannels = f.getnchannels()
        sampwidth = f.getsampwidth()
        framerate = f.getframerate()
        buffer = f.readframes(-1)
    if sampwidth == 1:
        dtype = np.dtype("<u1")
    elif sampwidth == 2:
        dtype = np.dtype("<i2")
    else:
        msg = f"sampwidth:{sampwidth} is not support."
        raise Exception(msg)
    data = np.frombuffer(buffer, dtype)
    data = data.astype(np.double)
    if sampwidth == 1:
        data -= 128
        data /= 128
    elif sampwidth == 2:
        data /= 2**15
    else:
        assert_never()
    assert data.max() <= 1
    assert data.min() >= -1
    if nchannels != 1:
        data = data.reshape((-1, nchannels))
    return data, framerate


@pytest.fixture
def dio_result(
    test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int],
) -> tuple[np.ndarray[tuple[int], np.dtype[np.double]], float]:
    x, fs = test_wave
    return wwopy.dio(x, fs)


@pytest.fixture
def cheaptrick_result(
    test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int],
    dio_result: tuple[np.ndarray[tuple[int], np.dtype[np.double]], float],
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.double]], int]:
    x, fs = test_wave
    temporal_positions, f0, frame_period = dio_result
    return wwopy.cheaptrick(x, fs, temporal_positions, f0)


@pytest.fixture
def d4c_result(
    test_wave: tuple[np.ndarray[tuple[int], np.dtype[np.double]], int],
    dio_result: tuple[np.ndarray[tuple[int], np.dtype[np.double]], float],
    cheaptrick_result: tuple[np.ndarray[tuple[int, int], np.dtype[np.double]], int],
) -> np.ndarray[tuple[int, int]]:
    x, fs = test_wave
    temporal_positions, f0, frame_period = dio_result
    spectrogram, fft_size = cheaptrick_result
    return wwopy.d4c(x, fs, temporal_positions, f0, fft_size)
