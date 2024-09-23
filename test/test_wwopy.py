from __future__ import annotations

import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING

if sys.version_info >= (3, 11):
    from typing import assert_never

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

import wwopy

_TEST_FILE = Path(__file__).parents[1] / "ext/World/test/vaiueo2d.wav"


def _read_wave() -> tuple[npt.NDArray[np.double], int]:
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


def test_cheaptrick():
    x, fs = _read_wave()
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    wwopy.cheaptrick(x, fs, temporal_positions, f0)


def test_d4c():
    x, fs = _read_wave()
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    wwopy.d4c(x, fs, temporal_positions, f0, fft_size)


def test_dio():
    x, fs = _read_wave()
    wwopy.dio(x, fs)


def test_stonemask():
    x, fs = _read_wave()
    temporal_positions, f0, frame_period = wwopy.dio(x, fs)
    wwopy.stonemask(x, fs, temporal_positions, f0)


def test_synthesis():
    x, fs = _read_wave()
    temporal_positions, f0, frame_period = wwopy.harvest(x, fs)
    spectrogram, fft_size = wwopy.cheaptrick(x, fs, temporal_positions, f0)
    aperiodicity = wwopy.d4c(x, fs, temporal_positions, f0, fft_size)
    wwopy.synthesis(f0, spectrogram, aperiodicity, fft_size, frame_period, fs)


def test_harvest():
    x, fs = _read_wave()
    wwopy.harvest(x, fs)
