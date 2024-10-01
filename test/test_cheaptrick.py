import numpy as np

import wwopy


def test_emptry():
    empty_x = np.empty(0, np.double)
    empty_temporal_positions = np.empty(0, np.double)
    empty_f0 = np.empty(0, np.double)
    spectrogram, fft_size = wwopy.cheaptrick(
        empty_x, 44100, empty_temporal_positions, empty_f0
    )
    assert spectrogram.dtype == np.double
    assert spectrogram.shape == (0, fft_size // 2 + 1)
