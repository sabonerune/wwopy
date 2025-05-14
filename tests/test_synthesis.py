import numpy as np

import wwopy


def test_synthesis():
    fft_size = 2048
    array_len = fft_size // 2 + 1
    empty_f0 = np.empty(0, np.double)
    empty_spectrogram = np.empty((0, array_len), np.double)
    empty_aperiodicity = np.empty((0, array_len), np.double)
    y = wwopy.synthesis(empty_f0, empty_spectrogram, empty_aperiodicity, 5.0, 44100)
    assert y.dtype == np.double
    assert y.shape == (0,)
