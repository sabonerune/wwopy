import numpy as np

import wwopy


def test_emptry():
    empty_x = np.empty(0, np.double)
    empty_temporal_positions = np.empty(0, np.double)
    empty_f0 = np.empty(0, np.double)
    fft_size = 2048
    aperiodicity = wwopy.d4c(
        empty_x, 44100, empty_temporal_positions, empty_f0, fft_size
    )
    assert aperiodicity.dtype == np.double
    assert aperiodicity.shape == (0, fft_size // 2 + 1)
