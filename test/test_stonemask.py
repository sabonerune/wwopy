import numpy as np

import wwopy


def test_empty():
    empty_x = np.empty(0, np.double)
    empty_temporal_positions = np.empty(0, np.double)
    empty_f0 = np.empty(0, np.double)
    f0 = wwopy.stonemask(empty_x, 44100, empty_temporal_positions, empty_f0)
    assert f0.dtype == np.double
    assert f0.shape == (0,)
