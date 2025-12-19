import numpy as np

import wwopy


def test_empty():
    empty_x = np.empty(0, np.double)
    temporal_positions, f0, _frame_period = wwopy.dio(empty_x, 44100)
    assert temporal_positions.dtype == np.double
    assert temporal_positions.shape == (0,)
    assert f0.dtype == np.double
    assert f0.shape == (0,)
