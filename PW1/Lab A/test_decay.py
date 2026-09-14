import pytest
import numpy as np
import decay

def test_decay_runs():
    res = decay.simulate(1000, 0.4)
    assert res[0] == 1000
    assert len(res) > 1

def test_negative_rate_raises_error():
    with pytest.raises(ValueError):
        decay.simulate(1000, -0.1)

def test_decay_average_approximation():
    N0 = 1000
    rate = 0.4
    res = decay.simulate(N0, rate)
    expected = N0 * np.exp(-rate)
    assert res[1] == pytest.approx(expected, rel=0.3)
