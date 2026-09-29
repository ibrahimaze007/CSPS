import pytest
import decay


def test_simulate_runs():
    res = decay.simulate(1000, 0.4)
    assert len(res) > 0
    assert res[0] == 1000


def test_negative_rate_raises_error():
    with pytest.raises(ValueError):
        decay.simulate(1000, -0.1)


def test_average_decay():
    import math
    N0 = 100000
    rate = 0.1
    result = decay.simulate(N0, rate)
    
    
    t_final = len(result) - 1
    expected = N0 * math.exp(-rate * t_final)
    
    
    assert result[-1] == pytest.approx(expected, abs=100)