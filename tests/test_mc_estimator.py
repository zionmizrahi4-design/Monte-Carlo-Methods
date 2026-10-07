import numpy as np
from monte_carlo import monte_carlo


def test_simple_mean():
    samples = np.array([1, 2, 3])
    f = lambda x : x

    result = monte_carlo(samples, f)

    assert result.estimate == 2