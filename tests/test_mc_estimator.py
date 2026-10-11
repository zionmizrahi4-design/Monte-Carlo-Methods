import numpy as np
import pytest

from monte_carlo import MonteCarloResult, monte_carlo


SEED = 42


def test_simple_mean():
    samples = np.array([1, 2, 3])
    expected_mean = 2.0

    result = monte_carlo(samples, lambda x: x)

    assert result.estimate == expected_mean


def test_applies_function():
    samples = np.array([1, 2, 3])
    expected_mean = 14 / 3

    result = monte_carlo(samples, lambda x: x**2)

    assert result.estimate == pytest.approx(expected_mean)


def test_standard_error():
    samples = np.array([1, 2, 3])
    expected_se = 1 / np.sqrt(3)

    result = monte_carlo(samples, lambda x: x)

    assert result.standard_error == pytest.approx(expected_se)


def test_constant_function():
    samples = np.array([1, 2, 3, 4])
    constant = 7

    result = monte_carlo(samples, lambda x: constant)

    assert result.estimate == constant
    assert result.standard_error == 0
    assert result.confidence_interval == pytest.approx([constant, constant])


def test_result_type_and_metadata():
    samples = np.array([1, 2, 3])
    confidence = 0.99

    result = monte_carlo(samples, lambda x: x, confidence)

    assert isinstance(result, MonteCarloResult)
    assert result.sample_size == 3
    assert result.confidence_level == confidence

#run pytest sevral times for each input 
@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1])
def test_invalid_confidence_raises(confidence):
    samples = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        monte_carlo(samples, lambda x: x, confidence)


def test_too_few_samples_raises():
    samples = np.array([1])

    with pytest.raises(ValueError):
        monte_carlo(samples, lambda x: x)

#run function several times for each input
@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_non_finite_function_output_raises(bad_value):
    samples = np.array([1, 2, 3])

    def f(x):
        return np.full(x.shape, bad_value)

    with pytest.raises(ValueError):
        monte_carlo(samples, f)


def test_wrong_output_shape_raises():
    samples = np.array([1, 2, 3])

    def f(x):
        return np.array([1, 2])

    with pytest.raises(ValueError):
        monte_carlo(samples, f)
