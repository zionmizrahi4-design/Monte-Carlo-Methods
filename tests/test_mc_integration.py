import numpy as np
import pytest 
import math

from monte_carlo import integrate

#DO NOT CHANGE (fixed for reproducibility) 
SEED = 42
SAMPLE_SIZE = 100000
CONFIDENCE = 0.95

def test_known_integral():
    f = lambda x : x**2
    lower_limit, upper_limit = 0, 1
    
    #test must pass under the fixed seed and sample_size
    result = integrate(f, lower_limit, upper_limit, SAMPLE_SIZE, CONFIDENCE, SEED)

    
    """
    Calculations (theoretical standard error):

    x~uniform(0,1) => p(x) = 1
    => E[f(x)] = int[-inf, inf](f(x)) * p(x))dx = int[0, 1](x^2 * 1)dx = 1/3
    also:
    E[x^4] = int[0, 1](x^4)dx = [x^5 / 5](0, 1) = 1/5
    => Var(f(x)) = E[f(x)^2] - (E[f(x)])^2 = 1/5 - 1/9 = (9-5) / 45 = 4/45

    SD(f(x)) = sqrt(Var(f(x)))

    monte carlo estimator:
    I_n = (b-a)/n * sigma(f(xi)) => var(I_n) = var((b-a)/n * sigma(f(xi))) =  1/n^2 * sigma(var(f(xi))) = (1/n^2) * n * 4/45 = 4/45n


    => SE(I_n) =  sqrt(4/45n)
    """
    true_value = 1/3
    theoretical_standard_error = math.sqrt(4/(45 * SAMPLE_SIZE))

    assert abs(true_value - result.estimate) < 2 * theoretical_standard_error 

def test_const_function_non_unit_intervals():
    f = lambda x : 10
    lower_limit, upper_limit = 0, 10
    result = integrate(f, lower_limit, upper_limit, SAMPLE_SIZE, CONFIDENCE, SEED)

    true_value = 100


    assert result.estimate == true_value
    assert result.standard_error == 0
    assert result.confidence_interval == (true_value, true_value)


@pytest.mark.parametrize("bad_function, expected_exception",
                        [
                            ("string", TypeError),
                            (5, TypeError),
                            (5.0, TypeError),
                            (lambda x: np.full(x.shape, np.nan), ValueError)
                        ]
                        )
def test_bad_function_input(bad_function, expected_exception):
    lower_limit, upper_limit = 0, 1
    with pytest.raises(expected_exception):
        integrate(bad_function, lower_limit, upper_limit, SAMPLE_SIZE, CONFIDENCE, SEED)



def test_too_large_intervals():
    f = lambda x: x**2
    lower, upper = -1e308, 1e308

    with pytest.raises(ValueError):
        integrate(f, lower, upper, SAMPLE_SIZE, CONFIDENCE, SEED)


@pytest.mark.parametrize("lower, upper",
                         [
                            (np.inf, 1),
                            (1, np.inf),
                            (np.nan, np.nan),
                            (np.inf, np.inf),
                            (-np.inf, 1),
                            (1, np.nan)
                         ]
                         )
def test_invalid_integral_limits(lower, upper):
    f = lambda x: x**2
    with pytest.raises(ValueError):
        integrate(f, lower, upper, SAMPLE_SIZE, CONFIDENCE, SEED)



def test_degenerate_case():
    f = lambda x: x**2
    lower = 1
    upper = lower

    result = integrate(f, lower, upper, SAMPLE_SIZE, CONFIDENCE, SEED)

    assert result.estimate == 0.0
    assert result.standard_error == 0.0
    assert result.confidence_interval == [0.0 ,0.0]
    assert result.sample_size == 0
    assert result.confidence_level == CONFIDENCE

def test_reversed_limits():
    f = lambda x : 10
    lower_limit, upper_limit = 10, 0
    result = integrate(f, lower_limit, upper_limit, SAMPLE_SIZE, CONFIDENCE, SEED)

    true_value = -100


    assert result.estimate == true_value
    assert result.standard_error == 0
    assert result.confidence_interval == (true_value, true_value)


@pytest.mark.parametrize(
    "sample_size, expected_exception",
    [
        (1, ValueError),
        (0, ValueError),
        (-5, ValueError),
        (10.5, TypeError),
        ("100", TypeError),
        (True, TypeError),
    ],
)
def test_invalid_sample_size(sample_size, expected_exception):
    f = lambda x: x**2
    lower, upper = 0, 1
    with pytest.raises(expected_exception):
        integrate(f,lower, upper, sample_size, CONFIDENCE, SEED)


@pytest.mark.parametrize(
    "confidence",
    [0, 1, -0.1, 1.1],
)
def test_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        integrate(
            lambda x: x**2,
            0,
            1,
            SAMPLE_SIZE,
            confidence,
            SEED,
        )

def test_same_seed_is_reproducible():
    f = lambda x: x**2

    result_1 = integrate(f, 0, 1, SAMPLE_SIZE, CONFIDENCE, SEED)
    result_2 = integrate(f, 0, 1, SAMPLE_SIZE, CONFIDENCE, SEED)

    assert result_1.estimate == result_2.estimate
    assert result_1.standard_error == result_2.standard_error
    assert result_1.confidence_interval == result_2.confidence_interval