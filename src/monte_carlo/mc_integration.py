import numpy as np
from monte_carlo_result import MonteCarloResult
import mc_estimation as mc
import math


def integrate (f, lower_limit, upper_limit, num_of_samples, confidence=0.95, seed=None):
    """
    Goal here is to demonstrate the simple usage of the monte carlo method by estimating integrals:

    function will estimate given integrals using uniform distribution's expected value of f(x) by the following estimator:
    In = ((b-a) / n) * sigma[i=1 -> i =n](f(x_i))      (derived straight from the ev formula of f(x))
    The estimator above is unbiased for the requested integral.

    input: 
        f - function to integrate
        lower_limit, upper limit - limits of the integral
        number of samples to generate
        requested confidence lvl
        seed (for reproductibility, None for random)

    output:
        MonteCarloResult object
    """
    #validate input:
    sgn = 1
    if (not (math.isfinite(lower_limit) and math.isfinite(upper_limit))) or (not math.isfinite(upper_limit - lower_limit)):
        raise ValueError("integral limits contain non finite values or interval too large!")
    if not callable(f):
        raise TypeError("f must be callable")
    if isinstance(num_of_samples, (bool, np.bool_)) or not isinstance(num_of_samples, (int, np.integer)):
        raise TypeError("num_of_samples must be an integer")
    if num_of_samples < 2:
        raise ValueError("At least 2 samples are required")
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1")

    if upper_limit == lower_limit: #degenerate case
        return MonteCarloResult(0.0, 0.0, [0.0, 0.0], confidence, 0)
    if upper_limit < lower_limit: 
        lower_limit, upper_limit = upper_limit, lower_limit
        sgn = -1
    
    
    


    rng = np.random.default_rng(seed)
    samples = rng.uniform(lower_limit, upper_limit, num_of_samples)

    #calc expecred value part of equation
    mc_result = mc.monte_carlo(samples, f, confidence) 

    #calc integral properties:
    i_estimation = sgn * mc_result.estimate * (upper_limit - lower_limit)
    i_se = (upper_limit - lower_limit) * mc_result.standard_error
    i_interval = mc.calc_interval(confidence, i_se, i_estimation)
    i_sample_size = num_of_samples

    result = MonteCarloResult(i_estimation, i_se, i_interval,confidence, i_sample_size)
    
    return result

