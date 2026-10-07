import numpy as np
from MonteCarloResult import MonteCarloResult
import MC_Estimation as mc

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
def integrate (f, lower_limit, upper_limit, num_of_samples, confidence, seed = None):
    rng = np.random.default_rng(seed)
    samples = rng.uniform(lower_limit, upper_limit, num_of_samples)#TODO: validate 

    #calc expecred value part of equation
    mc_result = mc.monte_carlo(samples, f, confidence) 

    #calc integral properties:
    i_estimation = mc_result.estimate * (upper_limit - lower_limit)
    i_se = (upper_limit - lower_limit) * mc_result.standard_error
    i_interval = mc.calc_interval(confidence, i_se, i_estimation)
    i_sample_size = num_of_samples

    result = MonteCarloResult(i_estimation, i_se, i_interval, i_sample_size)
    
    return result