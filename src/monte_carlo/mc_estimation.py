from scipy.stats import norm
import numpy as np

from .monte_carlo_result import MonteCarloResult

def monte_carlo(samples, f, confidence=0.95):
    
    f_samples = np.asarray(f(samples))

    if f_samples.ndim == 0:
        f_samples = np.full(samples.shape, f_samples)

    if f_samples.shape != samples.shape:
        raise ValueError("f(samples) must return one value per sample")

    if not np.all(np.isfinite(f_samples)):
        raise ValueError("f(samples) contains non-finite values")

    if not np.all(np.isfinite(f_samples)):
        raise ValueError("f(samples) contains non-finite values")
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    if len(f_samples) < 2:
        raise ValueError("At least 2 samples are required")
    
    mc_value = np.mean(f_samples)
    se = calc_se(f_samples, mc_value)
    lower_bound, upper_bound = calc_interval(confidence, se, mc_value)
    interval = [lower_bound, upper_bound]

    result = MonteCarloResult(mc_value, se, interval, confidence,  len(f_samples))

    return result

def calc_interval(confidence, se, mc_value):
    z_crit_value = norm.ppf((1 + confidence) / 2)
    return mc_value - z_crit_value * se, mc_value + z_crit_value * se

def calc_se(f_samples, mc_value):
    n = len(f_samples)
    s = np.sqrt(1/(n- 1) * ((f_samples - mc_value)**2).sum())
    return s/np.sqrt(n)

def test_mc():

    """Usage example: estimate the integral of x^2 between 0 and 1 with confidence of 99%"""
    seed = 45
    n = 100000
    confidence = 0.99

    #Create samples and function
    rng = np.random.default_rng(seed)
    samples = rng.uniform(0, 1, n)
    f = lambda x : x**2


    result = monte_carlo(samples, f, confidence)


if __name__ == "__main__":
    test_mc()