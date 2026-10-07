class MonteCarloResult:
    def __init__(self, estimate, standard_error, confidence_interval, n_samples):
        self.estimate = estimate
        self.standard_error = standard_error
        self.confidence_interval = confidence_interval
        self.sample_size = n_samples