"""Run from the repository root: experiments/.venv/bin/python -m unittest discover -s tests -v.

Seeded statistical checks use generous tolerances rather than requiring every
confidence interval to contain the true mean. A valid interval can miss it.
"""

import unittest

import numpy as np

from experiments.montecarlo import calc_interval, calc_se, mc


class TestMonteCarlo(unittest.TestCase):
    def test_small_known_sample(self):
        # Mean = 2; unbiased sample variance = 1; standard error = sqrt(1/3).
        result = mc(np.array([1.0, 2.0, 3.0]), lambda x: x)
        mean, se, lower, upper = result
        self.assertEqual(mean, 2.0)
        self.assertAlmostEqual(se, np.sqrt(1 / 3))
        margin = 1.959963984540054 * np.sqrt(1 / 3)
        self.assertAlmostEqual(lower, 2.0 - margin)
        self.assertAlmostEqual(upper, 2.0 + margin)

    def test_constant_function(self):
        result = mc(np.arange(10.0), lambda x: np.full_like(x, 7.0))
        np.testing.assert_allclose(result, [7.0, 0.0, 7.0, 7.0])

    def test_two_samples(self):
        mean, se, lower, upper = mc(np.array([0.0, 2.0]), lambda x: x)
        self.assertEqual(mean, 1.0)
        self.assertEqual(se, 1.0)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)

    def test_known_population_expectations(self):
        rng = np.random.default_rng(45)
        n = 100_000
        cases = [
            ("uniform squared", rng.uniform(0, 1, n), lambda x: x**2,
             1 / 3, 4 / 45),
            ("normal squared", rng.normal(size=n), lambda x: x**2,
             1.0, 2.0),
            ("bernoulli", rng.binomial(1, 0.3, n), lambda x: x,
             0.3, 0.21),
        ]
        for name, samples, f, expected_mean, population_variance in cases:
            with self.subTest(case=name):
                mean, se, _, _ = mc(samples, f)
                theoretical_se = np.sqrt(population_variance / n)
                self.assertLess(abs(mean - expected_mean), 5 * theoretical_se)
                self.assertAlmostEqual(se / theoretical_se, 1.0, delta=0.03)

    def test_unknown_expectation_consistency(self):
        # No exact population expectation is supplied for this nonlinear f.
        samples = np.random.default_rng(123).normal(size=2000)
        f = lambda x: np.sin(x**2) * np.exp(-np.abs(x))
        values = f(samples)
        mean, se, lower, upper = mc(samples, f)
        self.assertTrue(np.all(np.isfinite([mean, se, lower, upper])))
        self.assertGreaterEqual(mean, values.min())
        self.assertLessEqual(mean, values.max())
        self.assertGreater(se, 0)
        self.assertAlmostEqual(mean, values.mean())
        # Independent NumPy reference for the finite-sample standard error.
        self.assertAlmostEqual(se, values.std(ddof=1) / np.sqrt(values.size))
        self.assertAlmostEqual(mean - lower, upper - mean)

        # Permutation preserves results; affine transformations have known effects.
        shuffled = mc(samples[::-1], f)
        np.testing.assert_allclose(shuffled, [mean, se, lower, upper])
        transformed = mc(samples, lambda x: -3 * f(x) + 2)
        np.testing.assert_allclose(
            transformed, [-3 * mean + 2, 3 * se, -3 * upper + 2, -3 * lower + 2]
        )

    def test_higher_confidence_widens_interval(self):
        samples = np.arange(20.0)
        low = mc(samples, lambda x: x, confidence=0.90)
        high = mc(samples, lambda x: x, confidence=0.99)
        np.testing.assert_allclose(low[:2], high[:2])
        self.assertLess(high[2], low[2])
        self.assertGreater(high[3], low[3])

    def test_helper_functions(self):
        self.assertAlmostEqual(calc_se(np.array([1.0, 2.0, 3.0]), 2.0),
                               np.sqrt(1 / 3))
        np.testing.assert_allclose(calc_interval(0.95, 1.0, 0.0),
                                   [-1.959963984540054, 1.959963984540054])

    def test_invalid_confidence(self):
        for confidence in [0, 1, -0.1, 1.1, np.nan]:
            with self.subTest(confidence=confidence):
                with self.assertRaisesRegex(ValueError, "Confidence level"):
                    mc(np.array([1.0, 2.0]), lambda x: x, confidence)

    def test_insufficient_samples(self):
        for samples in [np.array([]), np.array([1.0])]:
            with self.subTest(size=samples.size):
                with self.assertRaisesRegex(ValueError, "At least 2"):
                    mc(samples, lambda x: x)

    def test_nonfinite_function_values(self):
        for invalid in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=invalid):
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    mc(np.array([0.0, 1.0]), lambda x: np.array([0.0, invalid]))


if __name__ == "__main__":
    unittest.main()
