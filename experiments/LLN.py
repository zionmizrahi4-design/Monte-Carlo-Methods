"""Illustrate the law of large numbers by plotting sample running means."""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Reject invalid sample counts, seeds, and distribution parameters.

    Report invalid input through parser.error, which exits with status 2.
    """
    if args.n <= 0:
        parser.error("--n must be positive")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    if args.distribution == "normal":
        if not math.isfinite(args.mu):
            parser.error("--mu must be finite")
        if not math.isfinite(args.sigma) or args.sigma <= 0:
            parser.error("--sigma must be finite and positive")
    elif args.distribution == "uniform":
        if not math.isfinite(args.low) or not math.isfinite(args.high):
            parser.error("--low and --high must be finite")
        if args.low >= args.high:
            parser.error("--low must be smaller than --high")
        if not math.isfinite(args.high - args.low):
            parser.error("--high minus --low must be finite")
    elif args.distribution == "bernoulli":
        if not math.isfinite(args.p) or not 0 <= args.p <= 1:
            parser.error("--p must be finite and between 0 and 1")


def parse_args() -> argparse.Namespace:
    """Parse command-line options and return the checked argparse namespace."""
    parser = argparse.ArgumentParser(
        description="Illustrate the law of large numbers with running sample means."
    )

    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--seed", type=int, default=42, help="Nonnegative random seed (default: 42)"
    )
    common_args.add_argument("--n", type=int, default=1000, help="Number of samples")

    subparsers = parser.add_subparsers(
        dest="distribution",
        required=True,
    )

    normal = subparsers.add_parser(
        "normal",
        parents=[common_args],
    )
    normal.add_argument(
        "--mu", type=float, default=0.0, help="Mean of the normal distribution"
    )
    normal.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Standard deviation of the normal distribution",
    )

    uniform = subparsers.add_parser(
        "uniform",
        parents=[common_args],
    )
    uniform.add_argument(
        "--low", type=float, default=0.0, help="Lower bound of the uniform distribution"
    )
    uniform.add_argument(
        "--high", type=float, default=1.0, help="Upper bound of the uniform distribution"
    )

    bernoulli = subparsers.add_parser(
        "bernoulli",
        parents=[common_args],
    )
    bernoulli.add_argument(
        "--p",
        type=float,
        default=0.5,
        help="Probability of success for the Bernoulli distribution",
    )

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def calc_expected_value(args: argparse.Namespace) -> float:
    """Return the theoretical mean for the selected distribution.

    Raise ValueError if the distribution name is unsupported.
    """
    if args.distribution == "normal":
        return args.mu
    elif args.distribution == "uniform":
        return args.low / 2 + args.high / 2
    elif args.distribution == "bernoulli":
        return args.p
    else:
        raise ValueError("Unknown distribution")


def get_samples(args: argparse.Namespace) -> np.ndarray:
    """Return a one-dimensional array of samples using the configured seed.

    Create a new NumPy generator on each call, so repeated calls with the
    same seed and distribution parameters reproduce the same samples.
    Raise ValueError if the distribution name is unsupported.
    """
    rng = np.random.default_rng(args.seed)

    if args.distribution == "normal":
        samples = rng.normal(args.mu, args.sigma, args.n)

    elif args.distribution == "uniform":
        samples = rng.uniform(args.low, args.high, args.n)

    elif args.distribution == "bernoulli":
        samples = rng.binomial(1, args.p, args.n)
    else:
        raise ValueError(f"Unsupported distribution: {args.distribution}")

    return samples


def main() -> None:
    """Sample the selected distribution and plot its running and theoretical means."""
    args = parse_args()
    expected_value = calc_expected_value(args)
    samples = get_samples(args)

    sample_counts = np.arange(1, len(samples) + 1)
    cumulative_sum = np.cumsum(samples)
    cumulative_mean = cumulative_sum / sample_counts

    if args.distribution == "normal":
        distribution_label = f"Normal (mean={args.mu:g}, std={args.sigma:g})"
    elif args.distribution == "uniform":
        distribution_label = f"Uniform (low={args.low:g}, high={args.high:g})"
    else:
        distribution_label = f"Bernoulli (p={args.p:g})"

    _, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.plot(sample_counts, cumulative_mean, label="Running sample mean")
    ax.axhline(
        y=expected_value,
        color="tab:orange",
        linestyle="--",
        label=f"Theoretical mean = {expected_value:g}",
    )
    ax.set_xlabel("Sample count (n)")
    ax.set_ylabel("Running sample mean")
    ax.set_title(
        f"Law of large numbers: {distribution_label}\n"
        f"Samples: {args.n:,} | Seed: {args.seed}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
