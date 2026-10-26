
"""
Experiment — Monte Carlo Convergence
disclaimer: used codex to style and refine graphs.

Objective:
    Investigate and measure the change in accuracy of Monte Carlo integration 
    as the number of random samples increases.

Measurments used:
    function: f(x) = x^2
    lower bound = 0
    upper bound = 1
    random seed
    rising sample sizes
    confidence level of 95%

Graph 1: Convergence estimation
    x-axis: sample size 
    y-axis: integral estimation
    purpose: validate weather the estimation converges into the true value (1/3)

Graph 2: Absolute error
    x-axis: sample size
    y-axis: absolute error
    purpose: observe estimation error convergemce 

Repeated experiments graphs:
Graph 3: Integral historgram
    x-axis: estimated integral
    y-axis: number of estimations
    purpose: demonstrate CLT

Graph 4: RMSE
    x-axis: RMSE
    y-axis: number of samples
    purpose: validate RMSE converges in the same rate as theoretical one //todo: log-log graph

Graph 5:

"""



from monte_carlo import integrate, MonteCarloResult
import matplotlib.pyplot as plt
import numpy as np


F  = lambda x: x**2
UPPER = 1
LOWER = 0
THEORETICAL_VALUE = 1/3                                 #integral value 
THEORETICAL_VARIANCE = 4 / 45                           #Var(f(x))

SEED = int(np.random.default_rng().integers(0, 2**32))  #Use the same random seed throught the experiment
SAMPLE_SIZES = list(range(10, 10000, 10))
CONFIDENCE = 0.95

REPEATED_EXPERIMENTS_SAMPLE_SIZE = 10000
NUMBER_OF_EXPERIMENTS = 10000


#Used for both rmse and ci success rate graphs
RMSE_SAMPLE_SIZE = [10, 30, 100, 1000, 3000, 10000]
RMSE_REPETITION = 1000


def main():
    #Single experiment 
    experiments_results = run_experiments()
    visualize_convergence_estimation(experiments_results)
    visualize_absolute_error(experiments_results)

    #Repeated experiments
    experiments_results = run_repeated_experiments(REPEATED_EXPERIMENTS_SAMPLE_SIZE, NUMBER_OF_EXPERIMENTS)
    visualize_estimate_distribution(experiments_results)

    #calc rmse and ci success rate for repeated experiments
    theoretical_rmse = [calculate_theoretical_rmse(n) for n in RMSE_SAMPLE_SIZE]
    empirical_rmse = []
    ci_rates = []
    for sample_size in RMSE_SAMPLE_SIZE:
        experiments_results = run_repeated_experiments(sample_size, RMSE_REPETITION)
        empirical_rmse.append(calculate_rmse(experiments_results))
        ci_rates.append(calculate_confidence_coverage_rate(experiments_results))

    visualize_rmse(RMSE_SAMPLE_SIZE, empirical_rmse, theoretical_rmse)
    visualize_confidence_coverage(RMSE_SAMPLE_SIZE, ci_rates)

    

#graph 1
def visualize_convergence_estimation(results: list[MonteCarloResult]):
    x_axis = SAMPLE_SIZES
    y_axis = [result.estimate for result in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Monte Carlo estimates
    ax.plot(
        x_axis,
        y_axis,
        color="royalblue",
        linewidth=1,
        alpha=0.8,
        label="Monte Carlo Estimate"
    )

    # Theoretical value
    ax.axhline(
        y=THEORETICAL_VALUE,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Theoretical Value ({THEORETICAL_VALUE:.4f})"
    )

    # Labels and title
    ax.set_title(
        "Monte Carlo Integration — Convergence",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Estimated Integral", fontsize=12)

    # Appearance
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


def visualize_absolute_error(results: list[MonteCarloResult]):
    sample_sizes = [result.sample_size for result in results]
    absolute_errors = [
        abs(result.estimate - THEORETICAL_VALUE)
        for result in results
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        sample_sizes,
        absolute_errors,
        color="royalblue",
        linewidth=1.2,
        alpha=0.85,
        label="Absolute Error"
    )

    ax.set_title(
        "Monte Carlo Integration — Absolute Error",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Absolute Error", fontsize=12)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True, which="major", linestyle="--", alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", alpha=0.12)

    ax.legend(loc="upper right", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    plt.show()


#visualizing CLT, expecting to see given the current input: In ~ N(1/3, 4/45n)
def visualize_estimate_distribution(results: list[MonteCarloResult]):

    estimates = [result.estimate for result in results]
    estimates_np_arr = np.array(estimates)
    empirical_mean = estimates_np_arr.mean()

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        estimates,
        bins=20,
        color="royalblue",
        alpha=0.55,
        edgecolor="white",
        label="Monte Carlo Estimates",
        density=True
    )

    sigma = np.sqrt(((UPPER - LOWER)**2) * THEORETICAL_VARIANCE / REPEATED_EXPERIMENTS_SAMPLE_SIZE)
    x = np.linspace(
        min(min(estimates), THEORETICAL_VALUE - 4 * sigma),
        max(max(estimates), THEORETICAL_VALUE + 4 * sigma),
        1000,
    )
    normal_density = normal_pdf(x, THEORETICAL_VALUE, sigma)
    ax.fill_between(
        x,
        normal_density,
        color="darkorange",
        alpha=0.2,
        zorder=0,
    )
    ax.plot(
        x,
        normal_density,
        color="darkorange",
        linewidth=2.5,
        label="Theoretical Normal Distribution",
        zorder=3,
    )

    ax.axvline(
        x=THEORETICAL_VALUE,
        color="crimson",
        linestyle="--",
        linewidth=4,
        label=f"True Integral ({THEORETICAL_VALUE:.4f})"
    )

    ax.set_title(
        f"Monte Carlo Estimate Distribution (n={results[0].sample_size:,})",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Estimated Integral", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


    ax.text(
        0.02, 0.95,
        f"Empirical Mean: {empirical_mean:.6f} \nTrue value: {THEORETICAL_VALUE:.6}\nAbsolute error: {(empirical_mean - THEORETICAL_VALUE):.3}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )
    fig.tight_layout()
    plt.show()


# Graph 4 — RMSE Convergence
def visualize_rmse(
    sample_sizes: list[int],
    empirical_rmse: list[float],
    theoretical_rmse: list[float]
    ) -> None:

    fig, ax = plt.subplots(figsize=(12, 6))

    # Empirical RMSE
    ax.plot(
        sample_sizes,
        empirical_rmse,
        color="royalblue",
        linewidth=2,
        marker="o",
        markersize=5,
        alpha=0.85,
        label="Empirical RMSE"
    )

    # Theoretical RMSE
    ax.plot(
        sample_sizes,
        theoretical_rmse,
        color="darkorange",
        linewidth=2.5,
        linestyle="--",
        label="Theoretical RMSE"
    )

    # Labels and title
    ax.set_title(
        "Monte Carlo Integration — RMSE Convergence",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Root Mean Squared Error (RMSE)", fontsize=12)

    # Appearance
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(bottom=0)

    fig.tight_layout()
    plt.show()

# Graph 5 — Confidence Interval Coverage
def visualize_confidence_coverage(
    sample_sizes: list[int],
    coverage_rates: list[float]
) -> None:

    fig, ax = plt.subplots(figsize=(12, 6))

    # Empirical coverage rate
    ax.plot(
        sample_sizes,
        coverage_rates,
        color="royalblue",
        linewidth=2,
        marker="o",
        markersize=5,
        alpha=0.85,
        label="Empirical Coverage"
    )

    # Nominal confidence level
    ax.axhline(
        y=CONFIDENCE,
        color="crimson",
        linestyle="--",
        linewidth=2,
        label=f"Nominal Coverage ({CONFIDENCE:.0%})"
    )

    # Labels and title
    ax.set_title(
        "Monte Carlo Integration — Confidence Interval Coverage",
        fontsize=16,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Number of Samples (n)", fontsize=12)
    ax.set_ylabel("Coverage Rate (%)", fontsize=12)

    # Convert y-axis values to percentages
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda y, _: f"{y:.0%}")
    )

    # Appearance
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="best", frameon=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, 1)

    fig.tight_layout()
    plt.show()

#normal pdf function:  1 / sqrt(2pi) * e^(x-mu)^2 / 2 * sigma^2
def normal_pdf(x, mu, sigma):
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu)**2) / (2 * sigma**2)

    return coefficient * np.exp(exponent)


def calculate_confidence_coverage_rate(results: list[MonteCarloResult]) ->float:
    ci_success_count = sum(
        result.confidence_interval[0] <= THEORETICAL_VALUE <= result.confidence_interval[1]
        for result in results
    )
    return ci_success_count / len(results)


def calculate_rmse(results: list[MonteCarloResult]) -> float:
    squared_errors = [(result.estimate - THEORETICAL_VALUE)**2 for result in results]
    return np.sqrt(np.mean(squared_errors))


def calculate_theoretical_rmse(n: int) -> float:
    return np.sqrt((UPPER - LOWER)**2 * THEORETICAL_VARIANCE * (1 / n))


def run_experiments() -> list[MonteCarloResult]:
    estimations = []

    for sample_size in SAMPLE_SIZES:
            estimations.append(integrate(F, LOWER, UPPER, sample_size, CONFIDENCE, SEED))

    return estimations



def run_repeated_experiments(n: int, repetitions: int):
    experiments = []
    for i in range(0, repetitions):
        experiments.append(integrate(F, LOWER, UPPER, n, CONFIDENCE, None))#TODO: seed = None disables reproducbility (fix later)
    return experiments


if __name__ == "__main__":
    main()
