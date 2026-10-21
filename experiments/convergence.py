
"""
Experiment — Monte Carlo Convergence
disclaimer: used condex to generate graphs

Objective:
    Investigate and measure how much does the accuracy of Monte Carlo integration changes
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
Graph 3:

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

REPEATED_EXPERIMENTS_SAMPLE_SIZE = 1000
NUMBER_OF_EXPERIMENTS = 1000


def main():
    #Single experiment 
    experiments_results = run_experiments()
    visualize_convergence_estimation(experiments_results)
    visualize_absolute_error(experiments_results)

    #Repeated experiments
    experiments_results = run_repeated_experiments(REPEATED_EXPERIMENTS_SAMPLE_SIZE, NUMBER_OF_EXPERIMENTS)
    visualize_estimate_distribution(experiments_results)
    

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
        bins=200,
        color="royalblue",
        alpha=0.55,
        edgecolor="white",
        label="Monte Carlo Estimates",
        density=True
    )

    sigma = np.sqrt(((UPPER - LOWER)**2) * THEORETICAL_VARIANCE * 1/ REPEATED_EXPERIMENTS_SAMPLE_SIZE)
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


#normal pdf function:  1 / sqrt(2pi) * e^(x-mu)^2 / 2 * sigma^2
def normal_pdf(x, mu, sigma):
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -((x - mu)**2) / (2 * sigma**2)

    return coefficient * np.exp(exponent)


def calculate_rmse(results: list[MonteCarloResult]) -> float:
    squared_errors = [(result.estimate - THEORETICAL_VALUE)**2 for result in results]
    return np.sqrt(np.mean(squared_errors))


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
