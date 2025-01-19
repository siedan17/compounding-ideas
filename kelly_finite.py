import numpy as np
import matplotlib.pyplot as plt

def simulate_kelly(n, p=0.55, T=1000, f_values=None, random_seed=None):
    """
    Simulate a coin-toss betting game for various fractions of wealth.

    Parameters:
    -----------
    n : int
        Number of coin tosses in each simulation.
    p : float
        Probability of winning a single coin toss. (Default=0.55)
    T : int
        Number of simulations for each fraction f. (Default=10,000)
    f_values : array-like
        Array of fractions of wealth to bet each toss.
        If None, it defaults to np.linspace(0, 1, 21).
    random_seed : int
        Optional random seed for reproducibility.

    Returns:
    --------
    f_values, median_wealths, p5_wealths
        Arrays of fraction values, median final wealth, and 5th percentile wealth
        after n tosses, respectively.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if f_values is None:
        # By default, consider fractions from 0 to 1 in 21 steps
        f_values = np.linspace(0, 1, 21)

    # Preallocate arrays
    median_wealths = np.zeros_like(f_values)
    p5_wealths     = np.zeros_like(f_values)

    # Run simulation for each fraction f
    for i, f in enumerate(f_values):
        final_wealth = np.zeros(T)

        # Simulate T paths
        for t in range(T):
            w = 1.0  # start wealth
            # Run n coin tosses
            for _ in range(n):
                # Generate random coin toss
                if np.random.rand() < p:
                    # Win => multiply wealth by (1+f)
                    w *= (1 + f)
                else:
                    # Lose => multiply wealth by (1-f)
                    w *= (1 - f)
            final_wealth[t] = w

        # Compute median & 5th percentile of final wealth
        median_wealths[i] = np.median(final_wealth)
        p5_wealths[i]     = np.percentile(final_wealth, 5)

    return f_values, median_wealths, p5_wealths

def plot_all_kelly_results(n_list, results):
    _, axes = plt.subplots(len(n_list), 1, figsize=(10, 6 * len(n_list)))
    for ax, n in zip(axes, n_list):
        f_values, median_wealth, p5_wealth = results[n]

        # Find the fraction that maximizes the median and 5th percentile
        idx_median_max = np.argmax(median_wealth)
        idx_p5_max = np.argmax(p5_wealth)

        f_opt_median = f_values[idx_median_max]
        f_opt_p5 = f_values[idx_p5_max]

        # Plot curves
        ax.plot(f_values, median_wealth, label='Median Final Wealth', marker='o')
        ax.plot(f_values, p5_wealth, label='5th Percentile Wealth', marker='o', linestyle='--')

        # Mark optima
        ax.scatter(f_opt_median, median_wealth[idx_median_max], color='red', zorder=5,
                    label=f'Median Opt (f={f_opt_median:.2f})')
        ax.scatter(f_opt_p5, p5_wealth[idx_p5_max], color='green', zorder=5,
                    label=f'5th Perc Opt (f={f_opt_p5:.2f})')

        # Add vertical dashed lines for optima
        ax.axvline(f_opt_median, color='red', linestyle='--', label='Median Opt Line')
        ax.axvline(f_opt_p5, color='green', linestyle='--', label='5th Perc Opt Line')

        ax.set_title(f'Kelly Criterion Simulation (n={n})')
        ax.set_xlabel('Fraction of Wealth Bet (f)')
        ax.set_ylabel('Final Wealth (log scale)')
        ax.set_yscale('log')
        ax.legend(loc="lower left")
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    n_list = [100, 200, 400, 800, 1600, 3200]
    seed = 42
    results = {}
    for n in n_list:
        f_vals, med_wealth, p5_wealth = simulate_kelly(
            n=n,
            p=0.65,
            T=1000,
            f_values=np.linspace(0, 0.4, 31),
            random_seed=seed
        )
        results[n] = (f_vals, med_wealth, p5_wealth)

    plot_all_kelly_results(n_list, results)
