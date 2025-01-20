import numpy as np
import matplotlib.pyplot as plt

###############################################################################
#               EXPLICIT SIMULATION PARAMETERS (TWEAK AS NEEDED)              #
###############################################################################
# Probability of winning each coin toss
P = 0.55

# Multipliers for the fraction f bet:
# Win => multiply wealth by (1 + a*f)
# Lose => multiply wealth by (1 - b*f)
a = 0.85
b = 0.65

# Number of simulations (Monte Carlo paths) per fraction
T = 2000

# List of number of coin tosses to explore
N_LIST = [100, 200, 400, 800, 1600, 3200]

# Range of fractions of wealth to bet
F_VALUES = np.linspace(0, 0.8, 51)

# Random seed for reproducibility (set to None for different runs each time)
SEED = 42

###############################################################################
#                           KELLY ANALYTICAL FRACTION                          #
###############################################################################
def kelly_fraction(p, a, b):
    """
    Compute the Kelly fraction for the scenario:
    Win => multiply by (1 + a*f)
    Lose => multiply by (1 - b*f)

    Maximizes p * ln(1 + a f) + (1-p) * ln(1 - b f).

    f_Kelly = [p * a - (1 - p) * b] / (a * b)
    """
    # The raw formula:
    f_k = (p * a - (1 - p) * b) / (a * b)
    return f_k

###############################################################################
#                       KELLY SIMULATION (DISTRIBUTION) FUNCTION               #
###############################################################################
def simulate_kelly_distribution(n, p, T, f_values, a=1.0, b=1.0, random_seed=None):
    """
    Simulate a coin-toss betting game for various fractions of wealth.
    Final wealth is multiplied by (1 + a*f) on a win, or (1 - b*f) on a loss.

    Parameters
    ----------
    n : int
        Number of coin tosses in each simulation.
    p : float
        Probability of winning a single coin toss.
    T : int
        Number of simulated paths for each fraction f.
    f_values : array-like
        Array of fractions of wealth to bet each toss.
    a : float
        Win multiplier coefficient.
    b : float
        Loss multiplier coefficient.
    random_seed : int or None
        Optional random seed for reproducibility.

    Returns
    -------
    f_values : np.ndarray
        Fractions of wealth bet.
    p10_wealths : np.ndarray
        10th percentile of final wealth for each fraction f.
    p50_wealths : np.ndarray
        50th percentile (median) of final wealth for each fraction f.
    p90_wealths : np.ndarray
        90th percentile of final wealth for each fraction f.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    p10_wealths = np.zeros_like(f_values)
    p50_wealths = np.zeros_like(f_values)
    p90_wealths = np.zeros_like(f_values)

    for i, f in enumerate(f_values):
        final_wealth = np.zeros(T)

        for t in range(T):
            w = 1.0  # start with wealth = 1
            for _ in range(n):
                # Coin toss
                if np.random.rand() < p:
                    w *= (1 + a * f)  # Win
                else:
                    w *= (1 - b * f)  # Lose
            final_wealth[t] = w

        # Compute the 10th, 50th, and 90th percentiles of final wealth
        p10_wealths[i] = (np.percentile(final_wealth, 5) +
                          np.percentile(final_wealth, 10) +
                          np.percentile(final_wealth, 15)) / 3
        p50_wealths[i] = (np.percentile(final_wealth, 45) +
                          np.percentile(final_wealth, 50) +
                          np.percentile(final_wealth, 55)) / 3
        p90_wealths[i] = (np.percentile(final_wealth, 85) +
                          np.percentile(final_wealth, 90) +
                          np.percentile(final_wealth, 95)) / 3

    return f_values, p10_wealths, p50_wealths, p90_wealths


###############################################################################
#                                 PLOTTING FUNCTION                            #
###############################################################################
def plot_kelly_results(n, f_values, p10, p50, p90, p, a, b):
    """
    Plot the 10th, 50th, and 90th percentile final wealth vs fraction f,
    along with the analytical Kelly fraction and the f-values that maximize
    each percentile.
    """
    # Discrete maxima
    idx_max_p10 = np.argmax(p10)
    idx_max_p50 = np.argmax(p50)
    idx_max_p90 = np.argmax(p90)

    f_opt_p10 = f_values[idx_max_p10]
    f_opt_p50 = f_values[idx_max_p50]
    f_opt_p90 = f_values[idx_max_p90]

    # Analytical Kelly fraction
    f_kelly = kelly_fraction(p, a, b)

    # Create a new figure for this value of n
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 15,
    'lines.linewidth': 2.5,
    'grid.linewidth': 1.5,
    'axes.linewidth': 1.5,
    })

    # Plot the percentiles
    plt.plot(f_values, p10, label='10th Percentile', color='blue', marker='o')
    plt.plot(f_values, p50, label='50th Percentile (Median)', color='green', marker='o')
    plt.plot(f_values, p90, label='90th Percentile', color='orange', marker='o')

    # Plot vertical lines for the discrete fractions that maximize each percentile
    plt.axvline(f_opt_p10, color='blue', linestyle='--',
                label=f'Max 10% @ f={f_opt_p10:.3f}')
    plt.axvline(f_opt_p50, color='green', linestyle='--',
                label=f'Max 50% @ f={f_opt_p50:.3f}')
    plt.axvline(f_opt_p90, color='orange', linestyle='--',
                label=f'Max 90% @ f={f_opt_p90:.3f}')

    # Plot the analytical Kelly fraction as a red dashed line (if it lies in [0,1], we still show it even if outside)
    plt.axvline(f_kelly, color='red', linestyle='--', label=f'Kelly @ f={f_kelly:.3f}')

    # Thicker horizontal line at y = 0 in linear scale
    plt.axhline(1, color='black', linewidth=2)

    plt.title(f'Kelly Simulation (t={n}, p={p:.2f}, a={a}, b={b})')
    plt.xlabel('Fraction of Wealth Bet (f)')
    plt.ylabel('Final Wealth (log scale)')
    plt.yscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', framealpha=1)
    plt.tight_layout()

    # plt.show()
    plt.savefig(f'kelly_sim_{n}.png', dpi=300, bbox_inches='tight', format='png')


###############################################################################
#                                MAIN SCRIPT                                  #
###############################################################################
if __name__ == "__main__":
    for n in N_LIST:
        f_vals, p10_vals, p50_vals, p90_vals = simulate_kelly_distribution(
            n=n,
            p=P,
            T=T,
            f_values=F_VALUES,
            a=a,
            b=b,
            random_seed=SEED
        )
        plot_kelly_results(n, f_vals, p10_vals, p50_vals, p90_vals, P, a, b)
