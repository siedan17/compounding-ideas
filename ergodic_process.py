import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 100
p = 0.6  # Probability of +1

# Generate coin flips: +1 or -1
flips = np.where(np.random.rand(N) < p, 1, -1)

# Time average up to step t
time_avg = np.cumsum(flips) / np.arange(1, N+1)

ensemble_avg = 2*p - 1

# Plot everything in one figure
plt.figure(figsize=(10,6))
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

steps = np.arange(1, N+1)

plt.plot(steps, flips, '-', label=f'Coin Flip at Step t')
plt.plot(steps, time_avg, 'r-', label='Time Average up to t')
plt.axhline(ensemble_avg, color='g', linestyle='--', label=f'Ensemble Average = {ensemble_avg:.1f}')

plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Ergodic Process: Coin Flips')
plt.legend(loc="upper right", framealpha=1)
plt.grid(True)
# plt.show()

# print(f"Final time average after {N} flips: {time_avg[-1]:.3f}")
# print(f"Ensemble average of one flip: {ensemble_avg:.3f}")

plt.savefig('ergodic.png', dpi=300, bbox_inches='tight', format='png')
plt.close()

