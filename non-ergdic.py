import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 200
p = 0.5
up_factor = 1.6
down_factor = 0.5

# Simulate one multiplicative path
Y = np.zeros(N+1)
Y[0] = 1.0
for t in range(1, N+1):
    if np.random.rand() < p:
        Y[t] = Y[t-1] * up_factor
    else:
        Y[t] = Y[t-1] * down_factor

# The ensemble average at step t: (E[M])^t
mean_M = p * up_factor + (1 - p) * down_factor
ensemble_avg = mean_M ** np.arange(N+1)

# Time average (arithmetic mean of the single path values up to step t)
time_avg = np.zeros(N+1)
for t in range(N+1):
    time_avg[t] = np.mean(Y[:t+1]) if t > 0 else Y[0]

# Plot all in one figure
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

steps = np.arange(N+1)

plt.plot(steps, Y, label='Single Realization Y(t)')
plt.plot(steps, time_avg,'r-', label='Time Average of Single Path Y(t)')
plt.plot(steps, ensemble_avg, color='g', linestyle='--', label=f'Ensemble Avg E[Y(t)] = {mean_M:.2f}^t')

plt.xlabel('Step')
plt.ylabel('Value')
plt.yscale('log')
plt.title('Non-Ergodic Process: Multiplicative Growth')
plt.legend(loc='upper left', framealpha=1)
plt.grid(True)

# plt.show()

# print(f"Final single-path value after {N} steps: {Y[-1]:.4f}")
# print(f"Final time average (arithmetic) of that path: {time_avg[-1]:.4f}")
# print(f"Ensemble average at step {N}: {ensemble_avg[-1]:.4f}")
# print("Observe how the single realization can drift below the ensemble average.")

plt.savefig('non-ergodic.png', dpi=300, bbox_inches='tight', format='png')
plt.close()
