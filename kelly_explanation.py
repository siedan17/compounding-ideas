import numpy as np
import matplotlib.pyplot as plt

# Parameters
p = 0.7   # Probability of winning
a = 1.5   # Fractional gain on a win
b = 0.95   # Fractional loss on a loss

# Define the expected log-growth function:
# G(f) = p*ln(1+a*f) + (1-p)*ln(1 - b*f)
def growth_rate(f):
    return p * np.log(1 + a*f) + (1 - p) * np.log(1 - b*f)

# Compute the Kelly-optimal fraction using the derived formula:
f_star = (p*a - (1-p)*b) / (a*b)

# We'll plot f from 0 to 1.
# (Note: betting f > 1 can exceed your entire bankroll;
#  also, 1 - b*f must stay >= 0 for a log to make sense.)
f_values = np.linspace(0, 1, 200)
G_values = growth_rate(f_values)

# Plot
plt.figure(figsize=(8, 6))
plt.plot(f_values, G_values, label='G(f)')

# Mark the Kelly fraction with a vertical dashed line
plt.axvline(x=f_star, color='red', linestyle='--',
            label=f'Kelly Fraction = {f_star:.2f}')

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

plt.xlabel('Fraction of Bankroll (f)')
plt.ylabel('G(f)')
plt.ylim(top = 0.3, bottom=-0.2)
plt.title('Kelly Criterion Growth Rate')
plt.grid(True)
plt.legend(loc='best', framealpha=1)
# plt.show()

plt.savefig('kelly_curve.png', dpi=300, bbox_inches='tight', format='png')
plt.close()
