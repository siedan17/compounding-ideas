import numpy as np
import matplotlib.pyplot as plt

# Define a Maxwellian function (normalized to unity)
def maxwellian(E, T):
    # E in MeV, T in MeV.
    # f(E) = (2/sqrt(pi)) * (sqrt(E)/T^(3/2)) * exp(-E/T)
    return (2/np.sqrt(np.pi)) * (E**0.5 / T**1.5) * np.exp(-E/T)

# Temperature parameters chosen so that the mean energy (1.5*T) agrees with Los Alamos Primer estimates
T_u235 = 1.33  # gives mean ~2.0 MeV
T_pu239 = 1.40 # gives mean ~2.1 MeV
T_u238 = 1.53  # gives mean ~2.3 MeV

# Calculate the mean energies
mean_u235 = 1.5 * T_u235   # ~2.0 MeV
mean_pu239 = 1.5 * T_pu239 # ~2.1 MeV
mean_u238 = 1.5 * T_u238   # ~2.3 MeV

# Define energy grid from 0 to 4 MeV
E = np.linspace(0, 4, 1000)

# Compute Maxwellian spectra for each isotope
f_u235 = maxwellian(E, T_u235)
f_pu239 = maxwellian(E, T_pu239)
f_u238 = maxwellian(E, T_u238)

# Create the plot with the same color ordering as the cross section plot:
# U-235: blue, U-238: orange, Pu-239: green.
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(E, f_u235, label=r'U-235 $(n,f)$', lw=2, color='blue')
ax.plot(E, f_u238, label=r'U-238 $(n,f)$', lw=2, color='orange')
ax.plot(E, f_pu239, label=r'Pu-239 $(n,f)$', lw=2, color='green')

# Plot vertical lines for the means
ax.axvline(mean_u235, color='blue', linestyle='--', lw=1.5,
           label=r'U-235 mean = {:.1f} MeV'.format(mean_u235))
ax.axvline(mean_u238, color='orange', linestyle='--', lw=1.5,
           label=r'U-238 mean = {:.1f} MeV'.format(mean_u238))
ax.axvline(mean_pu239, color='green', linestyle='--', lw=1.5,
           label=r'Pu-239 mean = {:.1f} MeV'.format(mean_pu239))

# Set axis limits
ax.set_xlim([0, 4.5])  # Extend x-axis to 4.5 MeV for style
ax.set_ylim([0, 0.5])  # Adjust y-axis for clarity

# Set labels and title
ax.set_xlabel('Neutron energy (MeV)', fontsize=12)
ax.set_ylabel('Normalized Probability Density', fontsize=12)
ax.set_title('Synthetic Prompt Fission Neutron Spectra', fontsize=14)

# Add gridlines and legend
ax.grid(True, linestyle='--', linewidth=0.5)
ax.legend(fontsize=10)

plt.savefig('energy-distribution-emitted-neutrons.png', dpi=300, bbox_inches='tight', format='png')
plt.close()
