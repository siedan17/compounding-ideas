import numpy as np
import matplotlib.pyplot as plt

# Define the x-axis values (log10(E/eV)) spanning from -2 to 7.
# These are the lower edges of each decade point.
x = np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7])

# Revised synthetic dataset:
#
# For U-235 fission, based on typical thermal and fast-region behavior:
# (Thermal: ~585 barns ~ 5.85e-22 cm² at ~0.025 eV; then falling roughly as 1/v until about 1 keV,
# after which it settles in the fast region near ~1.5e-24 cm².)
# Here we use approximate log10(σ) values:
#   x=-2 (0.01 eV):   log10(σ) ≈ -21.10   => σ ≈ 10^(-21.10) cm²
#   x=-1 (0.1 eV):    log10(σ) ≈ -21.30
#   x=0  (1 eV):      log10(σ) ≈ -21.70
#   x=1  (10 eV):     log10(σ) ≈ -22.22
#   x=2  (100 eV):    log10(σ) ≈ -22.70
#   x=3  (1 keV):     log10(σ) ≈ -23.30
#   x=4  (10 keV):    log10(σ) ≈ -23.70
#   x=5  (100 keV):   log10(σ) ≈ -24.00
#   x=6  (1 MeV):     log10(σ) ≈ -23.82   => ~1.5e-24 cm²
#   x=7  (10 MeV):    log10(σ) ≈ -23.89   => ~1.3e-24 cm²
log_sigma_u235 = np.array([-21.10, -21.30, -21.70, -22.22, -22.70, -23.30, -23.70, -24.00, -23.82, -23.89])
sigma_u235 = 10 ** log_sigma_u235

# For Pu-239 fission, thermal values are higher (~750 barns, ~7.5e-22 cm²),
# and in the fast region it settles near ~3e-24 cm².
# We use:
#   x=-2: log10(σ) ≈ -21.05
#   x=-1: ≈ -21.16
#   x=0:  ≈ -21.52
#   x=1:  ≈ -22.00
#   x=2:  ≈ -22.40
#   x=3:  ≈ -23.00
#   x=4:  ≈ -23.40
#   x=5:  ≈ -23.70
#   x=6:  ≈ -23.52   => ~3e-24 cm²
#   x=7:  ≈ -23.55
log_sigma_pu239 = np.array([-21.05, -21.16, -21.52, -22.00, -22.40, -23.00, -23.40, -23.70, -23.52, -23.55])
sigma_pu239 = 10 ** log_sigma_pu239

# For U-238 fission, the cross section is negligible (set to a nominal value ~10^-30)
# in the thermal region (x <= 3), then starts rising in the fast region:
#   x=-2 to 3: use 1e-30 cm²,
#   x=4: say ~1e-29 cm²,
#   x=5: ~1e-27 cm²,
#   x=6 (1 MeV): log10(σ) ≈ -24.16   => ~0.7e-24 cm²,
#   x=7 (10 MeV): log10(σ) ≈ -24.10   => ~0.8e-24 cm².
sigma_u238 = np.array([1e-30, 1e-30, 1e-30, 1e-30, 1e-29, 1e-27, 10**(-26.16), 10**(-25.10), 10**(-24.16), 10**(-24.10)])
# (For x=6 and 7 we repeat the fast-region values.)

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the curves for each isotope; note that the curves are only defined up to x = 7,
# but we set the x-axis limit to 8 for styling.
ax.plot(x, sigma_u238, marker='^', color="green", linestyle='-', label='U-238')
ax.plot(x, sigma_u235, marker='o', color="blue", linestyle='-', label='U-235')
ax.plot(x, sigma_pu239, marker='s', color="orange", linestyle='-', label='Pu-239')

# Set x-axis and y-axis limits.
ax.set_xlim([-2, 8])
ax.set_ylim([1e-25, 2e-21])

# Set labels and title.
ax.set_xlabel(r'Energy in $\log_{10}$(eV)')
ax.set_ylabel(r'Fission cross section, $\sigma_f$ [cm$^2$]')
ax.set_title('Fission Cross Sections for U-235, Pu-239, and U-238')

# Use a logarithmic scale for the y-axis.
ax.set_yscale('log')

# Add gridlines and legend.
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend()

plt.savefig('cross-sections.png', dpi=300, bbox_inches='tight', format='png')
plt.close()

