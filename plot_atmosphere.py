#!/usr/bin/env python3
"""
Plot atmospheric height profiles from G2_1D.dat
"""

import matplotlib.pyplot as plt
import numpy as np

# Read atmospheric data
data = np.loadtxt("G2_1D.dat")

z = data[:, 0]  # height [cm]
rho = data[:, 1]  # density [g/cm^3]
p = data[:, 2]  # pressure [dyn/cm^2]
T = data[:, 3]  # temperature [K]

# Convert height to Mm (megameters)
z_mm = z / 1e8

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# Plot 1: Temperature
axes[0].plot(z_mm, T, "r-", linewidth=2)
axes[0].set_xlabel("Height [Mm]", fontsize=12)
axes[0].set_ylabel("Temperature [K]", fontsize=12)
axes[0].set_title("Temperature Profile", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, max(T) * 1.05)

# Plot 2: Density (log scale)
axes[1].semilogy(z_mm, rho, "b-", linewidth=2)
axes[1].set_xlabel("Height [Mm]", fontsize=12)
axes[1].set_ylabel("Density [g/cm³]", fontsize=12)
axes[1].set_title("Density Profile", fontsize=13, fontweight="bold")
axes[1].grid(True, alpha=0.3, which="both")

# Plot 3: Pressure (log scale)
axes[2].semilogy(z_mm, p, "g-", linewidth=2)
axes[2].set_xlabel("Height [Mm]", fontsize=12)
axes[2].set_ylabel("Pressure [dyn/cm²]", fontsize=12)
axes[2].set_title("Pressure Profile", fontsize=13, fontweight="bold")
axes[2].grid(True, alpha=0.3, which="both")

plt.suptitle("1D Atmospheric Model Profiles (G2_1D.dat)", fontsize=15, fontweight="bold", y=1.02)

plt.tight_layout()

# Save figure
output_file = "atmosphere_profiles.jpg"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"✓ Plot saved to {output_file}")

# Show statistics
print("\n═══ Atmospheric Model Statistics ═══")
print(f"Number of levels: {len(z)}")
print(f"\nHeight range:      {z.min() / 1e8:.2f} - {z.max() / 1e8:.2f} Mm")
print(f"Temperature range: {T.min():.1f} - {T.max():.1f} K")
print(f"Density range:     {rho.min():.2e} - {rho.max():.2e} g/cm³")
print(f"Pressure range:    {p.min():.2e} - {p.max():.2e} dyn/cm²")
print(f"\nLog10(P) range:    {np.log10(p.min()):.2f} - {np.log10(p.max()):.2f}")
print(f"Log10(ρ) range:    {np.log10(rho.min()):.2f} - {np.log10(rho.max()):.2f}")

plt.show()
