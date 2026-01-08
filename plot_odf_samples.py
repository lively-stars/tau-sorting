#!/usr/bin/env python3
"""
Plot ODF (Opacity Distribution Function) at random T,P points

This script visualizes the opacity distribution functions from the NetCDF file
at 10 randomly selected temperature and pressure grid points.
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from pathlib import Path

# Read ODF data
print("Reading ODF data from ODF_nc_format.nc...")
with Dataset('ODF_nc_format.nc', 'r') as nc:
    # Read dimensions
    nt = nc.dimensions['nt'].size
    np_dim = nc.dimensions['np'].size
    nbins = nc.dimensions['nbins'].size
    nsubbins = nc.dimensions['nsubbins'].size
    
    # Read variables
    ODF = nc.variables['ODF'][:]  # [nt, np, nbins, nsubbins]
    FreqG = nc.variables['FreqG'][:]  # Frequency grid edges
    P = nc.variables['P'][:]
    T = nc.variables['T'][:]
    subbin = nc.variables['subbin'][:]
    
    vturb = nc.vturb if hasattr(nc, 'vturb') else None
    
    print(f"  ODF shape: {ODF.shape}")
    print(f"  Temperature range: {T.min():.1f} - {T.max():.1f} K")
    print(f"  Pressure range: {P.min():.2e} - {P.max():.2e} dyn/cm²")
    print(f"  Frequency bins: {nbins}, sub-bins: {nsubbins}")

# Convert ODF from short integer to actual opacity: kappa = 10^(ODF/1000)
# Select 10 random T,P points
np.random.seed(42)  # For reproducibility
n_samples = 10
random_t_indices = np.random.randint(0, nt, n_samples)
random_p_indices = np.random.randint(0, np_dim, n_samples)

print(f"\nSelected {n_samples} random (T, P) points:")
for i, (t_idx, p_idx) in enumerate(zip(random_t_indices, random_p_indices)):
    print(f"  {i+1}. T={T[t_idx]:.1f} K, P={P[p_idx]:.2e} dyn/cm², "
          f"log10(P)={P[p_idx]:.2f}")

# Create wavelength grid from frequency edges
# Convert frequency (Hz) to wavelength (nm)
c_light = 2.99792458e10  # cm/s
wavelength_edges_nm = (c_light / FreqG) * 1e7  # Convert to nm
wavelength_centers_nm = 0.5 * (wavelength_edges_nm[:-1] + wavelength_edges_nm[1:])

# Create figure with subplots (2x5 grid)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

# Color map for sub-bins
colors = plt.cm.viridis(np.linspace(0, 1, nsubbins))

for idx, (t_idx, p_idx) in enumerate(zip(random_t_indices, random_p_indices)):
    ax = axes[idx]
    
    # Extract ODF for this T,P point
    odf_tp = ODF[t_idx, p_idx, :, :]  # [nbins, nsubbins]
    
    # Convert to actual opacity: kappa = 10^(ODF/1000)
    kappa_tp = 10.0 ** (odf_tp / 1000.0)
    
    # Plot each sub-bin
    for sub_idx in range(nsubbins):
        ax.semilogy(wavelength_centers_nm, kappa_tp[:, sub_idx], 
                   alpha=0.6, linewidth=1.5, color=colors[sub_idx],
                   label=f'Sub-bin {sub_idx+1}' if idx == 0 else None)
    
    # Formatting
    ax.set_xlabel('Wavelength [nm]', fontsize=9)
    ax.set_ylabel('Opacity [cm²/g]', fontsize=9)
    ax.set_title(f'T={T[t_idx]:.0f} K, log₁₀(P)={P[p_idx]:.1f}', 
                fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(wavelength_centers_nm.min(), wavelength_centers_nm.max())
    
    # Add median line
    median_kappa = np.median(kappa_tp, axis=1)
    ax.semilogy(wavelength_centers_nm, median_kappa, 'r-', 
               linewidth=2, alpha=0.8, label='Median' if idx == 0 else None)

# Add legend to first subplot
axes[0].legend(fontsize=7, loc='upper right', ncol=2)

plt.suptitle(f'Opacity Distribution Functions at 10 Random (T, P) Points\n'
             f'ODF Data: {nbins} wavelength bins × {nsubbins} sub-bins per wavelength',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout()

# Save figure
output_file = 'odf_samples.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Plot saved to {output_file}")

# Print statistics
print("\n═══ ODF Statistics ═══")
kappa_all = 10.0 ** (ODF[:, :, :, :] / 1000.0)
print(f"Overall opacity range: {kappa_all.min():.2e} - {kappa_all.max():.2e} cm²/g")
print(f"Log10 opacity range:   {np.log10(kappa_all.min()):.2f} - {np.log10(kappa_all.max()):.2f}")

# Statistics for each sample
print("\nPer-sample statistics:")
for idx, (t_idx, p_idx) in enumerate(zip(random_t_indices, random_p_indices)):
    odf_tp = ODF[t_idx, p_idx, :, :]
    kappa_tp = 10.0 ** (odf_tp / 1000.0)
    print(f"  Sample {idx+1}: κ = {kappa_tp.min():.2e} - {kappa_tp.max():.2e} cm²/g "
          f"(range: {kappa_tp.max()/kappa_tp.min():.1e})")

plt.show()
