#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "typer",
#     "numpy",
#     "netcdf4",
#     "rich",
#     "tqdm",
#     "scipy",
#     "matplotlib",
# ]
# ///
"""
Tau-sorting: Opacity binning for stellar atmospheres

This script reads atmospheric model data, opacity distribution functions (ODFs),
and continuum opacity data to calculate binned opacities for radiative transfer.
"""

# from matplotlib.tests.test_widgets import ax
import json
import time
from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer
from netCDF4 import Dataset
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from group_derivatives import analyze_group
from kappa_band_reader import KappaBandComparison, write_kappa_4_band_comparison

console = Console()
app = typer.Typer(help="Tau-sorting opacity binning tool")


# Physical constants
CVAC = 2.99792458e10  # cm/s
H = 6.626196e-27  # erg*sec
K_B = 1.380622e-16  # erg/K
TENLOG = 2.30258509299405
LN10 = 4.342944819032518e-1


class AtmosphericData:
    """Container for 1D atmospheric model data"""

    def __init__(self):
        self.z: NDArray[np.float64] = None  # height [cm]
        self.rho: NDArray[np.float64] = None  # density [g/cm^3]
        self.p: NDArray[np.float64] = None  # pressure [dyn/cm^2]
        self.T: NDArray[np.float64] = None  # temperature [K]
        self.nlevels: int = 0

    def __repr__(self):
        return (
            f"AtmosphericData(nlevels={self.nlevels}, "
            f"T=[{self.T[0]:.1f}..{self.T[-1]:.1f}]K, "
            f"p=[{self.p[0]:.2e}..{self.p[-1]:.2e}])"
        )

    def plot(self):
        """Plot atmospheric structure (for debugging)"""
        import matplotlib.pyplot as plt

        z_mm = self.z / 1e8  # Convert to Mm

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # Temperature
        axes[0].plot(z_mm, self.T, "r-")
        axes[0].set_xlabel("Height [Mm]")
        axes[0].set_ylabel("Temperature [K]")
        axes[0].set_title("Temperature Profile")
        axes[0].grid(True)

        # Density
        axes[1].semilogy(z_mm, self.rho, "b-")
        axes[1].set_xlabel("Height [Mm]")
        axes[1].set_ylabel("Density [g/cm³]")
        axes[1].set_title("Density Profile")
        axes[1].grid(True, which="both")

        # Pressure
        axes[2].semilogy(z_mm, self.p, "g-")
        axes[2].set_xlabel("Height [Mm]")
        axes[2].set_ylabel("Pressure [dyn/cm²]")
        axes[2].set_title("Pressure Profile")
        axes[2].grid(True, which="both")

        plt.tight_layout()
        plt.show()


class ODFData:
    """Container for Opacity Distribution Function data"""

    def __init__(self):
        self.ODF: NDArray[np.float64] | None = None  # ODF values [nt, np, nbins, nsubbins]
        self.wavelength_grid: NDArray[np.float64] | None = None  # Frequency grid edges
        self.P: NDArray[np.float64] | None = None  # Pressure grid
        self.T: NDArray[np.float64] | None = None  # Temperature grid
        self.subbin: NDArray[np.float64] | None = None  # Sub-bin weights
        self.vturb: np.float64 | None = None  # Turbulent velocity

        # Dimensions
        self.np = 0
        self.nt = 0
        self.nbins = 0
        self.nsubbins = 0
        self.numfp = 0

    def __repr__(self):
        return f"ODFData(nt={self.nt}, np={self.np}, nbins={self.nbins}, nsubbins={self.nsubbins})"


class ContinuumData:
    """Container for continuum opacity data"""

    def __init__(self):
        self.kappa_abs: NDArray[np.float64] = None  # Absorption opacity

    def __repr__(self):
        return f"ContinuumData(shape={self.kappa_abs.shape if self.kappa_abs is not None else None})"


def read_atmospheric_model(filepath: Path) -> AtmosphericData:
    """
    Read 1D atmospheric model from G2_1D.dat file.

    Format: height(cm), density(g/cm^3), pressure(dyn/cm^2), temperature(K)

    Args:
        filepath: Path to G2_1D.dat file

    Returns:
        AtmosphericData object with atmospheric structure
    """
    console.print(f"[cyan]Reading atmospheric model from {filepath}[/cyan]")

    try:
        data = np.loadtxt(filepath)

        atm = AtmosphericData()
        atm.z = data[:, 0]
        atm.rho = data[:, 1]
        atm.p = data[:, 2]
        atm.T = data[:, 3]
        atm.nlevels = len(atm.z)

        console.print(f"  ✓ Loaded {atm.nlevels} atmospheric levels")
        console.print(f"  Temperature range: {atm.T.min():.1f} - {atm.T.max():.1f} K")
        console.print(f"  Pressure range: {atm.p.min():.2e} - {atm.p.max():.2e} dyn/cm²")

        return atm

    except Exception as e:
        console.print(f"[red]Error reading atmospheric model: {e}[/red]")
        raise


def read_odf_npy(filepath: Path) -> ODFData:
    """
    Read Opacity Distribution Function data from NumPy .npy file.

    This is much faster than reading NetCDF format.
    Expected structure: structured array with named fields from convert_odf_to_npy.py

    Args:
        filepath: Path to ODF .npy file

    Returns:
        ODFData object with opacity distributions
    """
    console.print(f"[cyan]Reading ODF data from {filepath}[/cyan]")
    console.print("  Using fast .npy format")

    try:
        data = np.load(filepath, allow_pickle=True)

        odf = ODFData()

        # Read dimensions
        odf.nt = int(data["nt"][0])
        odf.np = int(data["np"][0])
        odf.nbins = int(data["nbins"][0])
        odf.nsubbins = int(data["nsubbins"][0])
        odf.numfp = int(data["numfp"][0])

        console.print(f"  Dimensions: nt={odf.nt}, np={odf.np}, nbins={odf.nbins}, nsubbins={odf.nsubbins}")

        # Read arrays
        odf.ODF = data["ODF"][0]
        odf.wavelength_grid = data["wavelength_grid"][0] * 1e-7  # convert to cm
        odf.P = data["P"][0]
        odf.T = data["T"][0]
        odf.subbin = data["subbin"][0]
        odf.vturb = np.float64(data["vturb"][0])

        console.print(f"  Turbulent velocity: {odf.vturb} km/s")
        console.print("  ✓ ODF loaded successfully")
        console.print(f"  Temperature grid: {odf.T.min():.1f} - {odf.T.max():.1f} K")
        console.print(f"  Pressure grid: {odf.P.min():.2e} - {odf.P.max():.2e} dyn/cm²")
        console.print(f"  Frequency range: {odf.wavelength_grid.min():.2e} - {odf.wavelength_grid.max():.2e} Hz")
        console.print(f"  ODF sub-bins shape: {odf.subbin.shape}")

        return odf

    except Exception as e:
        console.print(f"[red]Error reading ODF .npy file: {e}[/red]")
        raise


def read_odf_netcdf(filepath: Path) -> ODFData:
    """
    Read Opacity Distribution Function data from NetCDF file.

    Expected structure:
    - dimensions: np, nt, nbins, nsubbins, numfp
    - variables: ODF(nt, np, nbins, nsubbins), FreqG(numfp), P(np), T(nt), subbin(nbins, nsubbins)

    Args:
        filepath: Path to ODF NetCDF file

    Returns:
        ODFData object with opacity distributions
    """
    console.print(f"[cyan]Reading ODF data from {filepath}[/cyan]")

    try:
        with Dataset(filepath, "r") as nc:
            odf = ODFData()

            # Read dimensions
            odf.np = nc.dimensions["np"].size
            odf.nt = nc.dimensions["nt"].size
            odf.nbins = nc.dimensions["nbins"].size
            odf.nsubbins = nc.dimensions["nsubbins"].size
            odf.numfp = nc.dimensions["numfp"].size

            console.print(f"  Dimensions: nt={odf.nt}, np={odf.np}, nbins={odf.nbins}, nsubbins={odf.nsubbins}")

            # Read variables
            odf.ODF = 10 ** (nc.variables["ODF"][:] / 1000)  # short integer, convert: 10^(ODF/1000)
            odf.wavelength_grid = nc.variables["FreqG"][:] * 1e-7
            odf.P = nc.variables["P"][:]
            odf.T = nc.variables["T"][:]
            odf.subbin = nc.variables["subbin"][:]

            # Read global attributes
            if hasattr(nc, "vturb"):
                odf.vturb = nc.vturb
                console.print(f"  Turbulent velocity: {odf.vturb} km/s")

            console.print("  ✓ ODF loaded successfully")
            console.print(f"  Temperature grid: {odf.T.min():.1f} - {odf.T.max():.1f} K")
            console.print(f"  Pressure grid: {odf.P.min():.2e} - {odf.P.max():.2e} dyn/cm²")
            console.print(
                f"  Wavelength range: {odf.wavelength_grid.min() * 1e7:.2e} - {odf.wavelength_grid.max() * 1e7:.2e} cm"
            )
            console.print(f"  ODF sub-bins: {odf.subbin.shape}")
            initial_sub_bins = odf.subbin[0]
            for i, item in enumerate(odf.subbin):
                if not np.array_equal(odf.subbin[i], initial_sub_bins):
                    console.print(f"  Warning: ODF sub-bins differ at index {i}")
                    console.print(f"  Initial ODF sub-bins: {initial_sub_bins}")
                    console.print(f"  Current ODF sub-bins: {odf.subbin[i]}")
                    break
            return odf

    except Exception as e:
        console.print(f"[red]Error reading ODF file: {e}[/red]")
        raise


def read_continuum_opacity(filepath: Path, n_bins: int, n_temperature: int, n_pressure: int) -> NDArray[np.float64]:
    """
    Read continuum opacity data from ASCII file.

    Format: Binary column data with shape (nlam, nt, n_pressure)
    Data is stored in Fortran order: for each lambda, for each T, for each P

    Args:
        filepath: Path to continuum data file
        nlam: Number of wavelength bins
        nt: Number of temperature points
        n_pressure: Number of pressure points

    Returns:
        3D numpy array with continuum opacity [nlam, nt, n_pressure]
    """
    console.print(f"[cyan]Reading continuum data from {filepath}[/cyan]")

    try:
        # Check for .npy version first (much faster - 75x speedup!)
        npy_path = Path(str(filepath).replace(".dat", ".npy"))

        if npy_path.exists():
            console.print("  [green]Using fast .npy format[/green]")
            kappa = np.load(npy_path)
        else:
            # Fall back to ASCII .dat file
            console.print("  [yellow]Loading ASCII (slow) - consider `tausort.py convert-continuum`[/yellow]")
            data = np.loadtxt(str(filepath))
            expected_size = n_bins * n_temperature * n_pressure

            if data.size != expected_size:
                console.print(f"[yellow]Warning: Expected {expected_size} values, got {data.size}[/yellow]")

            # .dat is (lambda, T, P) C-order -> (nbins, nt, np); transpose to (nt, np, nbins),
            # matching the pre-generated continuumabs.npy layout that `main` reads.
            kappa = data.reshape((n_bins, n_temperature, n_pressure)).transpose(1, 2, 0)

        console.print(f"  ✓ Loaded continuum opacity with shape {kappa.shape}")
        console.print(f"  Value range: {kappa.min():.2e} - {kappa.max():.2e}")

        return kappa

    except Exception as e:
        console.print(f"[red]Error reading continuum file: {e}[/red]")
        raise


def read_continuum_data(
    abs_file: Path,
    # scat_file: Path,
    # all_file: Path,
    nlam: int,
    nt: int,
    n_pressure: int,
) -> ContinuumData:
    """
    Read all continuum opacity data files.

    Args:
        abs_file: Path to absorption continuum file
        nlam: Number of wavelength bins
        nt: Number of temperature points
        n_pressure: Number of pressure points

    Returns:
        ContinuumData object with all continuum opacities
    """
    cont = ContinuumData()

    if abs_file.exists():
        cont.kappa_abs = read_continuum_opacity(abs_file, nlam, nt, n_pressure)

    return cont


def verify_data_consistency(atm: AtmosphericData, odf: ODFData, cont: ContinuumData):
    """
    Verify that all input data is consistent and ready for processing.

    Args:
        atm: Atmospheric data
        odf: ODF data
        cont: Continuum data
    """
    console.print("\n[cyan]Verifying data consistency...[/cyan]")

    checks_passed = 0
    checks_total = 0

    # Check atmospheric data
    checks_total += 1
    if atm.nlevels > 0:
        console.print("  ✓ Atmospheric model loaded")
        checks_passed += 1
    else:
        console.print("  ✗ Atmospheric model missing")

    # Check ODF data
    checks_total += 1
    if odf.ODF is not None:
        console.print(f"  ✓ ODF data loaded: {odf}")
        checks_passed += 1
    else:
        console.print("  ✗ ODF data missing")

    # Check continuum data
    checks_total += 1
    if cont.kappa_abs is not None:
        console.print(f"  ✓ Continuum data loaded: {cont}")
        checks_passed += 1
    else:
        console.print("  ✗ Continuum data missing")

    # Check dimensions match
    checks_total += 1
    if cont.kappa_abs is not None and odf.ODF is not None:
        expected_shape = (odf.nt, odf.np, odf.nbins)
        if cont.kappa_abs.shape == expected_shape:
            console.print(f"  ✓ Continuum shape matches ODF: {expected_shape}")
            checks_passed += 1
        else:
            console.print(f"  ✗ Shape mismatch: continuum={cont.kappa_abs.shape}, expected={expected_shape}")

    # Summary
    console.print(f"\n[bold]Verification: {checks_passed}/{checks_total} checks passed[/bold]")

    if checks_passed == checks_total:
        console.print("[green]✓ All data verified successfully![/green]")
    else:
        console.print("[yellow]⚠ Some verification checks failed[/yellow]")

    return checks_passed == checks_total


def planck_function(wavelength: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """
    Calculate the Planck function B_lambda(T) for given wavelengths and temperatures.

    Args:
        wavelength: Wavelength array [cm]
        temperature: Temperature array [K]
    Returns:
        Planck function values B_lambda [erg/s/cm²/ster/cm]
    """

    B = 2 * H * CVAC**2 / wavelength**5 / (np.exp(H * CVAC / wavelength / K_B / temperature) - 1)

    return B


def planck_derivative(wavelength: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """
    Calculate the derivative of the Planck function dB/dT.

    Args:
        wavelength: Wavelength array [cm]
        temperature: Temperature array [K]
    Returns:
        Derivative of the Planck function dB/dT [erg/s/cm²/ster/cm/K]
    """
    # Calculate the Planck function B_lambda(T)
    # B = planck_function(wavelength, temperature)

    # Calculate the derivative dB/dT using numerical differentiation
    dT = 1e-5 * temperature
    B_plus = planck_function(wavelength, temperature + dT)
    B_minus = planck_function(wavelength, temperature - dT)
    dB_dT = (B_plus - B_minus) / (2 * dT)

    return dB_dT


def planck_derivative_analytic(wavelength: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """
    Calculate the analytic derivative of the Planck function dB/dT.

    Args:
        wavelength: Wavelength array [cm]
        temperature: Temperature array [K]

    Returns:
        derivative: Derivative of the Planck function dB/dT [erg/s/cm²/ster/cm/K]
    """
    # Physical constants in CGS units
    h = H
    c = CVAC
    k_B = K_B

    # Pre-compute constants
    c1 = 2.0 * h * c**2
    c2 = h * c / k_B

    # Calculate the dimensionless exponent term: x = hc / (lambda * k * T)
    # Ensure no division by zero if inputs contain 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = c2 / (wavelength * temperature)

    # Calculate the exponential term
    # We trap overflow warnings here for very small wavelengths/temps where exp(x) -> inf
    with np.errstate(over="ignore"):
        exp_x = np.exp(x)

    # Calculate the derivative
    # Formula: dB/dT = (c1 / lambda^5) * (x * exp(x)) / (T * (exp(x) - 1)^2)
    # Using expm1 for numerical precision on small x: (exp(x) - 1)

    # Note: exp(x)/(exp(x)-1)^2 can be unstable for large x.
    # We can simplify calculation by grouping terms.

    denom_factor = np.expm1(x) ** 2

    # Handling potential overflow/NaNs for extreme values
    # If x is very large, exp(x)/(exp(x)-1)^2 approaches exp(-x), which is 0.
    # If x is very small, we approach the Rayleigh-Jeans limit derivative.

    term1 = c1 / (wavelength**5)
    term2 = (x * exp_x) / (temperature * denom_factor)

    derivative = term1 * term2

    # Clean up NaNs resulting from 0/0 or inf/inf in extreme limits if necessary
    # (Optional: depends on input guarantees, here we replace NaNs with 0 for safety)
    derivative = np.nan_to_num(derivative)

    return derivative


def plot_planck_and_derivatives(output_file: str = "planck_verification.pdf") -> None:
    """
    Plot Planck function and its derivatives for verification.

    Uses the existing planck_function() and planck_derivative_analytic()
    functions to create visualization for verification purposes.

    Args:
        output_file: Output filename for the plot
    """
    import matplotlib.pyplot as plt

    console.print("\n[cyan]═══ Planck Function Verification ═══[/cyan]\n")

    # Wavelength range in nm
    wavelengths_nm = np.linspace(10, 2000, 1000)  # 10-1000 nm
    wavelengths = wavelengths_nm * 1.0e-7  # convert to cm

    # Temperature range
    T_range = [5000, 5200, 5400, 5600, 5800, 6000, 6200, 6400, 6600]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # === Left plot: Planck functions ===
    cmap = matplotlib.colormaps.get_cmap("plasma")
    for i, T in enumerate(T_range):
        B = planck_function(wavelengths, T)
        color = cmap(i / len(T_range))
        ax1.plot(wavelengths_nm, B, linewidth=2, color=color, label=f"T={T} K")

    ax1.set_xlabel("Wavelength [nm]", fontsize=12)
    ax1.set_ylabel("Planck Function B [erg/s/cm²/ster/cm]", fontsize=12)
    ax1.set_title("Planck Function at Various Temperatures", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc="best")
    ax1.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

    # === Right plot: Derivatives with tangent lines ===
    selected_wavelengths_nm = [100, 250, 500, 750]  # nm
    selected_wavelengths = np.array(selected_wavelengths_nm) * 1.0e-7  # convert to cm
    colors_deriv = ["red", "blue", "green", "orange"]

    # Plot main Planck function on right panel for tangent visualization
    T_tangent = 5800  # K (Solar temperature for tangent lines)
    B_for_tangent = planck_function(wavelengths, T_tangent)
    ax2.plot(
        wavelengths_nm,
        B_for_tangent,
        "k-",
        linewidth=2,
        alpha=0.5,
        label=f"B(λ) at T={T_tangent}K",
    )

    # Add T±200K curves
    B_plus_200 = planck_function(wavelengths, T_tangent + 200)
    B_minus_200 = planck_function(wavelengths, T_tangent - 200)
    ax2.plot(
        wavelengths_nm,
        B_plus_200,
        "k--",
        linewidth=1.5,
        alpha=0.3,
        label=f"T={T_tangent + 200}K",
    )
    ax2.plot(
        wavelengths_nm,
        B_minus_200,
        "k:",
        linewidth=1.5,
        alpha=0.3,
        label=f"T={T_tangent - 200}K",
    )

    # Draw tangent lines at selected wavelengths
    for wl_nm, wl_cm, color in zip(selected_wavelengths_nm, selected_wavelengths, colors_deriv):
        # Get Planck function value at this wavelength
        idx = np.argmin(np.abs(wavelengths_nm - wl_nm))
        B_val = B_for_tangent[idx]

        # Calculate derivative dB/dT at this point
        dB_dT = planck_derivative_analytic(np.array([wl_cm]), T_tangent)[0]

        # Create temperature range for tangent line (linear approximation)
        # We'll show the tangent in "temperature space" around T_tangent
        dT_range = np.linspace(-500, 500, 100)  # ±500 K range

        # Tangent line: B(T) ≈ B(T0) + dB/dT * (T - T0)
        tangent_line = B_val + dB_dT * dT_range

        # For x-axis, we keep wavelength constant but need to show the line
        # Actually, we want wavelength on x-axis, so we create a small wavelength range
        wl_tangent_range = np.linspace(wl_nm - 100, wl_nm + 100, 100)

        # The tangent line slope in (wavelength, B) space would be dB/dλ
        # But we want to show dB/dT, so let's create the tangent differently
        # Linear tangent: for small changes in wavelength around wl,
        # B ≈ B_val + slope * (λ - λ0)
        # But slope here should represent the temperature derivative visually

        # Better approach: plot tangent as B vs small perturbation
        # Use a pseudo x-axis offset to show the tangent slope
        perturbation = (wl_tangent_range - wl_nm) / 100.0  # Normalize to [-1, 1]
        tangent_visual = B_val + dB_dT * perturbation * 500  # Scale by 500K for visibility

        # Plot the point
        ax2.plot(
            wl_nm,
            B_val,
            "o",
            color=color,
            markersize=10,
            label=f"λ={wl_nm} nm",
            zorder=5,
        )

        # Plot tangent line
        ax2.plot(
            wl_tangent_range,
            tangent_visual,
            "--",
            color=color,
            alpha=0.8,
            linewidth=2.5,
        )

    ax2.set_xlabel("Wavelength [nm]", fontsize=12)
    ax2.set_ylabel("Planck Function B [erg/s/cm²/ster/cm]", fontsize=12)
    ax2.set_title(f"dB/dT Tangent Lines at T={T_tangent}K", fontsize=13, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="best", ncol=2)
    ax2.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

    plt.suptitle("Planck Function and Derivative Verification", fontsize=15, fontweight="bold")
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ Plot saved to {output_file}[/green]")

    # Print numerical verification at selected wavelengths
    console.print("\n[cyan]Numerical verification at selected wavelengths:[/cyan]")
    T_test = 5800.0  # K (Solar temperature)
    for wl_nm, wl_cm in zip(selected_wavelengths_nm, selected_wavelengths):
        B = planck_function(np.array([wl_cm]), T_test)[0]
        dB_dT = planck_derivative_analytic(np.array([wl_cm]), T_test)[0]
        wien_peak = 2.898e6 / T_test  # Wien's displacement law

        console.print(f"\n  λ = {wl_nm} nm, T = {T_test} K:")
        console.print(f"    B(λ,T)      = {B:.6e} erg/s/cm²/ster/cm")
        console.print(f"    dB/dT       = {dB_dT:.6e} erg/s/cm²/ster/cm/K")
        console.print(f"    Wien peak λ = {wien_peak:.1f} nm")

    console.print("\n[green]✓ Planck function verification complete![/green]\n")


def interpolate_kappa_to_atmosphere(odf: ODFData, cont: ContinuumData, atmo: AtmosphericData) -> NDArray[np.float64]:
    """
    Calculate reference opacities on the atmosphere T-p grid.
    Returns:
        Reference opacity array with shape [n_layers, nbins, nsubbins]
    """

    # 1. Prepare the source data
    # continuum_kappa shape: [nt, np, nbins] -> broadcast to [nt, np, nbins, 1]
    # odf_kappa shape: [nt, np, nbins, nsubbins]
    continuum_kappa = cont.kappa_abs
    odf_kappa = odf.ODF

    # Combine opacities
    total_kappa = odf_kappa + continuum_kappa[..., np.newaxis]  # shape: [nt, np, nbins, nsubbins]

    class VectorGridInterpolator:
        def __init__(self, t_grid, p_grid, data_grid, method="linear"):
            """
            Args:
                t_grid: 1D array of shape [nt]
                p_grid: 1D array of shape [np]
                data_grid: ND array of shape [nt, np, ...payload_dims]
            """
            self.t_grid = t_grid
            self.p_grid = p_grid

            # Validation
            if data_grid.shape[:2] != (len(t_grid), len(p_grid)):
                raise ValueError(f"Data shape {data_grid.shape} does not match grids ({len(t_grid)}, {len(p_grid)})")

            # Initialize Scipy interpolator
            # We use bounds_error=True so Scipy also enforces checks efficiently
            self.interpolator = RegularGridInterpolator(
                points=(t_grid, p_grid),
                values=data_grid,
                method=method,
                bounds_error=True,
            )

        def get_vectors(self, t_targets, p_targets):
            """
            Interpolates at multiple T-P points simultaneously.

            Args:
                t_targets: array of shape [N]
                p_targets: array of shape [N]
            Returns:
                Interpolated array of shape [N, ...payload_dims]
            """
            # Ensure inputs are numpy arrays
            t_pts = np.atleast_1d(t_targets)
            p_pts = np.atleast_1d(p_targets)

            # 1. Custom Bounds Verification (Vectorized)
            # Check if ANY point is out of bounds
            if np.any(t_pts < self.t_grid.min()) or np.any(t_pts > self.t_grid.max()):
                min_t, max_t = self.t_grid.min(), self.t_grid.max()
                bad_vals = t_pts[(t_pts < min_t) | (t_pts > max_t)]
                t_pts = np.clip(t_pts, a_min=min_t, a_max=max_t)
                console.print(
                    f"Temperature query contains values out of bounds [{min_t}, {max_t}]. Violations: {bad_vals[:3]}..."
                )

            if np.any(p_pts < self.p_grid.min()) or np.any(p_pts > self.p_grid.max()):
                min_p, max_p = self.p_grid.min(), self.p_grid.max()
                bad_vals = p_pts[(p_pts < min_p) | (p_pts > max_p)]
                p_pts = np.clip(p_pts, a_min=min_p, a_max=max_p)
                console.print(
                    f"Pressure query contains values out of bounds [{min_p}, {max_p}]. Violations: {bad_vals[:3]}..."
                )

            # 2. Prepare Query Points
            # Scipy expects a shape of (N_points, N_dimensions) -> (N, 2)
            query_points = np.column_stack((t_pts, p_pts))

            # 3. Interpolate
            # Returns shape (N, nbins, nsubbins)
            return self.interpolator(query_points)

    # --- Execution ---

    # Initialize the interpolator with the calculated total_kappa
    interpolator = VectorGridInterpolator(odf.T, odf.P, total_kappa)

    try:
        # Interpolate onto the full atmospheric profile at once
        # atmo.T and atmo.p are arrays of length [n_layers]
        interpolated_opacity = interpolator.get_vectors(np.log10(atmo.T), np.log10(atmo.p))

        # console.print(f"Interpolation successful. Output shape: {interpolated_opacity.shape}")
        return interpolated_opacity

    except ValueError as e:
        # Handle bounds errors (e.g., atmosphere is hotter than ODF table)
        # console.print(f"[red]Interpolation Error:[/red] {e}")
        raise e


def calculate_reference_opacities(odf: ODFData, cont: ContinuumData, kind: str = "rosseland") -> NDArray[np.float64]:
    """
    Calculate reference opacities (Rosseland mean, Planck mean, etc.).

    This function implements the calculation of reference opacities such as
    Rosseland mean, 500nm opacity, and other reference opacities needed
    for tau-sorting.

    The Rosseland mean opacity is defined as:
        kappa_ross = (∫ (1/kappa) * (dB/dT) * dλ)^-1 / (∫ (dB/dT) * dλ)

    The Planck mean opacity is defined as:
        kappa_planck = ∫ kappa * B * dλ / ∫ B * dλ

    Args:
        odf: ODF data containing opacity distribution functions
            - ODF shape: [nt, np, nbins, nsubbins]
            - T grid: [nt]
            - P grid: [np]
        cont: Continuum opacity data
            - kappa_all shape: [nbins, nt, np]
        kind: Type of reference opacity to calculate
              Options: "rosseland", "planck", "500nm"

    Returns:
        Reference opacity array with shape [nt, np]
    """
    # TODO: Implement reference opacity calculations
    # - Calculate Planck function B_lambda(T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    # - Calculate derivative dB/dT
    # - Combine ODF and continuum opacities
    # - Integrate opacities weighted by Planck function (Planck mean)
    # - Integrate 1/kappa weighted by dB/dT (Rosseland mean)
    # - Extract opacity at specific wavelength (e.g., 500nm)

    # nt, np_dim = odf.nt, odf.np
    # result = np.zeros((nt, np_dim), dtype=np.float64)

    # first we add the continuum opacity to the ODF based on the bin
    continuum_kappa = cont.kappa_abs  # shape: [nt, np, nbins]
    odf_kappa = odf.ODF  # shape: [nt, np, nbins, nsubbins]

    console.print(f"shapes: continuum_kappa={continuum_kappa.shape}, odf_kappa={odf_kappa.shape}")

    # add continuum kappa to each subbin based on the bin
    # if odf.nbins != continuum_kappa.shape[-2]:
    #     console.print(f"[red]Error: ODF nbins ({odf.nbins}) does not match continuum kappa shape ({continuum_kappa.shape})[/red]")
    #     raise ValueError("Inconsistent nbins between ODF and continuum data")

    total_kappa = odf_kappa + continuum_kappa[..., np.newaxis]  # shape: [nt, np, nbins, nsubbins]

    # for each T, p point from the ODF calculate the reference opacity using np.nditerate
    temperature_grid = odf.T  # shape: [nt]
    pressure_grid = odf.P  # shape: [np]
    console.print(f"Calculating {kind} reference opacities...")
    console.print(f"  Temperature grid: {temperature_grid.shape}")
    console.print(f"  Pressure grid: {pressure_grid.shape}")
    temperature_pressure_grid = np.array([(temp, pres) for temp in temperature_grid for pres in pressure_grid])
    for idx, (temperature, pressure) in tqdm(
        enumerate(temperature_pressure_grid), total=temperature_pressure_grid.shape[0]
    ):
        t_idx = np.where(odf.T == temperature)[0][0]
        p_idx = np.where(odf.P == pressure)[0][0]

        kappa_values = total_kappa[t_idx, p_idx, ...].flatten()  # shape: [nbins * nsubbins]
        # verify kappa_values length matches expected
        expected_length = odf.nbins * odf.nsubbins
        if kappa_values.shape[0] != expected_length:
            console.print(
                f"[red]Error: kappa_values length ({kappa_values.shape[0]}) does not match expected ({expected_length})[/red]"
            )
            raise ValueError("Inconsistent kappa values length")

        # Assign sub-bin wavelengths based on ODF frequency grid and sub-bin weights
        wavelength_grid_bin_edges = odf.wavelength_grid  # cm
        wavelength_grid_bin_size = np.diff(wavelength_grid_bin_edges)
        # console.print(f"  Wavelength grid shape: {wavelength_grid_bin_edges.shape}")
        # console.print(f"  Wavelength grid values: {wavelength_grid_bin_edges[:10]}")

        wavelength_grid_subbin_weights = odf.subbin
        wavelength_grid_subbins_center = np.zeros_like(kappa_values)
        # console.print(f"odf.subbin shape: {odf.subbin.shape}")
        number_of_subbins: int = odf.subbin.shape[1]
        wavelength_grid_subbins_edges_shape = odf.subbin.shape[0] * (number_of_subbins) + 1
        wavelength_grid_subbins_edges = np.zeros(wavelength_grid_subbins_edges_shape, dtype=np.float64)
        counter = 1
        for bin_idx in range(odf.nbins):
            # if bin_idx == 4:
            #     sys.exit(0)
            left_edge = wavelength_grid_bin_edges[bin_idx]
            right_edge = wavelength_grid_bin_edges[bin_idx + 1]
            subbin_weights = wavelength_grid_subbin_weights[bin_idx]
            bin_size = wavelength_grid_bin_size[bin_idx]
            subbin_sizes = bin_size * subbin_weights
            # console.print(f"  Bin {bin_idx}: bin_size={bin_size:.2e}, subbin_sizes={subbin_sizes}")
            sub_bin_edges = np.cumsum(subbin_sizes) + left_edge
            sub_bin_edges[-1] = right_edge  # ensure last edge matches right edge
            # console.print(f"  sub_bin_edges: {sub_bin_edges}")
            # console.print(f" np.cumsum(subbin_sizes): {np.cumsum(subbin_sizes)}")
            # console.print(f" bin_size: {bin_size}")

            # for the first bin we set the left and right edges directly
            if bin_idx == 0:
                wavelength_grid_subbins_edges[0] = left_edge
                # console.print(f" setting wavelength_grid_subbins_edges[0] to {left_edge}")
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = sub_bin_edges
                # console.print(f"setting the subbins from {counter} to {counter+len(subbin_sizes)}")
                # console.print(f" sub_bin_edges: {sub_bin_edges}")
                # console.print(f"  First bin edges: {wavelength_grid_subbins_edges[:15]}")
                # console.print(f"  Bin {bin_idx} edges: {wavelength_grid_subbins_edges[counter:counter+len(subbin_sizes)+1]}")
                counter += len(subbin_weights)
            # for subsequent bins we set the left edge to the previous right edge
            else:
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = sub_bin_edges
                # console.print(f"  Bin {bin_idx} edges: {wavelength_grid_subbins_edges[counter-1:counter+len(subbin_sizes)]}")
                counter += len(subbin_weights)
            # console.print(f" After bin {bin_idx}, counter={counter}")
            # console.print(f" Current wavelength_grid_subbins_edges: {wavelength_grid_subbins_edges[:counter+2]}")
        counter -= 1  # adjust for last increment
        if counter != len(kappa_values):
            console.print("[red]Error: Mismatch in wavelength grid and kappa values length[/red]")
            raise ValueError(
                f"Wavelength grid and kappa values length mismatch: counter={counter}, kappa_values={len(kappa_values)}"
            )

        wavelength_grid_subbins_centers = 0.5 * (wavelength_grid_subbins_edges[:-1] + wavelength_grid_subbins_edges[1:])
        B_lambda = planck_function(wavelength_grid_subbins_centers, temperature)
        dB_dT = planck_derivative_analytic(wavelength_grid_subbins_centers, temperature)
        if kind == "rosseland":
            total_kappa[t_idx, p_idx] = compute_rosseland_mean(kappa_values, dB_dT, wavelength_grid_subbins_centers)
        elif kind == "planck":
            total_kappa[t_idx, p_idx] = compute_planck_mean(kappa_values, B_lambda, wavelength_grid_subbins_centers)
        elif kind == "500nm":
            total_kappa[t_idx, p_idx] = compute_opacity_at_wavelength(
                kappa_values, wavelength_grid_subbins_centers, 500e-7
            )

    return total_kappa


def compute_rosseland_mean(
    kappa_values: NDArray[np.float64],
    dB_dT: NDArray[np.float64],
    wavelength_grid: NDArray[np.float64],
) -> float:
    """
    Compute Rosseland mean opacity.

    Args:
        kappa_values: Opacity values at each wavelength point
        dB_dT: Planck derivative at each wavelength point
        wavelength_grid: Wavelength grid [cm]

    Returns:
        Rosseland mean opacity
    """
    if kappa_values.shape != wavelength_grid.shape:
        console.print(
            f"[red]Error: kappa_values shape {kappa_values.shape} does not match wavelength grid shape {wavelength_grid.shape}[/red]"
        )
        raise ValueError("Inconsistent shapes for kappa values and wavelength grid")
    integrand_num = np.trapezoid((1.0 / kappa_values) * dB_dT, wavelength_grid)
    integrand_den = np.trapezoid(dB_dT, wavelength_grid)
    return integrand_den / integrand_num if integrand_num != 0 else 0.0


def compute_planck_mean(
    kappa_values: NDArray[np.float64],
    B_lambda: NDArray[np.float64],
    wavelength_grid: NDArray[np.float64],
) -> float:
    """
    Compute Planck mean opacity.

    Args:
        kappa_values: Opacity values at each wavelength point
        B_lambda: Planck function at each wavelength point
        wavelength_grid: Wavelength grid [cm]

    Returns:
        Planck mean opacity
    """
    integrand_num = np.trapezoid(kappa_values * B_lambda, wavelength_grid)
    integrand_den = np.trapezoid(B_lambda, wavelength_grid)
    return integrand_num / integrand_den if integrand_den != 0 else 0.0


def compute_opacity_at_wavelength(
    kappa_values: NDArray[np.float64],
    wavelength_grid: NDArray[np.float64],
    target_wavelength: float,
) -> float:
    """
    Get opacity at a specific wavelength.

    Args:
        kappa_values: Opacity values at each wavelength point
        wavelength_grid: Wavelength grid [cm]
        target_wavelength: Target wavelength [cm]

    Returns:
        Opacity at the target wavelength
    """
    idx = np.argmin(np.abs(wavelength_grid - target_wavelength))
    return kappa_values[idx]


def compute_combined_opacity(
    kappa_values: NDArray[np.float64],
    B_lambda: NDArray[np.float64],
    dB_dT_lambda: NDArray[np.float64],
    wavelength_grid: NDArray[np.float64],
    tau_threshold: float,
) -> float:
    r"""
    Compute combined opacity considering a tau threshold.

    Eq. 12 from A. Voegler et al. 2004:
    $\bar{\kappa}_i=2^{-\frac{\tau_i}{\tau_0}} \bar{K}_{P, i}+\left(1-2^{-\frac{\tau_i}{\tau_0}}\right) \bar{K}_{R, i}$.

    Args:
        kappa_values: Opacity values at each wavelength point
        B_lambda: Planck function at each wavelength point
        dB_dT_lambda: Derivative of Planck function at each wavelength point
        wavelength_grid: Wavelength grid [cm]
        tau_threshold: Tau threshold value

    Returns:
        Combined opacity value
    """
    kappa_planck = compute_planck_mean(kappa_values, B_lambda, wavelength_grid)
    kappa_rosseland = compute_rosseland_mean(kappa_values, dB_dT_lambda, wavelength_grid)

    return 2 ** (-tau_threshold) * kappa_planck + (1 - 2 ** (-tau_threshold)) * kappa_rosseland


def calculate_reference_opacities_from_custom_tp_grid(
    atmo: AtmosphericData,
    reference_opacities: NDArray[np.float64],
    wavelength_grid: NDArray[np.float64],
    subbin: NDArray[np.float64],
    nbins: int,
    nsubbins: int,
    kind: str = "rosseland",
) -> NDArray[np.float64]:
    """
    Calculate reference opacities (Rosseland mean, Planck mean, etc.).

    This function implements the calculation of reference opacities such as
    Rosseland mean, 500nm opacity, and other reference opacities needed
    for tau-sorting based on a custom T-P grid.

    The Rosseland mean opacity is defined as:
        kappa_ross = (∫ (1/kappa) * (dB/dT) * dλ)^-1 / (∫ (dB/dT) * dλ)

    The Planck mean opacity is defined as:
        kappa_planck = ∫ kappa * B * dλ / ∫ B * dλ

    Args:
        atmo: Atmospheric data containing T and P grids
            - T grid: [n_t]
            - P grid: [n_p]
        reference_opacities: Opacities on the custom T-p grid
            - shape: [n_t, n_p, nbins, nsubbins]
        wavelength_grid: Wavelength grid for the opacity data [cm]
        nbins: Number of bins in the opacity data
        nsubbins: Number of sub-bins per bin in the opacity data
        kind: Type of reference opacity to calculate
              Options: "rosseland", "planck", "500nm"

    Returns:
        Reference opacity array with shape [nt, np]
    """
    # shape verification
    n_atmosphere_points = atmo.nlevels
    opacity_at_tp_points = np.zeros((n_atmosphere_points,), dtype=np.float64)

    total_kappa = reference_opacities  # shape: [nt, np, nbins, nsubbins]

    temperature_grid = atmo.T  # shape: [nt]
    pressure_grid = atmo.p  # shape: [np]
    console.print(f"Calculating {kind} reference opacities...")
    console.print(f"  Temperature grid: {temperature_grid.shape}")
    console.print(f"  Pressure grid: {pressure_grid.shape}")
    temperature_pressure_grid = np.column_stack((temperature_grid, pressure_grid))
    console.print(f"  Temp-Pressure grid shape: {temperature_pressure_grid.shape}")
    for atmosphere_depth_idx, (temperature, pressure) in tqdm(
        enumerate(temperature_pressure_grid), total=temperature_pressure_grid.shape[0]
    ):
        t_idx = np.where(atmo.T == temperature)[0][0]
        p_idx = np.where(atmo.p == pressure)[0][0]

        kappa_values = total_kappa[atmosphere_depth_idx, ...].flatten()  # shape: [nbins * nsubbins]
        # verify kappa_values length matches expected
        expected_length = nbins * nsubbins
        if kappa_values.shape[0] != expected_length:
            console.print(
                f"[red]Error: kappa_values length ({kappa_values.shape[0]}) does not match expected ({expected_length})[/red]"
            )
            raise ValueError("Inconsistent kappa values length")

        # Assign sub-bin wavelengths based on ODF frequency grid and sub-bin weights
        wavelength_grid_bin_edges = wavelength_grid  # cm
        wavelength_grid_bin_size = np.diff(wavelength_grid_bin_edges)

        wavelength_grid_subbin_weights = subbin
        wavelength_grid_subbins_center = np.zeros_like(kappa_values)
        number_of_subbins: int = subbin.shape[1]
        wavelength_grid_subbins_edges_shape = subbin.shape[0] * (number_of_subbins) + 1
        wavelength_grid_subbins_edges = np.zeros(wavelength_grid_subbins_edges_shape, dtype=np.float64)
        counter = 1
        for bin_idx in range(nbins):
            left_edge = wavelength_grid_bin_edges[bin_idx]
            right_edge = wavelength_grid_bin_edges[bin_idx + 1]
            subbin_weights = wavelength_grid_subbin_weights[bin_idx]
            bin_size = wavelength_grid_bin_size[bin_idx]
            subbin_sizes = bin_size * subbin_weights
            sub_bin_edges = np.cumsum(subbin_sizes) + left_edge
            sub_bin_edges[-1] = right_edge  # ensure last edge matches right edge

            # for the first bin we set the left and right edges directly
            if bin_idx == 0:
                wavelength_grid_subbins_edges[0] = left_edge
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = sub_bin_edges
                counter += len(subbin_weights)
            # for subsequent bins we set the left edge to the previous right edge
            else:
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = sub_bin_edges
                counter += len(subbin_weights)
        counter -= 1  # adjust for last increment
        if counter != len(kappa_values):
            console.print("[red]Error: Mismatch in wavelength grid and kappa values length[/red]")
            raise ValueError(
                f"Wavelength grid and kappa values length mismatch: counter={counter}, kappa_values={len(kappa_values)}"
            )

        wavelength_grid_subbins_centers = 0.5 * (wavelength_grid_subbins_edges[:-1] + wavelength_grid_subbins_edges[1:])
        B_lambda = planck_function(wavelength_grid_subbins_centers, temperature)
        dB_dT = planck_derivative_analytic(wavelength_grid_subbins_centers, temperature)
        if kind == "rosseland":
            # Rosseland mean opacity
            if kappa_values.shape != wavelength_grid_subbins_centers.shape:
                console.print(
                    f"[red]Error: kappa_values shape {kappa_values.shape} does not match wavelength grid shape {wavelength_grid_subbins_centers.shape}[/red]"
                )
                raise ValueError("Inconsistent shapes for kappa values and wavelength grid")
            integrand_num = np.trapezoid((1.0 / kappa_values) * dB_dT, wavelength_grid_subbins_centers)
            integrand_den = np.trapezoid(dB_dT, wavelength_grid_subbins_centers)
            kappa_rosseland = integrand_den / integrand_num if integrand_num != 0 else 0.0
            opacity_at_tp_points[atmosphere_depth_idx] = kappa_rosseland

        elif kind == "planck":
            # Planck mean opacity
            integrand_num = np.trapezoid(kappa_values * B_lambda, wavelength_grid_subbins_centers)
            integrand_den = np.trapezoid(B_lambda, wavelength_grid_subbins_centers)
            kappa_planck = integrand_num / integrand_den if integrand_den != 0 else 0.0
            opacity_at_tp_points[atmosphere_depth_idx] = kappa_planck

        elif kind == "500nm":
            # Opacity at 500nm
            wl_500nm = 500e-7  # cm
            idx_500nm = np.argmin(np.abs(wavelength_grid - wl_500nm))
            opacity_at_tp_points[atmosphere_depth_idx] = kappa_values[idx_500nm]

    return opacity_at_tp_points, wavelength_grid_subbins_centers


def compute_tau_rosseland(atmo: AtmosphericData, kappa_rosseland: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Compute Rosseland optical depth profile for the atmosphere.

    For each layer, the optical depth is calculated by integrating
    the product of the Rosseland mean opacity and the density over height,
    up to that layer.

    Args:
        atmo: Atmospheric data containing height and density profiles
            - height: [n_layers] in cm
            - density: [n_layers] in g/cm³
        kappa_rosseland: Rosseland mean opacity on the atmosphere T-p grid
            - shape: [temp_layers, pressure_layers]
    Returns:
        Optical depth profile array with shape [n_layers]

    """
    n_layers = atmo.nlevels
    tau_rosseland = np.zeros(n_layers - 1, dtype=np.float64)
    height = atmo.z[::-1]  # cm, reverse to go from top to bottom
    density: NDArray[np.float64] = atmo.rho  # g/cm³

    console.print("Calculating Rosseland optical depth profile...")

    if kappa_rosseland.shape[0] != n_layers:
        console.print(
            f"[red]Error: kappa_rosseland shape {kappa_rosseland.shape} does not match atmospheric layers {n_layers}[/red]"
        )
        raise ValueError("Inconsistent shapes for kappa rosseland and atmospheric layers")

    # Use cumulative_trapezoid for efficient computation
    tau_rosseland = cumulative_trapezoid(kappa_rosseland * density, height, initial=0)[
        1:
    ]  # Remove first element (which is 0) to match original shape

    console.print("Rosseland optical depth profile calculation complete.")

    return tau_rosseland


def plot_rosseland_tau(atm: AtmosphericData, tau_rosseland: NDArray[np.float64]) -> None:
    """
    Plot the Rosseland optical depth profile.

    Args:
        atm: Atmospheric data
        tau_rosseland: Optical depth profile array with shape [n_layers]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left panel: tau vs height
    height_mm = atm.z / 1e8  # Convert cm to Mm
    ax1.semilogy(
        height_mm[::-1][:-1],
        tau_rosseland,
        "b-",
        linewidth=2,
        label="Rosseland optical depth",
    )
    ax1.set_xlabel("Height [Mm]", fontsize=12)
    ax1.set_ylabel("Rosseland Optical Depth τ", fontsize=12)
    ax1.set_title("Rosseland Optical Depth Profile", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(fontsize=10)

    # Right panel: temperature vs tau
    temperature = atm.T
    ax2.semilogx(tau_rosseland, temperature[:-1], "r-", linewidth=2, label="Temperature")
    ax2.set_ylabel("Temperature [K]", fontsize=12)
    ax2.set_xlabel("Rosseland Optical Depth τ", fontsize=12)
    ax2.set_title("Temperature vs Optical Depth", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, which="both")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    tau_plot_file = "tau_rosseland_profile.pdf"
    plt.savefig(tau_plot_file, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ Tau Rosseland profile plot saved to {tau_plot_file}[/green]")
    # plt.show()


def get_depth_at_tau_values_from_full_opacity(
    atm: AtmosphericData,
    interpolated_opacity: NDArray[np.float64],
    wavelength_grid: NDArray[np.float64],
    tau_values: list[float],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """
    Get the atmospheric depth (height) for every subbin when it reaches
    specified optical depth based on full opacity values.

    Args:
        atm: Atmospheric data
        interpolated_opacity: Full opacity on the atmosphere T-p grid
            - shape: [n_layers, nbins, nsubbins]
        wavelength_grid: Wavelength grid for the opacity data [cm]
        tau_values: List of optical depth values to find depths for

    Returns:
        - height_at_tau_index: Indices of the atmospheric layers where tau values are reached
            - shape: [nbins * nsubbins, len(tau_values)]
        - height_at_tau: Heights at specified tau values [km]
            - shape: [nbins * nsubbins, len(tau_values)]
    """
    console.print("\n[cyan]Atmospheric Depths at Specified Rosseland Optical Depths:[/cyan]")

    heights = atm.z[::-1]  # cm, reversed to go from top to bottom
    opacity = interpolated_opacity.reshape(atm.nlevels, -1)  # shape: [n_layers, nbins * nsubbins]

    if opacity.shape[1] != wavelength_grid.shape[0]:
        console.print(
            f"[red]Error: opacity shape {opacity.shape} does not match wavelength grid shape {wavelength_grid.shape}[/red]"
        )
        raise ValueError(f"Inconsistent shapes for opacity {opacity.shape} and wavelength grid {wavelength_grid.shape}")

    # Initialize output array
    n_wavelengths = opacity.shape[1]
    console.print(f"n_wavelengths: {n_wavelengths}")
    height_at_tau = np.zeros((n_wavelengths, len(tau_values)), dtype=np.float64)

    # Compute cumulative optical depth for all wavelengths at once
    density_integrand = atm.rho[:, np.newaxis]  # shape: [n_layers, 1]
    kappa_integrand = opacity  # shape: [n_layers, n_wavelengths]

    # Cumulative tau from top of atmosphere downward
    tau_profile = cumulative_trapezoid(
        kappa_integrand * density_integrand, x=heights, axis=0, initial=0
    )  # shape: [n_layers, n_wavelengths]

    # For each wavelength, find heights where tau reaches specified values
    height_at_tau_index = np.zeros((n_wavelengths, len(tau_values)), dtype=np.int64)
    for wl_idx in range(n_wavelengths):
        for tau_idx, tau_val in enumerate(tau_values):
            layer_idx = np.searchsorted(tau_profile[:, wl_idx], tau_val)
            height_at_tau_index[wl_idx, tau_idx] = layer_idx
            if layer_idx >= len(heights):
                height_at_tau[wl_idx, tau_idx] = heights[-1] / 1e5  # Convert to km
            else:
                height_at_tau[wl_idx, tau_idx] = heights[layer_idx] / 1e5  # Convert to km

    return height_at_tau_index, height_at_tau


def plot_height_at_tau_values(
    wavelength_grid_input: NDArray[np.float64],
    height_at_tau: NDArray[np.float64],
    tau_values: list[float],
    output_file: str = "height_at_tau_values.pdf",
) -> None:
    """
    Plot the difference in atmospheric heights as a function of wavelength
    for specified optical depth values.

    Args:
        wavelength_grid: Wavelength grid [cm]
        height_at_tau: Heights at specified tau values [km]
            - shape: [n_wavelengths, len(tau_values)]
        tau_values: List of optical depth values
        output_file: Output filename for the plot
    """
    console.print("\n[cyan]Plotting height differences at specified tau values...[/cyan]")

    n_tau = len(tau_values)
    wavelength_grid = wavelength_grid_input * 1e7  # convet to nm
    TITLE_FONTSIZE = 8
    LEGEND_FONTSIZE = 5

    if n_tau == 2:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Left panel: Absolute heights at each tau
    ax1 = axes[0]
    cmap = matplotlib.colormaps.get_cmap("berlin")
    for i, tau_val in enumerate(tau_values):
        color = cmap(i / max(1, n_tau - 1))
        ax1.plot(
            wavelength_grid,
            height_at_tau[:, i],
            linewidth=0.2,
            color=color,
            label=f"τ = {tau_val}",
        )

    ax1.set_xlabel("Wavelength [nm]", fontsize=12)
    ax1.set_ylabel("Height [km]", fontsize=12)
    ax1.set_title(
        "Atmospheric Height at Optical Depth",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=LEGEND_FONTSIZE, loc="best", reverse=True)
    # ax1.set_xlim(0, 50)
    ax1.set_xlim(wavelength_grid.min(), wavelength_grid.max())

    # Right panel: Height differences between consecutive tau values
    ax2 = axes[1]
    if n_tau > 1:
        for i in range(n_tau - 1):
            height_diff = height_at_tau[:, i + 1] - height_at_tau[:, i]
            color = cmap((i + 0.5) / max(1, n_tau - 1))
            ax2.plot(
                wavelength_grid,
                height_diff,
                linewidth=0.2,
                color=color,
                label=f"Δh (τ={tau_values[i + 1]} - τ={tau_values[i]})",
            )

        ax2.set_xlabel("Wavelength [nm]", fontsize=12)
        ax2.set_ylabel("Height Difference [km]", fontsize=12)
        ax2.set_title(
            "Height Difference vs Wavelength",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=LEGEND_FONTSIZE, loc="best")
        ax2.set_xlim(wavelength_grid.min(), wavelength_grid.max())
        # ax2.set_xlim(0, 50)
        ax2.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            "Need at least 2 tau values\nto plot differences",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=12,
        )
        ax2.set_xlabel("Wavelength [nm]", fontsize=12)
        ax2.set_ylabel("Height Difference [km]", fontsize=12)
        ax2.set_title(
            "Height Difference vs Wavelength",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )

    # Right panel: Histogram of heights at each tau
    ax3 = axes[2]
    bins = np.arange(0, 900, 10)
    for i, tau_val in enumerate(tau_values):
        color = cmap(i / max(1, n_tau - 1))
        hist = ax3.hist(
            height_at_tau[:, i],
            bins=bins,
            alpha=0.6,
            color=color,
            label=f"τ = {tau_val}",
            edgecolor="black",
            linewidth=0.5,
            align="left",
        )

    ax3.set_xlabel("Height [km]", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title(
        "Height Distribution at Optical Depths",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    # Right panel: Histogram of differences in heights at each tau
    bins = np.arange(0, 310, 10)
    if n_tau == 2:
        ax4 = axes[3]
        hist = ax4.hist(
            height_at_tau[:, 1] - height_at_tau[:, 0],
            bins=bins,
            alpha=0.6,
            color="gray",
            label="Height Difference (τ2 - τ1)",
            edgecolor="black",
            linewidth=0.5,
            align="left",
        )
        ax4.set_xlabel("Height diff [km]", fontsize=12)
        ax4.set_ylabel("Count", fontsize=12)
        ax4.set_title(
            "Height Difference Distribution at Optical Depths",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )
        ax4.grid(True, alpha=0.3, axis="y")
        ax4.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ Height at tau plot saved to {output_file}[/green]")


def plot_tau_rosselend_at_tau_lambda_one_vs_wavelength(
    tau_rosseland: NDArray[np.float64],
    wavelength_grid_input: NDArray[np.float64],
    group_tau_edges: NDArray[np.float64],
    group_lam_edges: NDArray[np.float64],
    lambda_bin_edges: list[int | float],
    output_file: str = "tau_rosseland_at_tau_lambda_one.jpg",
    use_2d_histogram: bool = True,
    band_index: NDArray[np.int32] | None = None,
) -> None:
    """
    Plot the Rosseland optical depth at the height where the optical depth
    at each wavelength subbin is equal to one, as a function of wavelength.

    The fixed lambda edges are drawn as vertical lines. The tau-group edges are
    drawn *per lambda cell* as horizontal segments spanning only that cell's
    wavelength window, so per-cell edges appear discontinuous (they "jump")
    across the vertical lambda lines.

    Args:
        tau_rosseland: Rosseland optical depth profile [n_bins*n_subbins]
        wavelength_grid: Wavelength grid [Angstrom]
        tau_edges_per_lambda: Per-lambda-cell -log10(tau) edge lists.
        lambda_bin_edges: Wavelength (log10 Angstrom) edges to plot.
        output_file: Output filename for the plot
        use_2d_histogram: If True, plot as 2D histogram instead of scatter plot
        band_index: Optional (lambda cell, tau) group index per sub-bin (same
            length as tau_rosseland). If provided, the topmost (max -log10(τ))
            and lowermost (min -log10(τ)) sub-bin of each group are circled red.
    """
    console.print("\n[cyan]Plotting τ_Rosseland at τ_λ=1 vs Wavelength...[/cyan]")
    wavelength_grid = wavelength_grid_input * 1e8  # Angstrom
    f, ax = plt.subplots(figsize=(10, 6))

    x_data = np.log10(wavelength_grid)
    y_data = -np.log10(tau_rosseland)

    if use_2d_histogram:
        # 2D histogram
        h = ax.hist2d(x_data, y_data, bins=[100, 100], cmap="viridis", cmin=1)
        plt.colorbar(h[3], ax=ax, label="Count")
    else:
        ax.scatter(x=x_data, y=y_data, s=1)

    ax.set_xlabel(r"$\log_{10} \lambda \, [\AA]$", fontsize=12)
    ax.set_ylabel(
        r"-$\log_{10} \tau_\text{Ros} \left( d \left( \tau_\lambda = 1 \right) \right)$",
        fontsize=12,
    )
    # ax.set_title("Rosseland Optical Depth at τ_λ=1 vs Wavelength", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3, 5)
    ax.set_ylim(-1, 7)

    # Draw each group's (tau, lambda) box outline. Horizontal tau edges are confined
    # to the group's lambda window, and vertical lambda edges to its tau range, so a
    # lambda cut only appears within tau slots that are actually lambda-split (and
    # per-cell-optimized tau edges still "jump" across the lambda lines).
    n_groups = int(group_tau_edges.shape[0])
    for g in range(n_groups):
        x_lo = float(group_lam_edges[g, 0])
        x_hi = float(group_lam_edges[g, 1])
        y_lo = float(group_tau_edges[g, 0])
        y_hi = float(group_tau_edges[g, 1])
        ax.hlines([y_lo, y_hi], xmin=x_lo, xmax=x_hi, color="k", lw=0.8)
        ax.vlines([x_lo, x_hi], ymin=y_lo, ymax=y_hi, color="k", lw=0.8)

    if band_index is not None:
        if band_index.shape[0] != tau_rosseland.shape[0]:
            raise ValueError(
                f"band_index length ({band_index.shape[0]}) must match tau_rosseland length ({tau_rosseland.shape[0]})"
            )
        sel_x: list[float] = []
        sel_y: list[float] = []
        for g in range(n_groups):
            mask = band_index == g
            if not np.any(mask):
                continue
            member_y = y_data[mask]
            member_x = x_data[mask]
            i_top = int(np.argmax(member_y))
            i_bot = int(np.argmin(member_y))
            sel_x.extend([float(member_x[i_top]), float(member_x[i_bot])])
            sel_y.extend([float(member_y[i_top]), float(member_y[i_bot])])
        ax.scatter(
            sel_x,
            sel_y,
            s=120,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            zorder=5,
            label="topmost / lowermost per (λ, τ) group",
        )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ τ_Rosseland at τ_λ=1 plot saved to {output_file}[/green]")


def build_group_index_maps(
    tau_edges_per_lambda: list[list[float]],
) -> tuple[list[int], int, NDArray[np.int64], NDArray[np.int64]]:
    """
    Flatten per-lambda-cell tau-groups into a single group-index space.

    Each lambda cell ``ell`` has its own list of tau edges (``nTau[ell] + 1`` of
    them, possibly a different count per cell). Group ``g`` enumerates
    ``(lambda cell, tau index)`` pairs in row-major order:
    ``g = offsets[ell] + t`` with ``offsets[ell] = sum(nTau[:ell])``.

    Returns:
        offsets:       offsets[ell] = first group id of lambda cell ell.
        n_groups:      total number of (cell, tau) groups = sum(nTau).
        group_to_cell: [n_groups] lambda cell of each group.
        group_to_tau:  [n_groups] tau index within its cell for each group.

    With a single lambda cell, ``g == tau index`` (backward compatible with the
    old tau-only band index).
    """
    offsets: list[int] = []
    group_to_cell: list[int] = []
    group_to_tau: list[int] = []
    acc = 0
    for cell, edges in enumerate(tau_edges_per_lambda):
        offsets.append(acc)
        n_tau = len(edges) - 1
        for t in range(n_tau):
            group_to_cell.append(cell)
            group_to_tau.append(t)
        acc += n_tau
    return (
        offsets,
        acc,
        np.asarray(group_to_cell, dtype=np.int64),
        np.asarray(group_to_tau, dtype=np.int64),
    )


def build_group_specs_per_cell(
    tau_edges_per_lambda: list[list[float]],
    lambda_bin_edges: list[int | float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Per-(lambda cell, tau slot) group descriptor, in build_group_index_maps order
    (cell-major: for cell ell, for tau slot t). Used so the per-cell grouping and
    the shared-tau + flags grouping both feed the same downstream code.

    Returns:
        group_tau_edges: [n_groups, 2] -log10(tau) (lo, hi) per group.
        group_lam_edges: [n_groups, 2] log10(lambda/A) (lo, hi) per group.
    """
    lam = [float(e) for e in lambda_bin_edges]
    tau_rows: list[tuple[float, float]] = []
    lam_rows: list[tuple[float, float]] = []
    for cell, edges in enumerate(tau_edges_per_lambda):
        for t in range(len(edges) - 1):
            tau_rows.append((float(edges[t]), float(edges[t + 1])))
            lam_rows.append((lam[cell], lam[cell + 1]))
    return (
        np.asarray(tau_rows, dtype=np.float64).reshape(-1, 2),
        np.asarray(lam_rows, dtype=np.float64).reshape(-1, 2),
    )


def parse_split_lambda(spec: str) -> list[bool]:
    """Parse a --split-lambda spec into a list of booleans (one per tau group).

    Accepts a 0/1 string ("00111100") or comma/space-separated tokens
    (true/false/1/0/t/f/yes/no). Order matches the tau groups.
    """
    s = spec.strip()
    if not s:
        return []
    tokens = s.replace(",", " ").split() if ("," in s or " " in s or "\t" in s) else list(s)
    truthy = {"1", "t", "true", "y", "yes"}
    falsy = {"0", "f", "false", "n", "no"}
    out: list[bool] = []
    for tok in tokens:
        low = tok.lower()
        if low in truthy:
            out.append(True)
        elif low in falsy:
            out.append(False)
        else:
            raise typer.BadParameter(f"--split-lambda: cannot parse '{tok}' as a boolean")
    return out


def parse_lambda_per_tau(specs: list[str]) -> list[list[float]]:
    """Parse repeated ``--lambda-per-tau`` entries (one per tau group) into a list of
    per-group lambda-edge lists. Each entry is a comma/space-separated strictly
    increasing edge list with >= 2 values (exactly 2 = that group is not lambda-split).
    All groups must share the same outer window; that is checked by the caller.
    """
    out: list[list[float]] = []
    for s in specs:
        edges = [float(t) for t in s.replace(",", " ").split() if t]
        if len(edges) < 2:
            raise typer.BadParameter(f"--lambda-per-tau entry '{s}' needs >= 2 edges")
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise typer.BadParameter(f"--lambda-per-tau entry '{s}' must be strictly increasing")
        out.append(edges)
    return out


def build_group_specs_split_lambda(
    tau_bin_edges: list[float],
    lambda_bin_edges: list[int | float],
    split_along_lambda: list[bool],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """
    Shared-tau binning with a per-tau-group lambda-split flag (tau-outer).

    A single set of tau edges defines N tau groups (the *same* tau ranges at all
    wavelengths). For tau group k, ``split_along_lambda[k]`` decides whether it
    subdivides into the lambda cells (one group per cell) or stays a single group
    spanning the whole lambda range. Groups are enumerated slot-major: for k in
    0..N-1, the split cells (if any) then the single group.

    With a single lambda cell every group is "single" (the flag is a no-op).

    Returns:
        group_tau_edges:     [n_groups, 2] -log10(tau) (lo, hi).
        group_lam_edges:     [n_groups, 2] log10(lambda/A) (lo, hi).
        slot_cell_to_group:  [N, L] group id for (tau slot, lambda cell), -1 where
                             that slot is not lambda-split.
        slot_single_to_group:[N] group id for an unsplit tau slot, -1 where the
                             slot is lambda-split.
    """
    tau = [float(e) for e in tau_bin_edges]
    lam = [float(e) for e in lambda_bin_edges]
    n_tau = len(tau) - 1
    n_lambda = len(lam) - 1
    flags = [bool(b) for b in split_along_lambda]

    tau_rows: list[tuple[float, float]] = []
    lam_rows: list[tuple[float, float]] = []
    slot_cell_to_group = np.full((n_tau, n_lambda), -1, dtype=np.int64)
    slot_single_to_group = np.full(n_tau, -1, dtype=np.int64)

    g = 0
    for k in range(n_tau):
        if flags[k] and n_lambda > 1:
            for cell in range(n_lambda):
                tau_rows.append((tau[k], tau[k + 1]))
                lam_rows.append((lam[cell], lam[cell + 1]))
                slot_cell_to_group[k, cell] = g
                g += 1
        else:
            tau_rows.append((tau[k], tau[k + 1]))
            lam_rows.append((lam[0], lam[-1]))
            slot_single_to_group[k] = g
            g += 1

    return (
        np.asarray(tau_rows, dtype=np.float64).reshape(-1, 2),
        np.asarray(lam_rows, dtype=np.float64).reshape(-1, 2),
        slot_cell_to_group,
        slot_single_to_group,
    )


def assign_split_lambda(
    tau_rosseland: NDArray[np.float64],
    wavelength_grid_input: NDArray[np.float64],
    tau_bin_edges: list[float],
    lambda_bin_edges: list[int | float],
    slot_cell_to_group: NDArray[np.int64],
    slot_single_to_group: NDArray[np.int64],
) -> NDArray[np.int32]:
    """
    Assign sub-bins to shared-tau + per-group-lambda-flag groups.

    A sub-bin is placed by its shared tau slot k; if that slot is lambda-split it
    is further placed by its lambda cell, else it joins the slot's single group.
    Sub-bins outside the tau range or the overall lambda range are -1.
    """
    wavelength_grid = wavelength_grid_input * 1e8  # Angstrom
    x_data = np.log10(wavelength_grid)
    y_data = -np.log10(np.clip(tau_rosseland, 1.0e-300, None))

    tau_edges = np.asarray(tau_bin_edges, dtype=np.float64)
    lam_edges = np.asarray(lambda_bin_edges, dtype=np.float64)
    n_tau = len(tau_edges) - 1
    n_lambda = len(lam_edges) - 1

    k = np.digitize(y_data, tau_edges, right=False) - 1
    ell = np.digitize(x_data, lam_edges, right=False) - 1
    ok = (k >= 0) & (k < n_tau) & (ell >= 0) & (ell < n_lambda)

    group_index = np.full(x_data.shape, -1, dtype=np.int32)
    kk = k[ok]
    ll = ell[ok]
    # Split slots have slot_cell_to_group[k, ell] >= 0; unsplit slots use the single map.
    via_cell = slot_cell_to_group[kk, ll]
    out = np.where(via_cell >= 0, via_cell, slot_single_to_group[kk])
    group_index[ok] = out.astype(np.int32)
    return group_index


def build_group_specs_per_tau(
    tau_bin_edges: list[float],
    lambda_edges_per_tau: list[list[float]],
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """
    Shared-tau binning where each tau group carries its *own* lambda edges.

    A single set of tau edges defines N tau groups (same tau ranges at all
    wavelengths). Tau group k subdivides into ``len(lambda_edges_per_tau[k]) - 1``
    lambda sub-cells using that group's own lambda edges — so the wavelength split
    can differ (or be absent) per tau group, generalizing the shared-lambda +
    split-flag model (a group with edges ``[lam_min, lam_max]`` is unsplit; a group
    reusing the shared edges reproduces a flag=True group). Groups are enumerated
    tau-major: ``g = offsets[k] + j`` for lambda sub-cell j of tau group k.

    All groups must share the same outer lambda window (``lambda_edges_per_tau[k][0]``
    and ``[-1]`` equal for every k); only the interior cuts vary.

    Returns:
        group_tau_edges: [n_groups, 2] -log10(tau) (lo, hi).
        group_lam_edges: [n_groups, 2] log10(lambda/A) (lo, hi).
        offsets:         offsets[k] = first group id of tau group k.
    """
    tau = [float(e) for e in tau_bin_edges]
    n_tau = len(tau) - 1
    if len(lambda_edges_per_tau) != n_tau:
        raise ValueError(f"lambda_edges_per_tau has {len(lambda_edges_per_tau)} entries, expected n_tau={n_tau}")

    tau_rows: list[tuple[float, float]] = []
    lam_rows: list[tuple[float, float]] = []
    offsets: list[int] = []
    g = 0
    for k in range(n_tau):
        offsets.append(g)
        lam = [float(e) for e in lambda_edges_per_tau[k]]
        if len(lam) < 2:
            raise ValueError(f"tau group {k} needs >= 2 lambda edges, got {lam}")
        for j in range(len(lam) - 1):
            tau_rows.append((tau[k], tau[k + 1]))
            lam_rows.append((lam[j], lam[j + 1]))
            g += 1
    return (
        np.asarray(tau_rows, dtype=np.float64).reshape(-1, 2),
        np.asarray(lam_rows, dtype=np.float64).reshape(-1, 2),
        offsets,
    )


def assign_per_tau_lambda(
    tau_rosseland: NDArray[np.float64],
    wavelength_grid_input: NDArray[np.float64],
    tau_bin_edges: list[float],
    lambda_edges_per_tau: list[list[float]],
    offsets: list[int],
) -> NDArray[np.int32]:
    """
    Assign sub-bins to (tau group, that group's lambda sub-cell) groups.

    A sub-bin is placed by its shared tau slot k, then by its lambda sub-cell within
    tau group k's *own* lambda edges. Sub-bins outside the tau range, or outside a
    group's lambda window, are -1. Mirrors the digitize(right=False) convention of
    assign_split_lambda / assign_tau_to_bin.
    """
    x_data = np.log10(wavelength_grid_input * 1e8)  # log10 lambda [Angstrom]
    y_data = -np.log10(np.clip(tau_rosseland, 1.0e-300, None))  # -log10 tau
    tau_edges = np.asarray(tau_bin_edges, dtype=np.float64)
    n_tau = len(tau_edges) - 1
    if len(lambda_edges_per_tau) != n_tau:
        raise ValueError(f"lambda_edges_per_tau has {len(lambda_edges_per_tau)} entries, expected n_tau={n_tau}")

    k_all = np.digitize(y_data, tau_edges, right=False) - 1
    group_index = np.full(x_data.shape, -1, dtype=np.int32)
    for k in range(n_tau):
        in_k = k_all == k
        if not np.any(in_k):
            continue
        lam = np.asarray(lambda_edges_per_tau[k], dtype=np.float64)
        n_l = len(lam) - 1
        j = np.digitize(x_data[in_k], lam, right=False) - 1
        valid = (j >= 0) & (j < n_l)
        cell_groups = np.full(j.shape, -1, dtype=np.int32)
        cell_groups[valid] = (offsets[k] + j[valid]).astype(np.int32)
        group_index[in_k] = cell_groups
    return group_index


def assign_tau_to_bin(
    tau_rosseland: NDArray[np.float64],
    wavelength_grid_input: NDArray[np.float64],
    tau_edges_per_lambda: list[list[float]],
    lambda_bin_edges: list[int | float],
) -> NDArray[np.int32]:
    """
    Assign sub-bins to (lambda cell, tau group) groups.

    Each sub-bin is first placed into a lambda cell by its wavelength, then into
    a tau group using *that cell's* tau edges (edges may differ per cell). The
    returned value is the flattened group index ``g = offsets[cell] + tau_idx``
    (see build_group_index_maps), or -1 if the sub-bin falls outside the
    configured lambda range or its cell's tau range.

    Args:
        tau_rosseland: Tau values at sub-bin wavelengths [n_bins*n_subbins]
        wavelength_grid_input: Wavelength at sub-bin centers [cm]. [n_bins*n_subbins]
        tau_edges_per_lambda: Per-lambda-cell -log10(tau) edge lists (length n_lambda).
        lambda_bin_edges: Wavelength (log10 Angstrom) edges, length n_lambda+1.

    Returns:
        Group index per sub-bin wavelength point, in [0, n_groups-1] or -1.
    """
    wavelength_grid = wavelength_grid_input * 1e8  # Angstrom
    x_data = np.log10(wavelength_grid)
    # Prevent log10(0) when tau is numerically tiny.
    y_data = -np.log10(np.clip(tau_rosseland, 1.0e-300, None))

    lambda_edges = np.asarray(lambda_bin_edges, dtype=np.float64)
    n_lambda_bins = len(lambda_edges) - 1
    if len(tau_edges_per_lambda) != n_lambda_bins:
        raise ValueError(
            f"tau_edges_per_lambda has {len(tau_edges_per_lambda)} cells, expected n_lambda={n_lambda_bins}"
        )

    offsets, _n_groups, _g2cell, _g2tau = build_group_index_maps(tau_edges_per_lambda)
    lambda_idx = np.digitize(x_data, lambda_edges, right=False) - 1

    group_index = np.full(x_data.shape, -1, dtype=np.int32)
    for cell in range(n_lambda_bins):
        in_cell = lambda_idx == cell
        if not np.any(in_cell):
            continue
        tau_edges = np.asarray(tau_edges_per_lambda[cell], dtype=np.float64)
        n_tau_bins = len(tau_edges) - 1
        tau_idx = np.digitize(y_data[in_cell], tau_edges, right=False) - 1
        valid = (tau_idx >= 0) & (tau_idx < n_tau_bins)
        cell_groups = np.full(tau_idx.shape, -1, dtype=np.int32)
        cell_groups[valid] = (offsets[cell] + tau_idx[valid]).astype(np.int32)
        group_index[in_cell] = cell_groups

    return group_index


def sort_weighted_opacity_per_tau_bin(
    atm: AtmosphericData,
    odf: ODFData,
    interpolated_opacity: NDArray[np.float64],
    tau_rosseland: NDArray[np.float64],
    band_index: NDArray[np.int32],
    group_tau_edges: NDArray[np.float64],
    wavelength_grid_subbins_centers: NDArray[np.float64],
    write_debug_json: bool = True,
    verbose: bool = True,
) -> dict[int, dict[str, NDArray[np.float64] | NDArray[np.int64] | float | int | bool]]:
    r"""
    For each group, locate the two atmospheric layers (top and bottom) that
    bracket the group's Rosseland-tau range, then build a sorted distribution of
    weighted opacities at each (T, p) point.

    Per group g, using its -log10(tau) edges ``group_tau_edges[g] = (lo, hi)``:
      1. tau_low = 10^(-hi), tau_high = 10^(-lo)
         (matches digitize(right=False) in the assign step).
      2. j_top, j_bot = searchsorted(tau_full, tau_low/tau_high) on the
         reversed-z tau profile;
      3. For sub-bins with band_index == g, read opacities at
         interpolated_opacity[i_top] and interpolated_opacity[i_bot].
      4. Weight by Δλ_subbin × B_λ(λ_subbin, T) using the corresponding T.
      5. argsort weighted_kappa.

    Args:
        atm: Atmospheric model.
        odf: ODF table (uses .wavelength_grid edges and .subbin weights).
        interpolated_opacity: ODF+continuum opacity on the atm T-p grid,
            shape [n_layers, nbins, nsubbins], original atm ordering.
        tau_rosseland: Rosseland tau profile, shape [n_layers - 1],
            reversed-z ordering (heights = atm.z[::-1]).
        band_index: Group assignment per sub-bin, shape [nbins*nsubbins],
            values in [0, n_groups-1] or -1 for unassigned.
        group_tau_edges: [n_groups, 2] -log10(tau) (lo, hi) per group. The
            atmosphere-top clamp on the first tau slot is applied by the caller
            before this descriptor is built.
        wavelength_grid_subbins_centers: Sub-bin center wavelengths [cm],
            shape [nbins*nsubbins].

    Returns:
        Dict keyed by group index. Empty groups → {"empty": True, "members": 0}.
        Non-empty groups contain T_top, p_top, i_top, T_bot, p_bot, i_bot,
        member_indices, kappa_top, kappa_bot, weights_top, weights_bot,
        weighted_kappa_top, weighted_kappa_bot, sort_idx_top, sort_idx_bot,
        sorted_weighted_kappa_top, sorted_weighted_kappa_bot.
    """
    if verbose:
        console.print("\n[cyan]Sorting weighted opacities per group at top/bot TP points...[/cyan]")

    n_layers = atm.nlevels
    n_groups = int(group_tau_edges.shape[0])

    if interpolated_opacity.shape[0] != n_layers:
        raise ValueError(f"interpolated_opacity has {interpolated_opacity.shape[0]} layers, expected {n_layers}")
    if tau_rosseland.shape[0] != n_layers - 1:
        raise ValueError(f"tau_rosseland has {tau_rosseland.shape[0]} layers, expected {n_layers - 1}")
    if odf.wavelength_grid is None or odf.subbin is None:
        raise ValueError("ODF wavelength_grid and subbin weights are required.")

    # tau_rosseland[j-1] is τ at heights[j] (heights = atm.z[::-1]) because
    # compute_tau_rosseland used cumulative_trapezoid(initial=0)[1:] which
    # strips the leading 0 at heights[0]. Re-prepend 0 so tau_full[j] is τ
    # at heights[j].
    tau_full = np.concatenate(([0.0], tau_rosseland))

    bin_widths = np.diff(odf.wavelength_grid)[:, np.newaxis]
    subbin_widths_flat = (bin_widths * odf.subbin).reshape(-1)

    if subbin_widths_flat.shape[0] != wavelength_grid_subbins_centers.shape[0]:
        raise ValueError(
            f"sub-bin width ({subbin_widths_flat.shape[0]}) and center "
            f"({wavelength_grid_subbins_centers.shape[0]}) lengths differ"
        )
    if band_index.shape[0] != subbin_widths_flat.shape[0]:
        raise ValueError(
            f"band_index length ({band_index.shape[0]}) does not match nbins*nsubbins ({subbin_widths_flat.shape[0]})"
        )

    results: dict[
        int,
        dict[str, NDArray[np.float64] | NDArray[np.int64] | float | int | bool],
    ] = {}

    if verbose:
        console.print(f"First and last 10 atm.T values: {atm.T[:10]} ... {atm.T[-10:]}")

    for g in range(n_groups):
        neglogtau_lo = float(group_tau_edges[g, 0])
        neglogtau_hi = float(group_tau_edges[g, 1])
        tau_low = 10.0 ** (-neglogtau_hi)
        tau_high = 10.0 ** (-neglogtau_lo)

        if verbose:
            console.print(
                f"\nProcessing group {g}: -log10(tau) in [{neglogtau_lo:.2f}, {neglogtau_hi:.2f}] "
                f"→ tau in [{tau_low:.2e}, {tau_high:.2e}]"
            )

        j_top = int(np.clip(np.searchsorted(tau_full, tau_low, side="left"), 0, n_layers - 1))
        j_bot = int(np.clip(np.searchsorted(tau_full, tau_high, side="left"), 0, n_layers - 1))

        # i_top = n_layers - 1 - j_top
        # i_bot = n_layers - 1 - j_bot
        i_top = j_top
        i_bot = j_bot

        member_idx = np.flatnonzero(band_index == g)
        if member_idx.size == 0:
            results[g] = {"empty": True, "members": 0}
            continue

        kappa_top = interpolated_opacity[i_top].reshape(-1)[member_idx]
        kappa_bot = interpolated_opacity[i_bot].reshape(-1)[member_idx]
        widths = subbin_widths_flat[member_idx]
        lambdas = wavelength_grid_subbins_centers[member_idx]

        T_top = float(atm.T[i_top])
        T_bot = float(atm.T[i_bot])

        weights_top = widths * planck_function(lambdas, T_top)
        weights_bot = widths * planck_function(lambdas, T_bot)

        weighted_top = kappa_top * weights_top
        weighted_bot = kappa_bot * weights_bot

        sort_idx_top = np.argsort(weighted_top)
        sort_idx_bot = np.argsort(weighted_bot)

        results[g] = {
            "members": int(member_idx.size),
            "T_top": T_top,
            "p_top": float(atm.p[i_top]),
            "i_top": i_top,
            "T_bot": T_bot,
            "p_bot": float(atm.p[i_bot]),
            "i_bot": i_bot,
            "member_indices": member_idx,
            "kappa_top": kappa_top,
            "kappa_bot": kappa_bot,
            "weights_top": weights_top,
            "weights_bot": weights_bot,
            "weighted_kappa_top": weighted_top,
            "weighted_kappa_bot": weighted_bot,
            "sort_idx_top": sort_idx_top,
            "sort_idx_bot": sort_idx_bot,
            "sorted_weighted_kappa_top": weighted_top[sort_idx_top],
            "sorted_weighted_kappa_bot": weighted_bot[sort_idx_bot],
        }

        if write_debug_json:
            debug_path = f"debug_tau_bin_{g}.json"
            serializable = {
                key: (val.tolist() if isinstance(val, np.ndarray) else val) for key, val in results[g].items()
            }
            with open(debug_path, "w") as f:
                json.dump(serializable, f, indent=2)

    return results


def plot_sorted_weighted_opacity_per_tau_bin(
    sorted_per_bin: dict[int, dict],
    group_tau_edges: NDArray[np.float64],
    group_lam_edges: NDArray[np.float64],
    output_file: str = "sorted_weighted_opacity_per_tau_bin.jpg",
    smooth_window: int = 7,
    overlay_breaks: bool = True,
    refine_mid: bool = True,
) -> None:
    """
    Plot sorted weighted opacities per group.

    For each group, draw a subplot with both 'top' (smaller-τ edge of the
    group) and 'bot' (larger-τ edge) sorted κ·Δλ·B_λ curves on a log y-axis.
    If overlay_breaks is True, run analyze_group on each curve and overlay
    vertical lines at the segmentation break points (b1, b2, optional b_mid).

    Args:
        sorted_per_bin: Output of sort_weighted_opacity_per_tau_bin (keyed by group g).
        group_tau_edges: [n_groups, 2] -log10(τ_Ros) (lo, hi) per group.
        group_lam_edges: [n_groups, 2] log10(λ/Å) (lo, hi) per group.
        output_file: Output figure path.
        smooth_window: Smoothing window passed to analyze_group.
        overlay_breaks: If True, draw piecewise-linear break lines.
        refine_mid: Forwarded to analyze_group (skip iterative b_mid refinement
            when False).
    """
    console.print("\n[cyan]Plotting sorted weighted opacities per group...[/cyan]")

    n_groups = int(group_tau_edges.shape[0])
    ncols = int(np.ceil(np.sqrt(n_groups)))
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for g in range(n_groups):
        ax = axes_flat[g]
        r = sorted_per_bin.get(g, {})
        edge_low = float(group_tau_edges[g, 0])
        edge_high = float(group_tau_edges[g, 1])
        title = (
            f"grp {g} (λ[{group_lam_edges[g, 0]:.2f},{group_lam_edges[g, 1]:.2f}], "
            f"τ[{edge_low:.2f},{edge_high:.2f}]): "
            rf"$-\log_{{10}}\tau_{{\rm Ros}}$"
        )

        if r.get("empty", False):
            ax.text(
                0.5,
                0.5,
                "empty",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=10, fontweight="bold")
            continue

        s_top = np.asarray(r["sorted_weighted_kappa_top"], dtype=np.float64)
        s_bot = np.asarray(r["sorted_weighted_kappa_bot"], dtype=np.float64)
        T_top = float(r["T_top"])
        T_bot = float(r["T_bot"])
        p_top = float(r["p_top"])
        p_bot = float(r["p_bot"])
        x = np.arange(s_top.size)

        # Mask non-positive entries so semilogy doesn't blow up.
        s_top_pos = np.where(s_top > 0, s_top, np.nan)
        s_bot_pos = np.where(s_bot > 0, s_bot, np.nan)

        ax.semilogy(
            x,
            s_top_pos,
            "-",
            lw=1.0,
            label=f"top  T={T_top:.0f}K, p={p_top:.2e}",
            color="C0",
        )
        ax.semilogy(
            x,
            s_bot_pos,
            "-",
            lw=1.0,
            label=f"bot  T={T_bot:.0f}K, p={p_bot:.2e}",
            color="C1",
        )

        ax.set_xlabel("sorted sub-bin index", fontsize=10)
        ax.set_ylabel(r"$\kappa \cdot \Delta\lambda \cdot B_\lambda$", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")

        if overlay_breaks and s_bot.size > 10:
            try:
                res = analyze_group(s_bot, smooth_window=smooth_window, refine_mid=refine_mid)
            except Exception as e:  # pragma: no cover
                console.print(f"[yellow]group {g} bot: analyze_group failed: {e}[/yellow]")
            else:
                seg = res["seg"]
                ax.axvline(seg["b1"], color="C1", ls=":", lw=1.0, alpha=0.9)
                ax.axvline(seg["b2"], color="C1", ls=":", lw=1.0, alpha=0.9)
                if seg.get("split_mid", False):
                    ax.axvline(seg["b_mid"], color="C1", ls="--", lw=1.0, alpha=0.7)

        ax.legend(fontsize=8, loc="lower right")

    for j in range(n_groups, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]✓ Sorted weighted opacity plot saved to {output_file}[/green]")


def _segment_defs(
    b1: int,
    b2: int,
    b_mid: int | None,
    split_mid: bool,
    n: int,
) -> list[tuple[str, int, int]]:
    """
    Segment (name, lo, hi) ranges in sorted-index space from analyze_group breaks.

    Returns 3 segments (low/mid/high) for the standard case, or 4
    (low/mid1/mid2/high) when the middle was further split (split_mid=True). The
    half-open ranges [lo, hi) index into a sorted-opacity curve of length n.
    """
    if split_mid:
        return [
            ("low", 0, b1 + 1),
            ("mid1", b1 + 1, b_mid + 1),
            ("mid2", b_mid + 1, b2),
            ("high", b2, n),
        ]
    return [
        ("low", 0, b1 + 1),
        ("mid", b1 + 1, b2),
        ("high", b2, n),
    ]


def compute_bot_segment_overlap_per_tau_bin(
    sorted_per_bin: dict[int, dict],
    group_tau_edges: NDArray[np.float64],
    group_lam_edges: NDArray[np.float64],
    smooth_window: int = 7,
    print_table: bool = True,
    verbose: bool = True,
    refine_mid: bool = True,
) -> dict[int, dict]:
    """
    Per group: run analyze_group on s_bot only, then reuse those break positions
    (b1, b2, optional b_mid) as segment boundaries in *sorted-index* space for
    both s_top and s_bot. For each segment, build the set of original sub-bin
    indices via sort_idx_top / sort_idx_bot and compute the overlap between the
    top and bot sets. By construction both sets have the same size (= segment
    length). Group τ/λ ranges come from the descriptor arrays.
    """
    n_groups = int(group_tau_edges.shape[0])
    overlaps: dict[int, dict] = {}

    table = Table(
        title="Segment overlap (bot-derived breaks applied to top) per (τ, λ) group",
        show_lines=False,
    )
    table.add_column("grp", justify="right")
    table.add_column("λ range", justify="center")
    table.add_column("τ range", justify="center")
    table.add_column("segment", justify="left")
    table.add_column("|A|=|B|", justify="right")
    table.add_column("|A∩B|", justify="right")
    table.add_column("overlap %", justify="right")

    for g in range(n_groups):
        tau_label = f"[{group_tau_edges[g, 0]:.2f}, {group_tau_edges[g, 1]:.2f}]"
        cell_label = f"[{group_lam_edges[g, 0]:.2f}, {group_lam_edges[g, 1]:.2f}]"
        r = sorted_per_bin.get(g, {})

        if r.get("empty", False) or "sorted_weighted_kappa_bot" not in r:
            table.add_row(str(g), cell_label, tau_label, "—", "—", "—", "—")
            overlaps[g] = {"empty": True}
            continue

        s_bot = np.asarray(r["sorted_weighted_kappa_bot"], dtype=np.float64)
        member_indices = np.asarray(r["member_indices"])
        sort_idx_top = np.asarray(r["sort_idx_top"])
        sort_idx_bot = np.asarray(r["sort_idx_bot"])
        n = s_bot.size

        if n <= 10:
            table.add_row(str(g), cell_label, tau_label, "—", str(n), "—", "—")
            overlaps[g] = {"too_small": True, "n": int(n)}
            continue

        try:
            res = analyze_group(
                s_bot,
                smooth_window=smooth_window,
                max_tail_frac_high=0.1,
                refine_mid=refine_mid,
            )
        except Exception as e:
            if verbose:
                console.print(f"[yellow]group {g}: analyze_group failed: {e}[/yellow]")
            table.add_row(str(g), cell_label, tau_label, "—", str(n), "—", "—")
            overlaps[g] = {"failed": True, "error": str(e)}
            continue

        seg = res["seg"]
        b1 = int(seg["b1"])
        b2 = int(seg["b2"])
        split_mid = bool(seg.get("split_mid", False))
        b_mid = int(seg["b_mid"]) if split_mid else None

        seg_defs = _segment_defs(b1, b2, b_mid, split_mid, n)

        bin_overlaps: dict[str, dict] = {}
        for i, (name, lo, hi) in enumerate(seg_defs):
            idx_range = slice(lo, hi)
            set_top = set(member_indices[sort_idx_top[idx_range]].tolist())
            set_bot = set(member_indices[sort_idx_bot[idx_range]].tolist())
            size = len(set_top)
            n_inter = len(set_top & set_bot)
            pct = (n_inter / size * 100.0) if size > 0 else 0.0

            bin_overlaps[name] = {
                "lo": lo,
                "hi": hi,
                "set_top": set_top,
                "set_bot": set_bot,
                "size": size,
                "n_intersection": n_inter,
                "overlap_pct": pct,
            }

            table.add_row(
                str(g) if i == 0 else "",
                cell_label if i == 0 else "",
                tau_label if i == 0 else "",
                name,
                str(size),
                str(n_inter),
                f"{pct:.1f}",
            )

        overlaps[g] = {
            "b1": b1,
            "b2": b2,
            "b_mid": b_mid,
            "split_mid": split_mid,
            "segments": bin_overlaps,
        }

    if print_table:
        console.print(table)
    return overlaps


def build_split_band_index(
    sorted_per_bin: dict[int, dict],
    n_subbin_points: int,
    n_groups: int,
    n_splits: int = 3,
    smooth_window: int = 7,
) -> NDArray[np.int32]:
    """
    Build a split-resolved band index from the per-group sorted-opacity curves.

    Each group g (a (lambda cell, tau) pair; with a single lambda cell g is just
    the tau index) is subdivided into ``n_splits`` opacity segments (low/mid/high)
    by running analyze_group on its bot-sorted weighted-opacity curve with
    ``refine_mid=False`` (so exactly 3 segments — no mid split). Every member
    sub-bin of group g is assigned the combined band index ``g * n_splits + seg``,
    where ``seg`` is its segment (0=low, 1=mid, 2=high) by rank in the bot-sorted
    curve. Unassigned sub-bins stay -1.

    The result is suitable as the ``band_index`` argument to
    calculate_tau_bin_opacities with ``n_bins = n_groups * n_splits``, producing
    a ``[nt, np, n_groups*n_splits]`` opacity table whose band axis factorizes as
    ``band -> (g = band // n_splits, seg = band % n_splits)``.

    Args:
        sorted_per_bin: Output of sort_weighted_opacity_per_tau_bin (keyed by group g).
        n_subbin_points: Total number of flattened wavelength sub-bins (nbins * nsubbins).
        n_groups: Number of (lambda cell, tau) groups (= sum of tau-groups over cells).
        n_splits: Number of opacity segments per group (fixed 3: low/mid/high).
        smooth_window: Smoothing window forwarded to analyze_group.

    Returns:
        split_band_index: int32 array of shape [n_subbin_points], values in
            [0, n_groups*n_splits) for assigned sub-bins, -1 otherwise.
    """
    split_band_index = np.full(n_subbin_points, -1, dtype=np.int32)

    for g in range(n_groups):
        r = sorted_per_bin.get(g, {})
        if r.get("empty", False) or "sorted_weighted_kappa_bot" not in r:
            continue

        member_indices = np.asarray(r["member_indices"])
        sort_idx_bot = np.asarray(r["sort_idx_bot"])
        s_bot = np.asarray(r["sorted_weighted_kappa_bot"], dtype=np.float64)
        n = s_bot.size

        if n <= 10:
            # Too few points to segment reliably: assign the whole group to split 0 (low).
            split_band_index[member_indices] = g * n_splits
            console.print(f"[yellow]group {g}: only {n} members (<=10); assigned all to split 0 (low).[/yellow]")
            continue

        try:
            res = analyze_group(
                s_bot,
                smooth_window=smooth_window,
                max_tail_frac_high=0.1,
                refine_mid=False,
            )
        except Exception as e:
            split_band_index[member_indices] = g * n_splits
            console.print(f"[yellow]group {g}: analyze_group failed ({e}); assigned all to split 0 (low).[/yellow]")
            continue

        seg = res["seg"]
        b1 = int(seg["b1"])
        b2 = int(seg["b2"])
        # refine_mid=False guarantees split_mid=False -> exactly n_splits (3) segments.
        for seg_id, (_name, lo, hi) in enumerate(_segment_defs(b1, b2, None, False, n)):
            if hi <= lo:
                continue
            members = member_indices[sort_idx_bot[lo:hi]]
            split_band_index[members] = g * n_splits + seg_id

    return split_band_index


def optimize_tau_bin_edges(
    atm: AtmosphericData,
    odf: ODFData,
    interpolated_opacity: NDArray[np.float64],
    tau_rosseland: NDArray[np.float64],
    tau_rosseland_at_tau_lambda_one: NDArray[np.float64],
    wavelength_grid_subbins_centers: NDArray[np.float64],
    max_height_idx: int,
    initial_tau_bin_edges: list[float],
    lambda_bin_edges: list[int | float],
    threshold: float = 0.70,
    max_bins: int = 8,
    adjust_steps: tuple[float, ...] = (0.10, 0.05, 0.02),
    max_outer_iters: int = 50,
    smooth_window: int = 7,
    refine_mid: bool = True,
) -> list[list[float]]:
    """
    Greedy adjust+insert search over tau edges, run INDEPENDENTLY within each
    lambda cell, to push the high-segment overlap (from
    compute_bot_segment_overlap_per_tau_bin) above `threshold` in every
    non-empty (cell, tau) group. Within a cell it nudges interior edges on a
    small step grid; when no nudge improves the minimum score it inserts a new
    edge at the midpoint of the worst tau group; it stops when all groups clear
    the threshold, when stuck at `max_bins`, or at `max_outer_iters`.

    The lambda edges stay fixed; only tau edges move, so each cell can end with
    a different number of tau groups (the per-cell "jump").

    Returns tau_edges_per_lambda: one optimized tau-edge list per lambda cell.
    """
    threshold_pct = threshold * 100.0
    lambda_edges = [float(e) for e in lambda_bin_edges]
    n_lambda = len(lambda_edges) - 1

    def _valid_monotone(edges: list[float]) -> bool:
        return all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))

    def _high_score(overlaps: dict, k: int) -> float:
        rec = overlaps.get(k, {})
        segs = rec.get("segments")
        if not segs or "high" not in segs:
            return 0.0
        return float(segs["high"]["overlap_pct"])

    def _optimize_one_lambda_cell(cell: int, window: list[float]) -> list[float]:
        # Restrict to this lambda cell by passing a single-cell window, so only
        # this column's sub-bins participate; groups are then just tau indices.
        def _evaluate(edges: list[float]) -> dict:
            # Membership uses the un-clamped edges; the descriptor for sort/overlap
            # gets the atmosphere-top clamp on the first tau slot (as the original
            # in-sort clamp did), so top/bot layer lookup is unchanged.
            band_idx = assign_tau_to_bin(
                tau_rosseland_at_tau_lambda_one,
                wavelength_grid_subbins_centers,
                tau_edges_per_lambda=[list(edges)],
                lambda_bin_edges=window,
            )
            clamped = list(edges)
            clamped[0] = -np.log10(tau_rosseland[max_height_idx] + 0.2)
            gt, gl = build_group_specs_per_cell([clamped], window)
            sorted_per_bin = sort_weighted_opacity_per_tau_bin(
                atm=atm,
                odf=odf,
                interpolated_opacity=interpolated_opacity,
                tau_rosseland=tau_rosseland,
                band_index=band_idx,
                group_tau_edges=gt,
                wavelength_grid_subbins_centers=wavelength_grid_subbins_centers,
                write_debug_json=False,
                verbose=False,
            )
            return compute_bot_segment_overlap_per_tau_bin(
                sorted_per_bin,
                group_tau_edges=gt,
                group_lam_edges=gl,
                smooth_window=smooth_window,
                print_table=False,
                verbose=False,
                refine_mid=refine_mid,
            )

        def _scores(edges: list[float], overlaps: dict) -> list[float]:
            return [_high_score(overlaps, k) for k in range(len(edges) - 1)]

        console.print(
            f"\n[cyan]Optimizing tau edges for λ cell {cell} "
            f"(λ∈[{window[0]}, {window[1]}], threshold={threshold_pct:.1f}%, max_bins={max_bins})...[/cyan]"
        )
        edges = list(initial_tau_bin_edges)
        overlaps = _evaluate(edges)
        scores = _scores(edges, overlaps)
        min_score = min(scores) if scores else 0.0
        console.print(f"  cell {cell} iter   0: bins={len(edges) - 1:2d}  min_high={min_score:6.2f}  action=start")

        for it in range(1, max_outer_iters + 1):
            if min_score >= threshold_pct:
                console.print(f"[green]✓ cell {cell} converged at iter {it - 1}[/green]")
                return edges

            best_edges = None
            best_overlaps = None
            best_min = min_score

            for i in range(1, len(edges)):
                for step in adjust_steps:
                    for direction in (-1.0, +1.0):
                        cand = list(edges)
                        cand[i] = edges[i] + direction * step
                        if not _valid_monotone(cand):
                            continue
                        cand_overlaps = _evaluate(cand)
                        cand_scores = _scores(cand, cand_overlaps)
                        cand_min = min(cand_scores) if cand_scores else 0.0
                        if cand_min > best_min + 1e-9:
                            best_min = cand_min
                            best_edges = cand
                            best_overlaps = cand_overlaps

            if best_edges is not None:
                edges = best_edges
                scores = _scores(edges, best_overlaps)
                min_score = best_min
                console.print(
                    f"  cell {cell} iter {it:3d}: bins={len(edges) - 1:2d}  min_high={min_score:6.2f}  action=adjust"
                )
                continue

            if len(edges) - 1 >= max_bins:
                console.print(
                    f"[yellow]✗ cell {cell} stuck at cap (bins={len(edges) - 1}, min_high={min_score:.2f}%)[/yellow]"
                )
                return edges

            k_worst = int(np.argmin(scores))
            new_edge = (edges[k_worst] + edges[k_worst + 1]) / 2.0
            edges = list(edges[: k_worst + 1]) + [new_edge] + list(edges[k_worst + 1 :])
            overlaps = _evaluate(edges)
            scores = _scores(edges, overlaps)
            min_score = min(scores) if scores else 0.0
            console.print(
                f"  cell {cell} iter {it:3d}: bins={len(edges) - 1:2d}  min_high={min_score:6.2f}  "
                f"action=split@k={k_worst} new_edge={new_edge:.3f}"
            )

        console.print(
            f"[yellow]✗ cell {cell} max_outer_iters={max_outer_iters} reached (min_high={min_score:.2f}%)[/yellow]"
        )
        return edges

    tau_edges_per_lambda: list[list[float]] = []
    for cell in range(n_lambda):
        window = [lambda_edges[cell], lambda_edges[cell + 1]]
        tau_edges_per_lambda.append(_optimize_one_lambda_cell(cell, window))
    return tau_edges_per_lambda


def calculate_tau_bin_opacities(
    odf: ODFData,
    cont: ContinuumData,
    band_index: NDArray[np.int32],
    n_bins: int,
    tau_transition: float = 0.35,
) -> dict[str, NDArray[np.float64] | NDArray[np.int32]]:
    r"""
    Calculate tau-binned opacities.

    After we perform the tau-sorting and assign each sub-bin to a tau-wavelength bin, we can calculate the binned opacities for each bin.
    We do so by calculating the average of planck mean or rosseland mean opacities for all sub-bins that fall into the same tau-wavelength bin,
    weighted by their wavelength contribution to the total opacity in that bin.

    Following eq. 16
    $B_l=\sum_i \Delta v_i B_{\Delta v_i} \sum_{j(i, l)} w_{j(i, l)}$
    and eq. 17
    $\bar{\kappa}_{P, l}=\frac{1}{B_l} \sum_i \Delta v_i B_{\Delta v_i} \sum_{j(i, l)} w_{j(i, l)} \kappa_{i j(i, l)}$,

    Args:
        odf: ODF table containing line opacity and T/P grids
        cont: Continuum opacity table
        band_index: Assigned tau-wavelength band index per sub-bin wavelength point
            Shape: [nbins * nsubbins], values -1 for out-of-range
        n_bins: Number of tau bins in output (len(tau_bin_edges) - 1)
        tau_transition: Optical depth transition scale in Eq. 12

    Returns:
        Dictionary with:
          - "kappa_planck": Planck mean opacity [nt, np, n_bands]
          - "kappa_rosseland": Rosseland mean opacity [nt, np, n_bands]
          - "kappa_mixed": Eq. 12 mixed opacity [nt, np, n_bands]
          - "B_band": Band-integrated Planck source [nt, n_bands]
          - "dBdT_band": Band-integrated dB/dT source [nt, n_bands]
          - "members_per_band": Number of sub-bin wavelength points in each band [n_bands]
    """
    if odf.ODF is None or cont.kappa_abs is None:
        raise ValueError("ODF and continuum data must be loaded.")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")
    if tau_transition <= 0.0:
        raise ValueError(f"tau_transition must be > 0, got {tau_transition}")

    if cont.kappa_abs.shape != (odf.nt, odf.np, odf.nbins):
        raise ValueError(
            f"Continuum shape {cont.kappa_abs.shape} does not match expected {(odf.nt, odf.np, odf.nbins)}"
        )
    if odf.wavelength_grid is None or odf.subbin is None:
        raise ValueError("ODF wavelength grid and subbin weights are required.")

    wavelength_grid = odf.wavelength_grid
    subbin_weights = odf.subbin
    if wavelength_grid.shape[0] != odf.nbins + 1:
        raise ValueError(f"Wavelength grid length ({wavelength_grid.shape[0]}) must be nbins+1 ({odf.nbins + 1})")
    if subbin_weights.shape != (odf.nbins, odf.nsubbins):
        raise ValueError(f"Sub-bin weights shape {subbin_weights.shape} does not match ({odf.nbins}, {odf.nsubbins})")

    n_subbin_points = odf.nbins * odf.nsubbins
    if band_index.shape[0] != n_subbin_points:
        raise ValueError(f"Band index length ({band_index.shape[0]}) does not match nbins*nsubbins ({n_subbin_points})")

    valid_band_mask = band_index >= 0
    if not np.any(valid_band_mask):
        raise ValueError("No wavelength points were assigned to tau-wavelength bins.")
    if np.any(band_index[valid_band_mask] >= n_bins):
        raise ValueError(f"Found band_index >= n_bins ({n_bins}). Check tau-bin assignment.")

    n_bands = n_bins

    # Build sub-bin centers and widths from bin edges and per-bin relative weights.
    bin_widths = np.diff(wavelength_grid)[:, np.newaxis]
    subbin_widths_2d = bin_widths * subbin_weights
    subbin_offsets = np.cumsum(subbin_widths_2d, axis=1) - 0.5 * subbin_widths_2d
    subbin_centers_2d = wavelength_grid[:-1, np.newaxis] + subbin_offsets

    subbin_widths = subbin_widths_2d.reshape(-1)
    subbin_centers = subbin_centers_2d.reshape(-1)

    total_kappa = odf.ODF + cont.kappa_abs[..., np.newaxis]
    opacity_flat = total_kappa.reshape(odf.nt, odf.np, -1)

    # ODF tables store log10(T), convert to K for Planck weighting.
    temperature_1d = np.power(10.0, odf.T)
    temperature_2d = temperature_1d[:, np.newaxis]
    wavelength_2d = subbin_centers[np.newaxis, :]

    B_lambda = planck_function(wavelength_2d, temperature_2d)
    dB_dT = planck_derivative_analytic(wavelength_2d, temperature_2d)

    B_band = np.zeros((odf.nt, n_bands), dtype=np.float64)
    dBdT_band = np.zeros((odf.nt, n_bands), dtype=np.float64)
    kappa_planck = np.zeros((odf.nt, odf.np, n_bands), dtype=np.float64)
    kappa_rosseland = np.zeros((odf.nt, odf.np, n_bands), dtype=np.float64)
    members_per_band = np.zeros((n_bands,), dtype=np.int32)

    for band in range(n_bands):
        member_mask = band_index == band
        members_per_band[band] = int(np.sum(member_mask))
        if members_per_band[band] == 0:
            continue

        widths = subbin_widths[member_mask]
        B_sel = B_lambda[:, member_mask]
        dB_dT_sel = dB_dT[:, member_mask]
        kappa_sel = opacity_flat[:, :, member_mask]
        safe_kappa = np.clip(kappa_sel, 1.0e-300, None)

        weighted_B = B_sel * widths[np.newaxis, :]
        weighted_dBdT = dB_dT_sel * widths[np.newaxis, :]

        B_sum = np.sum(weighted_B, axis=1)
        dBdT_sum = np.sum(weighted_dBdT, axis=1)
        planck_num = np.sum(kappa_sel * weighted_B[:, np.newaxis, :], axis=2)
        rosseland_denom = np.sum(weighted_dBdT[:, np.newaxis, :] / safe_kappa, axis=2)

        B_band[:, band] = B_sum
        dBdT_band[:, band] = dBdT_sum
        kappa_planck[:, :, band] = np.divide(
            planck_num,
            B_sum[:, np.newaxis],
            out=np.zeros_like(planck_num),
            where=B_sum[:, np.newaxis] > 0.0,
        )
        kappa_rosseland[:, :, band] = np.divide(
            dBdT_sum[:, np.newaxis],
            rosseland_denom,
            out=np.zeros_like(rosseland_denom),
            where=rosseland_denom > 0.0,
        )

    # Match tausort.c (meanop): tau_i = kappa_ro * p / 2.74e4
    # ODF pressure grid is log10(p), convert to linear pressure.
    pressure_linear = np.power(10.0, odf.P)
    tau_i = kappa_rosseland * pressure_linear[np.newaxis, :, np.newaxis] / 2.74e4
    mix_planck = np.power(2.0, -(tau_i / tau_transition))
    mix_planck = np.clip(mix_planck, 0.0, 1.0)
    kappa_mixed = mix_planck * kappa_planck + (1.0 - mix_planck) * kappa_rosseland

    return {
        "kappa_planck": kappa_planck,
        "kappa_rosseland": kappa_rosseland,
        "kappa_mixed": kappa_mixed,
        "B_band": B_band,
        "dBdT_band": dBdT_band,
        "members_per_band": members_per_band,
    }


def save_tau_bin_opacities_npy(
    output_file: Path,
    tau_bin_results: dict[str, NDArray[np.float64] | NDArray[np.int32]],
    temperature_grid: NDArray[np.float64],
    pressure_grid: NDArray[np.float64],
    group_tau_edges: NDArray[np.float64],
    group_lam_edges: NDArray[np.float64],
    n_splits: int = 1,
    lambda_bin_edges: NDArray[np.float64] | list[float] | None = None,
    tau_edges_per_lambda: list[list[float]] | None = None,
    split_along_lambda: list[bool] | None = None,
) -> None:
    """
    Save tau-binned opacity products to a structured .npy file.

    The band axis (length n_bands) factorizes as ``n_bands = n_groups * n_splits``
    with ``band -> (group = band // n_splits, split = band % n_splits)`` (split 0/1/2 =
    low/mid/high). The **authoritative** grouping is the per-group descriptor — each
    group's tau and lambda window — which covers every mode (uniform, per-cell, and
    the shared-tau + split-flag mode):

        data = np.load("tau_bin_opacities.npy")
        mixed = data["mixed"]                  # [nt, np, n_bands]
        n_splits = int(data["n_splits"])
        group_tau_edges = data["group_tau_edges"]   # [n_groups, 2] -log10(tau) (lo, hi)
        group_lam_edges = data["group_lam_edges"]   # [n_groups, 2] log10(lambda/A) (lo, hi)
        lambda_bin_edges = data["lambda_bin_edges"]
        split_along_lambda = data["split_along_lambda"]  # int8[N], empty unless split-flag mode

    For the per-cell mode the ragged ``n_tau_per_lambda`` / ``tau_edges_concat`` are
    also stored for convenience.
    """
    planck = np.asarray(tau_bin_results["kappa_planck"], dtype=np.float64)
    rosseland = np.asarray(tau_bin_results["kappa_rosseland"], dtype=np.float64)
    mixed = np.asarray(tau_bin_results["kappa_mixed"], dtype=np.float64)
    members = np.asarray(tau_bin_results["members_per_band"], dtype=np.int32)
    temperature = np.asarray(temperature_grid, dtype=np.float64)
    pressure = np.asarray(pressure_grid, dtype=np.float64)

    g_tau = np.asarray(group_tau_edges, dtype=np.float64).reshape(-1, 2)
    g_lam = np.asarray(group_lam_edges, dtype=np.float64).reshape(-1, 2)
    cells = list(tau_edges_per_lambda) if tau_edges_per_lambda is not None else []
    lambda_edges = np.asarray([] if lambda_bin_edges is None else lambda_bin_edges, dtype=np.float64)
    n_tau_per_lambda = np.asarray([len(e) - 1 for e in cells], dtype=np.int32)
    tau_edges_concat = np.asarray([v for e in cells for v in e], dtype=np.float64)
    tau_edges_cell0 = np.asarray(cells[0] if cells else [], dtype=np.float64)
    split_flags = np.asarray([] if split_along_lambda is None else split_along_lambda, dtype=np.int8)

    if not (planck.shape == rosseland.shape == mixed.shape):
        raise ValueError(
            f"Opacity shapes must match; got planck={planck.shape}, rosseland={rosseland.shape}, mixed={mixed.shape}"
        )

    if planck.ndim != 3:
        raise ValueError(f"Expected opacity arrays of shape [nt, np, nbands], got {planck.shape}")

    nt, n_pressure, n_bands = planck.shape
    if members.shape != (n_bands,):
        raise ValueError(f"members_per_band shape must be ({n_bands},), got {members.shape}")
    if temperature.shape != (nt,):
        raise ValueError(f"temperature_grid shape must be ({nt},), got {temperature.shape}")
    if pressure.shape != (n_pressure,):
        raise ValueError(f"pressure_grid shape must be ({n_pressure},), got {pressure.shape}")
    if n_splits <= 0 or n_bands % n_splits != 0:
        raise ValueError(f"n_bands ({n_bands}) must be a positive multiple of n_splits ({n_splits}).")
    n_groups = int(g_tau.shape[0])
    if n_bands != n_splits * n_groups:
        raise ValueError(f"n_bands ({n_bands}) must equal n_splits ({n_splits}) * n_groups ({n_groups}).")

    dtype = np.dtype(
        [
            ("planck", np.float64, (nt, n_pressure, n_bands)),
            ("rosseland", np.float64, (nt, n_pressure, n_bands)),
            ("mixed", np.float64, (nt, n_pressure, n_bands)),
            ("T", np.float64, (nt,)),
            ("p", np.float64, (n_pressure,)),
            ("members_per_band", np.int32, (n_bands,)),
            ("n_splits", np.int32),
            ("group_tau_edges", np.float64, (n_groups, 2)),
            ("group_lam_edges", np.float64, (n_groups, 2)),
            ("tau_bin_edges", np.float64, (tau_edges_cell0.size,)),
            ("lambda_bin_edges", np.float64, (lambda_edges.size,)),
            ("n_tau_per_lambda", np.int32, (n_tau_per_lambda.size,)),
            ("tau_edges_concat", np.float64, (tau_edges_concat.size,)),
            ("split_along_lambda", np.int8, (split_flags.size,)),
        ]
    )

    packed = np.empty((), dtype=dtype)
    packed["planck"] = planck
    packed["rosseland"] = rosseland
    packed["mixed"] = mixed
    packed["T"] = temperature
    packed["p"] = pressure
    packed["members_per_band"] = members
    packed["n_splits"] = np.int32(n_splits)
    packed["group_tau_edges"] = g_tau
    packed["group_lam_edges"] = g_lam
    packed["tau_bin_edges"] = tau_edges_cell0
    packed["lambda_bin_edges"] = lambda_edges
    packed["n_tau_per_lambda"] = n_tau_per_lambda
    packed["tau_edges_concat"] = tau_edges_concat
    packed["split_along_lambda"] = split_flags

    np.save(output_file, packed)
    console.print(f"[green]✓ Saved tau-bin opacities to {output_file}[/green]")


def build_kappa_dat_filename(
    nbands: int,
    n_splits: int,
    lambda_bin_edges: list[float],
    tau_edges_per_lambda: list[list[float]] | None = None,
    tau_bin_edges: list[float] | None = None,
    split_along_lambda: list[bool] | None = None,
    lambda_edges_per_tau: list[list[float]] | None = None,
) -> str:
    """
    Build a .dat filename that encodes the binning parameters.

    - Single lambda cell (backward compatible) spells out the tau edges, e.g.
      ``kappa_24band_tg8_sp3_tau_-0.6347_-0.4_..._7_lam_3_5.dat``.
    - Per-cell multi-lambda: ragged tau edges would make the name unbounded, so
      only lambda edges + per-cell tau-group counts are encoded, e.g.
      ``kappa_30band_lm2_tg3-2_sp3_lam_3_4_5.dat`` (full edges live in the .npy).
    - Split-flag mode (shared tau + per-group flags): the flags are encoded as a
      1/0 string, e.g. ``kappa_30band_lm2_sl11001101_sp3_tau_..._lam_3_3.8_5.dat``.
    - Per-tau-group lambda: shared tau edges + each group's own interior lambda
      cut(s) (``x`` = no split, ``+`` joins multiple cuts in one group), e.g.
      ``kappa_21band_pt_sp3_tau_..._lam_3_5_cuts_3.82-3.65-x-3.8.dat``.
    Edges round to 4 decimals.
    """

    def _fmt(vals: list[float]) -> str:
        return "_".join(f"{round(float(v), 4):g}" for v in vals)

    if lambda_edges_per_tau is not None and tau_bin_edges is not None:
        lmin, lmax = lambda_edges_per_tau[0][0], lambda_edges_per_tau[0][-1]
        cuts = "-".join(
            "x" if len(e) <= 2 else "+".join(f"{round(float(v), 4):g}" for v in e[1:-1]) for e in lambda_edges_per_tau
        )
        return f"kappa_{nbands}band_pt_sp{n_splits}_tau_{_fmt(tau_bin_edges)}_lam_{_fmt([lmin, lmax])}_cuts_{cuts}.dat"

    if split_along_lambda is not None and tau_bin_edges is not None:
        n_lambda = len(lambda_bin_edges) - 1
        sl = "".join("1" if b else "0" for b in split_along_lambda)
        return (
            f"kappa_{nbands}band_lm{n_lambda}_sl{sl}_sp{n_splits}"
            f"_tau_{_fmt(tau_bin_edges)}_lam_{_fmt(lambda_bin_edges)}.dat"
        )

    if tau_edges_per_lambda is None:
        raise ValueError("build_kappa_dat_filename needs tau_edges_per_lambda or (tau_bin_edges + split_along_lambda)")
    n_tau = [len(e) - 1 for e in tau_edges_per_lambda]
    if len(tau_edges_per_lambda) == 1:
        return (
            f"kappa_{nbands}band_tg{n_tau[0]}_sp{n_splits}"
            f"_tau_{_fmt(tau_edges_per_lambda[0])}_lam_{_fmt(lambda_bin_edges)}.dat"
        )
    tg = "-".join(str(g) for g in n_tau)
    return f"kappa_{nbands}band_lm{len(n_tau)}_tg{tg}_sp{n_splits}_lam_{_fmt(lambda_bin_edges)}.dat"


def build_kappa_band_comparison(
    tau_bin_results: dict[str, NDArray[np.float64] | NDArray[np.int32]],
    odf: ODFData,
) -> KappaBandComparison:
    """
    Pack tau-binned opacities into a KappaBandComparison in the C ``tausort``
    convention, ready for write_kappa_4_band_comparison.

    Matches the C ``output()`` writer (tausort.c): the merged opacity and the
    band-integrated Planck term are stored as **natural logs**, the band axis is
    leading (``[Nbands, NT, Np]`` / ``[Nbands, NT]``), and the T/p axes are
    ``log10(T)`` / ``log10(p)`` (= ``odf.T`` / ``odf.P`` as stored). This is the
    same ``ln(mixed)`` that plot_kap_mean_grid overlays against the .npy.

    Empty bands (NaN / non-positive opacity) are written as NaN.
    """
    mixed = np.asarray(tau_bin_results["kappa_mixed"], dtype=np.float64)  # [NT, Np, Nbands]
    b_band = np.asarray(tau_bin_results["B_band"], dtype=np.float64)  # [NT, Nbands]
    nt, n_pressure, n_bands = mixed.shape

    def _safe_log(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.log(np.where(np.isfinite(arr) & (arr > 0.0), arr, np.nan))

    kap_mean = np.ascontiguousarray(_safe_log(mixed).transpose(2, 0, 1))  # [Nbands, NT, Np]
    b_band_log = np.ascontiguousarray(_safe_log(b_band).T)  # [Nbands, NT]

    return KappaBandComparison(
        tau5000bin=0,
        NT=nt,
        Np=n_pressure,
        Nbands_out=n_bands,
        pp_axis=0,
        full_odf=0,
        scatter_on=0,
        back_heating=0,
        tab_T=np.asarray(odf.T, dtype=np.float64),  # log10(T)
        tab_p=np.asarray(odf.P, dtype=np.float64),  # log10(p)
        kap_5000=None,
        B_5000=None,
        kap_mean=kap_mean,
        B_band=b_band_log,
        nuout=None,
    )


@app.command()
def main(
    atm_file: Path = typer.Option(
        "G2_1D.dat",
        "--atm",
        "-a",
        help="1D atmospheric model file (height, density, pressure, temperature)",
    ),
    odf_file: Path = typer.Option("ODF_nc_format.nc", "--odf", "-o", help="ODF data in NetCDF format"),
    continuum_abs: Path = typer.Option("continuumabs.dat", "--cont-abs", help="Continuum absorption opacity file"),
    tau_values: Annotated[
        list[float],
        typer.Option(
            "--tau-values",
            "-t",
            help="List of optical depth values to evaluate atmospheric depths at",
        ),
    ] = [0.1, 1.0],
    tau_bin_edges: Annotated[
        list[float],
        typer.Option(
            "--tau-bin-edges",
            help="List of optical depth bin edges to sort opacities at",
        ),
        # ] = [-.63, -0.3, .75, 1.5, 3.8, 7.0], # original
        # ] = [-0.63, 7.0],  # original
        # ] = [-0.63, -0.3, -0.15, -0.0, 0.25, 0.7, 1.5, 3.9, 7.0], # refined with mid bin
    ] = [-0.63, -0.4, -0.2375, -0.075, 0.15, 0.7, 1.5, 3.8, 7.0],  # refined without mid
    lambda_bin_edges: Annotated[
        list[float],
        typer.Option(
            "--lambda-bin-edges",
            help="List of wavelength edges to sort opacities at",
        ),
    ] = [3.0, 5.0],
    split_lambda: Annotated[
        str | None,
        typer.Option(
            "--split-lambda",
            help="Per-tau-group lambda-split flags, one per tau group in order, as a 0/1 "
            "string (e.g. 00111100) or comma/space-separated true/false. Selects which tau "
            "groups subdivide along lambda. Activates the shared-tau split-flag mode; "
            "mutually exclusive with --optimize-high-overlap. Omit to keep the default "
            "(uniform split when >1 lambda cell).",
        ),
    ] = None,
    lambda_per_tau: Annotated[
        list[str],
        typer.Option(
            "--lambda-per-tau",
            help="Per-tau-group lambda edges: repeat once per tau group (in order), each a "
            "comma-separated increasing edge list, e.g. --lambda-per-tau=3,3.82,5 "
            "--lambda-per-tau=3,5 ... . Each tau group gets its OWN wavelength split (2 edges = "
            "no split); all groups must share the same outer [min,max] window. Activates "
            "per-tau-lambda mode; mutually exclusive with --split-lambda / --optimize-high-overlap.",
        ),
    ] = [],
    skip_first_n_wavelengths: int | None = typer.Option(
        1440,
        "--skip-first-n-wavelengths",
        "-s",
        help="Number of initial wavelength points to skip",
    ),
    tau_bin_output: Path = typer.Option(
        "tau_bin_opacities.npy",
        "--tau-bin-output",
        help="Output .npy file for tau-binned Planck/Rosseland/mixed opacities",
    ),
    optimize_high_overlap: bool = typer.Option(
        False,
        "--optimize-high-overlap",
        help="Run greedy optimizer over tau_bin_edges until every bin's "
        "high-segment overlap clears the threshold, then print and stop.",
    ),
    high_overlap_threshold: float = typer.Option(
        0.70,
        "--high-overlap-threshold",
        help="Minimum acceptable high-segment overlap (0–1) for the optimizer.",
    ),
    max_bins: int = typer.Option(
        8,
        "--max-bins",
        help="Cap on the number of tau groups --optimize-high-overlap may grow to.",
    ),
    save_after_optimize: bool = typer.Option(
        False,
        "--save-after-optimize/--no-save-after-optimize",
        help="After --optimize-high-overlap finishes, continue the normal "
        "pipeline with the optimized edges (saving .npy/.dat) instead of stopping.",
    ),
    refine_mid: bool = typer.Option(
        False,
        "--refine-mid/--no-refine-mid",
        help="Forward to analyze_group: enable/disable iterative b_mid "
        "refinement when segmenting sorted-opacity curves.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Tau-sorting: Bin opacities for stellar atmosphere radiative transfer.

    This tool processes opacity distribution functions (ODFs) and atmospheric
    models to create binned opacities suitable for radiative transfer calculations.
    """

    console.print("\n[bold cyan]═══ Tau-Sorting Opacity Binning Tool ═══[/bold cyan]\n")

    # Step 1: Read atmospheric model
    t0 = time.perf_counter()
    try:
        atm = read_atmospheric_model(atm_file)
    except Exception:
        raise typer.Exit(code=1)
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  read_atmospheric_model: {t1 - t0:.3f}s[/dim]")

    # Step 2: Read ODF data
    t0 = time.perf_counter()
    try:
        # Try .npy format first (much faster), fall back to NetCDF
        npy_file = odf_file.with_suffix(".npy")
        if npy_file.name == "ODF_nc_format.npy":
            npy_file = npy_file.with_name("ODF_format.npy")

        if npy_file.exists():
            odf = read_odf_npy(npy_file)
        else:
            console.print("[yellow]  .npy file not found, using NetCDF (slower)[/yellow]")
            console.print(f"[yellow]  Hint: Run 'python convert_odf_to_npy.py' to create {npy_file}[/yellow]")
            odf = read_odf_netcdf(odf_file)
    except Exception:
        raise typer.Exit(code=1)
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  read_odf: {t1 - t0:.3f}s[/dim]")

    # Step 3: Read continuum data
    t0 = time.perf_counter()
    try:
        cont = read_continuum_data(
            continuum_abs,
            odf.nbins,
            odf.nt,
            odf.np,
        )
    except Exception:
        raise typer.Exit(code=1)
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  read_continuum_data: {t1 - t0:.3f}s[/dim]")

    # Step 4: Verify data consistency
    t0 = time.perf_counter()
    if not verify_data_consistency(atm, odf, cont):
        console.print("\n[red]Data verification failed. Please check your input files.[/red]")
        raise typer.Exit(code=1)
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  verify_data_consistency: {t1 - t0:.3f}s[/dim]")

    console.print("\n[green]✓ All input data loaded and verified successfully![/green]")
    console.print("\n[cyan]Ready to proceed with opacity binning...[/cyan]")

    # using reference opacities calculate Kappa_reference for each T, p point
    # reference_opacities = calculate_reference_opacities(odf, cont, kind="rosseland")

    # reference_opacities.shape = (nt, np)

    # Interpolate reference opacities onto atmospheric model grid (T, p)
    t0 = time.perf_counter()
    interpolated_opacity = interpolate_kappa_to_atmosphere(odf, cont, atm)
    t1 = time.perf_counter()
    console.print(f"interpolated_opacity shape: {interpolated_opacity.shape}")
    console.print(f"[dim]⏱  interpolate_kappa_to_atmosphere: {t1 - t0:.3f}s[/dim]")

    console.print("Calculate kappa rosseland at each atmosphere T, p point...")

    t0 = time.perf_counter()
    kappa_on_atmosphere_tp, wavelength_grid_subbins_centers = calculate_reference_opacities_from_custom_tp_grid(
        atm,
        interpolated_opacity,
        odf.wavelength_grid,  # convert to cm
        odf.subbin,
        odf.nbins,
        odf.nsubbins,
        kind="rosseland",
    )
    t1 = time.perf_counter()
    console.print(f"kappa_on_atmosphere_tp shape: {kappa_on_atmosphere_tp.shape}")
    console.print(f"[dim]⏱  calculate_reference_opacities_from_custom_tp_grid: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    tau_rosseland = compute_tau_rosseland(atm, kappa_on_atmosphere_tp)
    t1 = time.perf_counter()
    console.print(f"tau_rosseland shape: {tau_rosseland.shape}")
    console.print(f"[dim]⏱  compute_tau_rosseland: {t1 - t0:.3f}s[/dim]")

    console.print("Plotting Rosseland optical depth profile...")

    t0 = time.perf_counter()
    plot_rosseland_tau(atm, tau_rosseland)
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  plot_rosseland_tau: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    height_at_tau_index, height_at_tau = get_depth_at_tau_values_from_full_opacity(
        atm,
        interpolated_opacity,
        wavelength_grid_subbins_centers,
        tau_values=tau_values,
    )
    console.print(f"height_at_tau shape: {height_at_tau.shape}")
    select_tau_index = 1
    max_height_idx = np.max(height_at_tau_index[:, select_tau_index])
    console.print(
        f"first and last 10 of height_at_tau_index: {height_at_tau_index[:10, select_tau_index]} ... {height_at_tau_index[-10:, select_tau_index]}"
    )
    console.print(
        f"first and last 10 of height_at_tau: {height_at_tau[:10, select_tau_index]} ... {height_at_tau[-10:, select_tau_index]}"
    )
    console.print(f"max_height_idx is: {max_height_idx}")

    t1 = time.perf_counter()
    console.print(f"[dim]⏱  get_depth_at_tau_values_from_full_opacity: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    plot_height_at_tau_values(
        wavelength_grid_subbins_centers[skip_first_n_wavelengths:],
        height_at_tau[skip_first_n_wavelengths:, :],
        tau_values=tau_values,
        output_file="height_at_tau_values.pdf",
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  plot_height_at_tau_values: {t1 - t0:.3f}s[/dim]")

    tau_rosseland_at_tau_lambda_one = tau_rosseland[height_at_tau_index[:, -1]]

    n_lambda = len(lambda_bin_edges) - 1
    n_splits = 3
    # Atmosphere-top clamp for the first (smallest -log10 tau) edge. Applied to the
    # descriptor used for sort/plots/save AFTER the membership assign (which uses the
    # un-clamped edges), reproducing the original in-sort clamp.
    top_edge = -np.log10(tau_rosseland[max_height_idx] + 0.2)

    tau_edges_per_lambda: list[list[float]] | None = None  # per-cell / uniform modes
    flag_tau_bin_edges: list[float] | None = None  # flag mode / per-tau mode (clamped shared edges)
    split_flags: list[bool] | None = None
    lambda_edges_per_tau: list[list[float]] | None = None  # per-tau-group lambda mode

    if lambda_per_tau:
        # ---- Per-tau-group lambda mode: each tau group has its OWN lambda edges ----
        if optimize_high_overlap or split_lambda is not None:
            raise typer.BadParameter(
                "--lambda-per-tau is mutually exclusive with --split-lambda and --optimize-high-overlap."
            )
        n_tau = len(tau_bin_edges) - 1
        lambda_edges_per_tau = parse_lambda_per_tau(lambda_per_tau)
        if len(lambda_edges_per_tau) != n_tau:
            raise typer.BadParameter(
                f"--lambda-per-tau has {len(lambda_edges_per_tau)} entries, expected one per tau group ({n_tau})."
            )
        if len({e[0] for e in lambda_edges_per_tau}) != 1 or len({e[-1] for e in lambda_edges_per_tau}) != 1:
            raise typer.BadParameter("all --lambda-per-tau groups must share the same outer [min, max] lambda window.")
        lambda_bin_edges = [lambda_edges_per_tau[0][0], lambda_edges_per_tau[0][-1]]  # outer window for plot/save
        n_lambda = 1
        # Membership from un-clamped edges; descriptor from the atmosphere-top-clamped tau.
        _gt0, _gl0, offsets = build_group_specs_per_tau(tau_bin_edges, lambda_edges_per_tau)
        bin_number = assign_per_tau_lambda(
            tau_rosseland_at_tau_lambda_one,
            wavelength_grid_subbins_centers,
            tau_bin_edges,
            lambda_edges_per_tau,
            offsets,
        )
        flag_tau_bin_edges = list(tau_bin_edges)
        flag_tau_bin_edges[0] = top_edge
        group_tau_edges, group_lam_edges, _offc = build_group_specs_per_tau(flag_tau_bin_edges, lambda_edges_per_tau)
        n_split_groups = sum(1 for e in lambda_edges_per_tau if len(e) > 2)
        console.print(
            f"[green]per-tau-lambda mode: {n_split_groups}/{n_tau} tau-groups split, each with its own λ cut[/green]"
        )
    elif split_lambda is not None:
        # ---- Flag mode: shared tau binning + per-tau-group lambda-split flags ----
        if optimize_high_overlap:
            raise typer.BadParameter("--split-lambda and --optimize-high-overlap are mutually exclusive.")
        n_tau = len(tau_bin_edges) - 1
        split_flags = parse_split_lambda(split_lambda)
        if len(split_flags) != n_tau:
            raise typer.BadParameter(
                f"--split-lambda has {len(split_flags)} entries, expected one per tau group ({n_tau})."
            )
        if n_lambda == 1:
            console.print("[yellow]--split-lambda given but only one lambda cell; flags are a no-op.[/yellow]")
        # Membership from un-clamped edges.
        _gt0, _gl0, s2cg, s2sg = build_group_specs_split_lambda(tau_bin_edges, lambda_bin_edges, split_flags)
        bin_number = assign_split_lambda(
            tau_rosseland_at_tau_lambda_one,
            wavelength_grid_subbins_centers,
            tau_bin_edges,
            lambda_bin_edges,
            s2cg,
            s2sg,
        )
        # Clamped descriptor for sort/plots/save.
        flag_tau_bin_edges = list(tau_bin_edges)
        flag_tau_bin_edges[0] = top_edge
        group_tau_edges, group_lam_edges, _s2cg, _s2sg = build_group_specs_split_lambda(
            flag_tau_bin_edges, lambda_bin_edges, split_flags
        )
        console.print(
            f"[green]split-lambda mode: {sum(split_flags)}/{n_tau} tau-groups split into {n_lambda} λ cells[/green]"
        )
    else:
        if optimize_high_overlap:
            t0 = time.perf_counter()
            tau_edges_per_lambda = optimize_tau_bin_edges(
                atm=atm,
                odf=odf,
                interpolated_opacity=interpolated_opacity,
                tau_rosseland=tau_rosseland,
                tau_rosseland_at_tau_lambda_one=tau_rosseland_at_tau_lambda_one,
                wavelength_grid_subbins_centers=wavelength_grid_subbins_centers,
                max_height_idx=max_height_idx,
                initial_tau_bin_edges=tau_bin_edges,
                lambda_bin_edges=lambda_bin_edges,
                threshold=high_overlap_threshold,
                max_bins=max_bins,
                refine_mid=refine_mid,
            )
            t1 = time.perf_counter()
            console.print(f"[dim]⏱  optimize_tau_bin_edges: {t1 - t0:.3f}s[/dim]")
            for cell, edges in enumerate(tau_edges_per_lambda):
                console.print(
                    f"[green]λ cell {cell}: {len(edges) - 1} tau-groups, edges = {[round(e, 4) for e in edges]}[/green]"
                )
            if not save_after_optimize:
                # Print-only: diagnostic overlap table for the optimized edges, then stop.
                diag_bin_number = assign_tau_to_bin(
                    tau_rosseland_at_tau_lambda_one,
                    wavelength_grid_subbins_centers,
                    tau_edges_per_lambda=tau_edges_per_lambda,
                    lambda_bin_edges=lambda_bin_edges,
                )
                diag_edges = [list(e) for e in tau_edges_per_lambda]
                for ce in diag_edges:
                    ce[0] = top_edge
                diag_gt, diag_gl = build_group_specs_per_cell(diag_edges, lambda_bin_edges)
                diag_sorted = sort_weighted_opacity_per_tau_bin(
                    atm=atm,
                    odf=odf,
                    interpolated_opacity=interpolated_opacity,
                    tau_rosseland=tau_rosseland,
                    band_index=diag_bin_number,
                    group_tau_edges=diag_gt,
                    wavelength_grid_subbins_centers=wavelength_grid_subbins_centers,
                    write_debug_json=False,
                    verbose=False,
                )
                compute_bot_segment_overlap_per_tau_bin(
                    diag_sorted,
                    group_tau_edges=diag_gt,
                    group_lam_edges=diag_gl,
                    print_table=True,
                    verbose=True,
                    refine_mid=refine_mid,
                )
                return
            console.print("[cyan]--save-after-optimize: continuing pipeline with optimized per-cell edges[/cyan]")
        else:
            # No optimization: the same tau edges apply in every lambda cell.
            tau_edges_per_lambda = [list(tau_bin_edges) for _ in range(n_lambda)]

        # Membership from un-clamped per-cell edges, then clamp + build the descriptor.
        bin_number = assign_tau_to_bin(
            tau_rosseland_at_tau_lambda_one,
            wavelength_grid_subbins_centers,
            tau_edges_per_lambda=tau_edges_per_lambda,
            lambda_bin_edges=lambda_bin_edges,
        )
        for ce in tau_edges_per_lambda:
            ce[0] = top_edge
        group_tau_edges, group_lam_edges = build_group_specs_per_cell(tau_edges_per_lambda, lambda_bin_edges)

    console.print("\n[cyan]Calculating tau-binned opacities...[/cyan]")

    n_groups = int(group_tau_edges.shape[0])
    n_bands = n_groups * n_splits

    unique_bins = np.unique(bin_number[bin_number >= 0])
    unassigned = int(np.sum(bin_number < 0))
    console.print(f"assigned groups: {unique_bins}")
    console.print(f"unassigned wavelength points: {unassigned}/{len(bin_number)}")

    plot_tau_rosselend_at_tau_lambda_one_vs_wavelength(
        tau_rosseland_at_tau_lambda_one[skip_first_n_wavelengths:],
        wavelength_grid_subbins_centers[skip_first_n_wavelengths:],
        group_tau_edges=group_tau_edges,
        group_lam_edges=group_lam_edges,
        lambda_bin_edges=lambda_bin_edges,
        band_index=bin_number[skip_first_n_wavelengths:],
    )

    t0 = time.perf_counter()
    sorted_per_bin = sort_weighted_opacity_per_tau_bin(
        atm=atm,
        odf=odf,
        interpolated_opacity=interpolated_opacity,
        tau_rosseland=tau_rosseland,
        band_index=bin_number,
        group_tau_edges=group_tau_edges,
        wavelength_grid_subbins_centers=wavelength_grid_subbins_centers,
    )
    t1 = time.perf_counter()
    n_nonempty = sum(1 for v in sorted_per_bin.values() if not v.get("empty", False))
    console.print(f"sorted opacity dist: {n_nonempty}/{len(sorted_per_bin)} non-empty groups")
    for g, r in sorted_per_bin.items():
        if r.get("empty", False):
            console.print(f"  group {g}: empty")
        else:
            console.print(
                f"  group {g} (λ[{group_lam_edges[g, 0]:.2f},{group_lam_edges[g, 1]:.2f}], "
                f"τ[{group_tau_edges[g, 0]:.2f},{group_tau_edges[g, 1]:.2f}]): members={r['members']}, "
                f"T_top={r['T_top']:.1f}K (i={r['i_top']}), "
                f"T_bot={r['T_bot']:.1f}K (i={r['i_bot']})"
            )
    console.print(f"[dim]⏱  sort_weighted_opacity_per_tau_bin: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    plot_sorted_weighted_opacity_per_tau_bin(
        sorted_per_bin,
        group_tau_edges=group_tau_edges,
        group_lam_edges=group_lam_edges,
        refine_mid=refine_mid,
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  plot_sorted_weighted_opacity_per_tau_bin: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    compute_bot_segment_overlap_per_tau_bin(
        sorted_per_bin,
        group_tau_edges=group_tau_edges,
        group_lam_edges=group_lam_edges,
        refine_mid=refine_mid,
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  compute_bot_segment_overlap_per_tau_bin: {t1 - t0:.3f}s[/dim]")

    # Subdivide each (lambda, tau) group into n_splits opacity segments (low/mid/high) and
    # build a finer band index so the saved table resolves (lambda cell, tau group, split).
    # The table is always 3 splits (analyze_group with refine_mid=False); the --refine-mid
    # flag only affects the diagnostic overlap table / plot above. The band axis factorizes
    # as band -> (group = band // n_splits, split = band % n_splits).
    t0 = time.perf_counter()
    split_band_index = build_split_band_index(
        sorted_per_bin,
        n_subbin_points=len(bin_number),
        n_groups=n_groups,
        n_splits=n_splits,
    )
    t1 = time.perf_counter()
    n_assigned = int(np.sum(split_band_index >= 0))
    console.print(
        f"split band index: {n_assigned}/{len(split_band_index)} sub-bins assigned across "
        f"{n_bands} bands ({n_groups} (λ,τ) groups x {n_splits} splits)"
    )
    console.print(f"[dim]⏱  build_split_band_index: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    tau_bin_results = calculate_tau_bin_opacities(
        odf=odf,
        cont=cont,
        band_index=split_band_index,
        n_bins=n_bands,
        tau_transition=0.35,
    )
    t1 = time.perf_counter()
    console.print(
        f"tau-binned opacity shapes: "
        f"planck={tau_bin_results['kappa_planck'].shape}, "
        f"rosseland={tau_bin_results['kappa_rosseland'].shape}, "
        f"mixed={tau_bin_results['kappa_mixed'].shape}"
    )
    console.print(f"[dim]⏱  calculate_tau_bin_opacities: {t1 - t0:.3f}s[/dim]")

    # Mark empty (group, split) bands as NaN so they are distinguishable from genuine
    # zero opacity (calculate_tau_bin_opacities leaves empty bands at 0.0).
    members_per_band = np.asarray(tau_bin_results["members_per_band"])
    empty_band_mask = members_per_band == 0
    if np.any(empty_band_mask):
        for key in ("kappa_planck", "kappa_rosseland", "kappa_mixed"):
            tau_bin_results[key][:, :, empty_band_mask] = np.nan
        console.print(f"[yellow]{int(np.sum(empty_band_mask))}/{n_bands} bands empty -> set to NaN[/yellow]")

    t0 = time.perf_counter()
    save_tau_bin_opacities_npy(
        tau_bin_output,
        tau_bin_results,
        temperature_grid=np.power(10.0, odf.T),
        pressure_grid=np.power(10.0, odf.P),
        group_tau_edges=group_tau_edges,
        group_lam_edges=group_lam_edges,
        n_splits=n_splits,
        lambda_bin_edges=lambda_bin_edges,
        tau_edges_per_lambda=tau_edges_per_lambda,
        split_along_lambda=split_flags,
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  save_tau_bin_opacities_npy: {t1 - t0:.3f}s[/dim]")

    # Also save the C-format kappa_<N>_band comparison binary, under a filename that
    # encodes the binning parameters so each run lands in a distinct, self-describing
    # file. kap_mean = ln(mixed) in [Nbands, NT, Np], matching tausort.c output().
    t0 = time.perf_counter()
    kappa_dat_path = build_kappa_dat_filename(
        nbands=n_bands,
        n_splits=n_splits,
        lambda_bin_edges=lambda_bin_edges,
        tau_edges_per_lambda=tau_edges_per_lambda,
        tau_bin_edges=flag_tau_bin_edges,
        split_along_lambda=split_flags,
        lambda_edges_per_tau=lambda_edges_per_tau,
    )
    write_kappa_4_band_comparison(kappa_dat_path, build_kappa_band_comparison(tau_bin_results, odf))
    t1 = time.perf_counter()
    console.print(f"[green]✓ Saved kappa band comparison to {kappa_dat_path}[/green]")
    console.print(f"[dim]⏱  write_kappa_4_band_comparison: {t1 - t0:.3f}s[/dim]")


@app.command()
def verify_planck(
    output: str = typer.Option(
        "planck_verification.pdf",
        "--output",
        "-o",
        help="Output filename for verification plot",
    ),
):
    """
    Verify Planck function and derivative calculations.

    Creates plots of the Planck function and its temperature derivative
    at various temperatures for verification purposes.
    """
    plot_planck_and_derivatives(output)


@app.command("convert-continuum")
def convert_continuum(
    abs_file: Path = typer.Argument(..., help="Input ASCII continuum .dat to convert."),
    output: str = typer.Argument("", help="Output .npy path (default: the input name with .npy)."),
    nt: int = typer.Option(300, "--nt", help="Number of temperature points."),
    n_pressure: int = typer.Option(150, "--np", help="Number of pressure points."),
    nbins: int = typer.Option(328, "--nbins", help="Number of wavelength bins."),
):
    """
    Convert an ASCII continuum .dat to the fast .npy cache that `main` reads.

    Usage: `convert-continuum INPUT.dat [OUTPUT.npy]`. `main` reads `continuumabs.npy` if present
    (a ~75x speedup over the ASCII .dat) but never writes it; this produces it. The .dat is one
    value per line in (lambda, T, P) order (nbins*nt*n_pressure = 328*300*150 by default); it is
    reshaped and transposed to the (nt, n_pressure, nbins) layout `main` expects. Pass
    --nt/--np/--nbins for other grids.
    """
    if not abs_file.exists():
        raise typer.BadParameter(f"{abs_file} not found")
    console.print(f"[cyan]Reading ASCII continuum from {abs_file} (slow for large files)...[/cyan]")
    data = np.loadtxt(str(abs_file))
    expected = nt * n_pressure * nbins
    if data.size != expected:
        raise typer.BadParameter(
            f"{abs_file}: got {data.size} values, expected {expected} = nt*np*nbins "
            f"({nt}*{n_pressure}*{nbins}); pass --nt/--np/--nbins to match your file."
        )
    # .dat is (lambda, T, P) C-order -> (nbins, nt, np); transpose to (nt, np, nbins).
    kappa = data.reshape((nbins, nt, n_pressure)).transpose(1, 2, 0)
    out = Path(output) if output else Path(str(abs_file).replace(".dat", ".npy"))
    np.save(out, kappa)
    console.print(f"[green]✓ {abs_file} -> {out}  shape={kappa.shape}  ({kappa.min():.2e} … {kappa.max():.2e})[/green]")


@app.command("convert-odf")
def convert_odf(
    input_file: Path = typer.Argument(..., help="Input ODF NetCDF (.nc) file to convert."),
    output: str = typer.Argument("", help="Output .npy path (default: ODF_format.npy, what `main` reads)."),
):
    """
    Convert an ODF NetCDF (.nc) file to the .npy structured array that `main` loads.

    Usage: `convert-odf INPUT.nc [OUTPUT.npy]`. `main` reads `ODF_format.npy` if present (much
    faster than parsing the NetCDF) and hints at this converter when it is missing; the default
    output is `ODF_format.npy` (not a suffix swap of the input). Thin wrapper over
    `convert_odf_to_npy.convert_odf_netcdf_to_npy`.
    """
    from convert_odf_to_npy import convert_odf_netcdf_to_npy

    if not input_file.exists():
        raise typer.BadParameter(f"{input_file} not found")
    out = Path(output) if output else Path("ODF_format.npy")
    convert_odf_netcdf_to_npy(input_file, out)
    console.print(f"[green]✓ {input_file} -> {out}[/green]")


@app.command("convert-model")
def convert_model(
    input_file: Path = typer.Argument(..., help="Input binary STAGGER model atmosphere, e.g. models/G_SSD."),
    output: str = typer.Argument("", help="Output ASCII .dat (default: the input name with .dat appended)."),
):
    """
    Convert a binary STAGGER model atmosphere to G2_1D.dat's ASCII 4-column format.

    Usage: `convert-model INPUT [OUTPUT.dat]`. Reads the float32 STAGGER dump (a 4-float header
    then density/·/pressure/temperature rows; z is the uniform 1e6 cm grid), and writes columns
    `z[cm] density[g/cm^3] pressure[dyn/cm^2] temperature[K]` ordered top-of-atmosphere first
    (z descending, cool→hot) to match G2_1D.dat, so the result is a drop-in `--atm` input.
    """
    if not input_file.exists():
        raise typer.BadParameter(f"{input_file} not found")
    dz = 1.0e6  # STAGGER grid spacing [cm]
    tvar = np.fromfile(str(input_file), dtype=np.float32)
    tvar = tvar[4:].reshape([int(tvar[0]), int(tvar[1])])  # skip 4-float header -> (nvars, nz)
    nz = tvar.shape[-1]
    # rows 0/2/3 are density/pressure/temperature; flip to cool(top)->hot(bottom), and give the
    # cool top the largest height so z runs 4.99e8 -> 0 like G2_1D.dat.
    z = np.flip(np.arange(nz) * dz)
    rho, pre, tem = np.flip(tvar[0]), np.flip(tvar[2]), np.flip(tvar[3])
    out = Path(output) if output else Path(str(input_file) + ".dat")
    np.savetxt(out, np.column_stack([z, rho, pre, tem]), fmt="%e")
    console.print(f"[green]✓ {input_file} -> {out}  ({nz} levels; T {tem.min():.0f}..{tem.max():.0f} K)[/green]")


if __name__ == "__main__":
    app()
