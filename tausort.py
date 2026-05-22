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
            console.print("  [yellow]Loading ASCII (slow) - consider converting to .npy[/yellow]")
            data = np.loadtxt(str(filepath))
            expected_size = n_bins * n_temperature * n_pressure

            if data.size != expected_size:
                console.print(f"[yellow]Warning: Expected {expected_size} values, got {data.size}[/yellow]")

            # Reshape to (nt, n_pressure, nb)
            kappa = data.reshape((n_temperature, n_pressure, n_bins))

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
    tau_bin_edges: list[int | float],
    lambda_bin_edges: list[int | float],
    output_file: str = "tau_rosseland_at_tau_lambda_one.jpg",
    use_2d_histogram: bool = True,
    band_index: NDArray[np.int32] | None = None,
) -> None:
    """
    Plot the Rosseland optical depth at the height where the optical depth
    at each wavelength subbin is equal to one, as a function of wavelength.

    Args:
        tau_rosseland: Rosseland optical depth profile [n_bins*n_subbins]
        wavelength_grid: Wavelength grid [Angstrom]
        tau_bin_edges: Optical depth bin edges to plot
        lambda_bin_edges: Wavelength bin edges to plot
        output_file: Output filename for the plot
        use_2d_histogram: If True, plot as 2D histogram instead of scatter plot
        band_index: Optional tau-bin assignment per sub-bin (same length as
            tau_rosseland). If provided, the topmost (max -log10(τ)) and
            lowermost (min -log10(τ)) sub-bin per tau-bin are circled in red.
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

    for _, edge in enumerate(tau_bin_edges):
        ax.axhline(edge, c="k")
    for _, edge in enumerate(lambda_bin_edges):
        ax.axvline(edge, c="k")

    if band_index is not None:
        if band_index.shape[0] != tau_rosseland.shape[0]:
            raise ValueError(
                f"band_index length ({band_index.shape[0]}) must match tau_rosseland length ({tau_rosseland.shape[0]})"
            )
        n_bins = len(tau_bin_edges) - 1
        sel_x: list[float] = []
        sel_y: list[float] = []
        for k in range(n_bins):
            mask = band_index == k
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
            label="topmost / lowermost per tau-bin",
        )
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ τ_Rosseland at τ_λ=1 plot saved to {output_file}[/green]")


def assign_tau_to_bin(
    tau_rosseland: NDArray[np.float64],
    wavelength_grid_input: NDArray[np.float64],
    tau_bin_edges: list[int | float],
    lambda_bin_edges: list[int | float],
) -> NDArray[np.int32]:
    """
    Assign sub-bins to tau-wavelength bins.

    Args:
        tau_rosseland: Tau values at sub-bin wavelengths [n_bins*n_subbins]
        wavelength_grid_subbins_center: Wavelength at sub-bin centers. [n_bins*n_subbins]
        tau_bin_edges: Tau edges.
        lambda_bin_edges: Wavelength edges.

    Returns:
        Tau-bin index per sub-bin wavelength point.
        Values are in [0, n_tau_bins-1], or -1 if out of configured ranges.
    """
    wavelength_grid = wavelength_grid_input * 1e8  # Angstrom
    x_data = np.log10(wavelength_grid)
    # Prevent log10(0) when tau is numerically tiny.
    y_data = -np.log10(np.clip(tau_rosseland, 1.0e-300, None))

    tau_edges = np.asarray(tau_bin_edges, dtype=np.float64)
    lambda_edges = np.asarray(lambda_bin_edges, dtype=np.float64)

    n_lambda_bins = len(lambda_edges) - 1
    n_tau_bins = len(tau_edges) - 1

    lambda_idx = np.digitize(x_data, lambda_edges, right=False) - 1
    tau_idx = np.digitize(y_data, tau_edges, right=False) - 1

    valid = (lambda_idx >= 0) & (lambda_idx < n_lambda_bins) & (tau_idx >= 0) & (tau_idx < n_tau_bins)

    band_index = np.full(x_data.shape, -1, dtype=np.int32)
    # We keep only tau-bin identity for band means (n_bins = n_tau_bins).
    band_index[valid] = tau_idx[valid].astype(np.int32)

    return band_index


def sort_weighted_opacity_per_tau_bin(
    atm: AtmosphericData,
    odf: ODFData,
    interpolated_opacity: NDArray[np.float64],
    tau_rosseland: NDArray[np.float64],
    band_index: NDArray[np.int32],
    tau_bin_edges: list[float],
    wavelength_grid_subbins_centers: NDArray[np.float64],
    max_height_idx: int,
    write_debug_json: bool = True,
    verbose: bool = True,
) -> dict[int, dict[str, NDArray[np.float64] | NDArray[np.int64] | float | int | bool]]:
    r"""
    For each tau-bin, locate the two atmospheric layers (top and bottom)
    that bracket the bin's Rosseland-tau range, then build a sorted
    distribution of weighted opacities at each (T, p) point.

    Per tau-bin k:
      1. tau_low = 10^(-tau_bin_edges[k+1]), tau_high = 10^(-tau_bin_edges[k])
         (matches digitize(right=False) in assign_tau_to_bin).
      2. j_top, j_bot = searchsorted(tau_full, tau_low/tau_high) on the
         reversed-z tau profile;
      3. For sub-bins with band_index == k, read opacities at
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
        band_index: Tau-bin assignment per sub-bin, shape [nbins*nsubbins],
            values in [0, n_bins-1] or -1 for unassigned.
        tau_bin_edges: -log10(τ_Ros) bin edges (increasing).
        wavelength_grid_subbins_centers: Sub-bin center wavelengths [cm],
            shape [nbins*nsubbins].

    Returns:
        Dict keyed by tau-bin index. Empty bins → {"empty": True, "members": 0}.
        Non-empty bins contain T_top, p_top, i_top, T_bot, p_bot, i_bot,
        member_indices, kappa_top, kappa_bot, weights_top, weights_bot,
        weighted_kappa_top, weighted_kappa_bot, sort_idx_top, sort_idx_bot,
        sorted_weighted_kappa_top, sorted_weighted_kappa_bot.
    """
    if verbose:
        console.print("\n[cyan]Sorting weighted opacities per tau-bin at top/bot TP points...[/cyan]")

    n_layers = atm.nlevels
    n_bins = len(tau_bin_edges) - 1

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

    tau_bin_edges[0] = -np.log10(tau_rosseland[max_height_idx] + 0.2)

    for k in range(n_bins):
        tau_low = 10.0 ** (-tau_bin_edges[k + 1])
        tau_high = 10.0 ** (-tau_bin_edges[k])

        if verbose:
            console.print(
                f"\nProcessing tau-bin {k}: -log10(tau) in [{tau_bin_edges[k]:.2f}, {tau_bin_edges[k + 1]:.2f}] "
                f"→ tau in [{tau_low:.2e}, {tau_high:.2e}]"
            )

        j_top = int(np.clip(np.searchsorted(tau_full, tau_low, side="left"), 0, n_layers - 1))
        j_bot = int(np.clip(np.searchsorted(tau_full, tau_high, side="left"), 0, n_layers - 1))

        # i_top = n_layers - 1 - j_top
        # i_bot = n_layers - 1 - j_bot
        i_top = j_top
        i_bot = j_bot

        member_idx = np.flatnonzero(band_index == k)
        if member_idx.size == 0:
            results[k] = {"empty": True, "members": 0}
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

        results[k] = {
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
            debug_path = f"debug_tau_bin_{k}.json"
            serializable = {
                key: (val.tolist() if isinstance(val, np.ndarray) else val) for key, val in results[k].items()
            }
            with open(debug_path, "w") as f:
                json.dump(serializable, f, indent=2)

    return results


def plot_sorted_weighted_opacity_per_tau_bin(
    sorted_per_bin: dict[int, dict],
    tau_bin_edges: list[float],
    output_file: str = "sorted_weighted_opacity_per_tau_bin.jpg",
    smooth_window: int = 7,
    overlay_breaks: bool = True,
    refine_mid: bool = True,
) -> None:
    """
    Plot sorted weighted opacities per tau-bin.

    For each tau-bin, draw a subplot with both 'top' (smaller-τ edge of the
    bin) and 'bot' (larger-τ edge) sorted κ·Δλ·B_λ curves on a log y-axis.
    If overlay_breaks is True, run analyze_group on each curve and overlay
    vertical lines at the segmentation break points (b1, b2, optional b_mid).

    Args:
        sorted_per_bin: Output of sort_weighted_opacity_per_tau_bin.
        tau_bin_edges: -log10(τ_Ros) bin edges.
        output_file: Output figure path.
        smooth_window: Smoothing window passed to analyze_group.
        overlay_breaks: If True, draw piecewise-linear break lines.
        refine_mid: Forwarded to analyze_group (skip iterative b_mid refinement
            when False).
    """
    console.print("\n[cyan]Plotting sorted weighted opacities per tau-bin...[/cyan]")

    n_bins = len(tau_bin_edges) - 1
    ncols = int(np.ceil(np.sqrt(n_bins)))
    nrows = int(np.ceil(n_bins / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for k in range(n_bins):
        ax = axes_flat[k]
        r = sorted_per_bin.get(k, {})
        edge_low = tau_bin_edges[k]
        edge_high = tau_bin_edges[k + 1]
        title = (
            f"tau-bin {k}: "
            rf"$-\log_{{10}}\tau_{{\rm Ros}} \in [{edge_low:.2f}, {edge_high:.2f}]$"
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
                console.print(f"[yellow]bin {k} bot: analyze_group failed: {e}[/yellow]")
            else:
                seg = res["seg"]
                ax.axvline(seg["b1"], color="C1", ls=":", lw=1.0, alpha=0.9)
                ax.axvline(seg["b2"], color="C1", ls=":", lw=1.0, alpha=0.9)
                if seg.get("split_mid", False):
                    ax.axvline(seg["b_mid"], color="C1", ls="--", lw=1.0, alpha=0.7)

        ax.legend(fontsize=8, loc="lower right")

    for k in range(n_bins, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    console.print(f"[green]✓ Sorted weighted opacity plot saved to {output_file}[/green]")


def compute_bot_segment_overlap_per_tau_bin(
    sorted_per_bin: dict[int, dict],
    tau_bin_edges: list[float],
    smooth_window: int = 7,
    print_table: bool = True,
    verbose: bool = True,
    refine_mid: bool = True,
) -> dict[int, dict]:
    """
    Per tau-bin: run analyze_group on s_bot only, then reuse those break
    positions (b1, b2, optional b_mid) as segment boundaries in *sorted-index*
    space for both s_top and s_bot. For each segment, build the set of
    original sub-bin indices via sort_idx_top / sort_idx_bot and compute the
    overlap between the top and bot sets. By construction both sets have the
    same size (= segment length).
    """
    n_bins = len(tau_bin_edges) - 1
    overlaps: dict[int, dict] = {}

    table = Table(
        title="Segment overlap (bot-derived breaks applied to top) per tau-bin",
        show_lines=False,
    )
    table.add_column("bin", justify="right")
    table.add_column("τ range", justify="center")
    table.add_column("segment", justify="left")
    table.add_column("|A|=|B|", justify="right")
    table.add_column("|A∩B|", justify="right")
    table.add_column("overlap %", justify="right")

    for k in range(n_bins):
        r = sorted_per_bin.get(k, {})
        edge_low = tau_bin_edges[k]
        edge_high = tau_bin_edges[k + 1]
        tau_label = f"[{edge_low:.2f}, {edge_high:.2f}]"

        if r.get("empty", False) or "sorted_weighted_kappa_bot" not in r:
            table.add_row(str(k), tau_label, "—", "—", "—", "—")
            overlaps[k] = {"empty": True}
            continue

        s_bot = np.asarray(r["sorted_weighted_kappa_bot"], dtype=np.float64)
        member_indices = np.asarray(r["member_indices"])
        sort_idx_top = np.asarray(r["sort_idx_top"])
        sort_idx_bot = np.asarray(r["sort_idx_bot"])
        n = s_bot.size

        if n <= 10:
            table.add_row(str(k), tau_label, "—", str(n), "—", "—")
            overlaps[k] = {"too_small": True, "n": int(n)}
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
                console.print(f"[yellow]bin {k}: analyze_group failed: {e}[/yellow]")
            table.add_row(str(k), tau_label, "—", str(n), "—", "—")
            overlaps[k] = {"failed": True, "error": str(e)}
            continue

        seg = res["seg"]
        b1 = int(seg["b1"])
        b2 = int(seg["b2"])
        split_mid = bool(seg.get("split_mid", False))
        b_mid = int(seg["b_mid"]) if split_mid else None

        if split_mid:
            seg_defs = [
                ("low", 0, b1 + 1),
                ("mid1", b1 + 1, b_mid + 1),
                ("mid2", b_mid + 1, b2),
                ("high", b2, n),
            ]
        else:
            seg_defs = [
                ("low", 0, b1 + 1),
                ("mid", b1 + 1, b2),
                ("high", b2, n),
            ]

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
                str(k) if i == 0 else "",
                tau_label if i == 0 else "",
                name,
                str(size),
                str(n_inter),
                f"{pct:.1f}",
            )

        overlaps[k] = {
            "b1": b1,
            "b2": b2,
            "b_mid": b_mid,
            "split_mid": split_mid,
            "segments": bin_overlaps,
        }

    if print_table:
        console.print(table)
    return overlaps


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
) -> tuple[list[float], dict[int, dict], dict[int, dict]]:
    """
    Greedy adjust+insert search over tau_bin_edges to push the high-segment
    overlap (from compute_bot_segment_overlap_per_tau_bin) above `threshold`
    in every non-empty tau-bin. First nudges interior edges on a small step
    grid; when no local nudge improves the minimum score, inserts a new edge
    at the midpoint of the worst bin's tau range. Stops when all bins clear
    the threshold, when no move improves and bin count is at `max_bins`, or
    when `max_outer_iters` is reached.

    Returns (final_edges, last_sorted_per_bin, last_overlaps).
    """
    threshold_pct = threshold * 100.0

    def _evaluate(edges: list[float]) -> tuple[dict, dict]:
        edges_copy = list(edges)
        band_idx = assign_tau_to_bin(
            tau_rosseland_at_tau_lambda_one,
            wavelength_grid_subbins_centers,
            tau_bin_edges=edges_copy,
            lambda_bin_edges=lambda_bin_edges,
        )
        sorted_per_bin = sort_weighted_opacity_per_tau_bin(
            atm=atm,
            odf=odf,
            interpolated_opacity=interpolated_opacity,
            tau_rosseland=tau_rosseland,
            band_index=band_idx,
            tau_bin_edges=list(edges),
            wavelength_grid_subbins_centers=wavelength_grid_subbins_centers,
            max_height_idx=max_height_idx,
            write_debug_json=False,
            verbose=False,
        )
        overlaps = compute_bot_segment_overlap_per_tau_bin(
            sorted_per_bin,
            tau_bin_edges=list(edges),
            smooth_window=smooth_window,
            print_table=False,
            verbose=False,
            refine_mid=refine_mid,
        )
        return sorted_per_bin, overlaps

    def _high_score(overlaps: dict, k: int) -> float:
        rec = overlaps.get(k, {})
        segs = rec.get("segments")
        if not segs or "high" not in segs:
            return 0.0
        return float(segs["high"]["overlap_pct"])

    def _scores(edges: list[float], overlaps: dict) -> list[float]:
        return [_high_score(overlaps, k) for k in range(len(edges) - 1)]

    def _valid_monotone(edges: list[float]) -> bool:
        return all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))

    console.print(f"\n[cyan]Optimizing tau_bin_edges (threshold={threshold_pct:.1f}%, max_bins={max_bins})...[/cyan]")
    edges = list(initial_tau_bin_edges)
    sorted_per_bin, overlaps = _evaluate(edges)
    scores = _scores(edges, overlaps)
    min_score = min(scores) if scores else 0.0
    console.print(
        f"  iter   0: bins={len(edges) - 1:2d}  min_high={min_score:6.2f}  "
        f"scores={[f'{s:.1f}' for s in scores]}  action=start"
    )

    for it in range(1, max_outer_iters + 1):
        if min_score >= threshold_pct:
            console.print(f"[green]✓ converged at iter {it - 1}[/green]")
            return edges, sorted_per_bin, overlaps

        best_edges = None
        best_sorted_per_bin = None
        best_overlaps = None
        best_min = min_score

        for i in range(1, len(edges)):
            for step in adjust_steps:
                for direction in (-1.0, +1.0):
                    cand = list(edges)
                    cand[i] = edges[i] + direction * step
                    if not _valid_monotone(cand):
                        continue
                    cand_sorted, cand_overlaps = _evaluate(cand)
                    cand_scores = _scores(cand, cand_overlaps)
                    cand_min = min(cand_scores) if cand_scores else 0.0
                    if cand_min > best_min + 1e-9:
                        best_min = cand_min
                        best_edges = cand
                        best_sorted_per_bin = cand_sorted
                        best_overlaps = cand_overlaps

        if best_edges is not None:
            edges = best_edges
            sorted_per_bin = best_sorted_per_bin
            overlaps = best_overlaps
            scores = _scores(edges, overlaps)
            min_score = best_min
            console.print(
                f"  iter {it:3d}: bins={len(edges) - 1:2d}  min_high={min_score:6.2f}  "
                f"scores={[f'{s:.1f}' for s in scores]}  action=adjust"
            )
            continue

        if len(edges) - 1 >= max_bins:
            console.print(f"[yellow]✗ stuck at cap (bins={len(edges) - 1}, min_high={min_score:.2f}%)[/yellow]")
            return edges, sorted_per_bin, overlaps

        k_worst = int(np.argmin(scores))
        new_edge = (edges[k_worst] + edges[k_worst + 1]) / 2.0
        edges = list(edges[: k_worst + 1]) + [new_edge] + list(edges[k_worst + 1 :])
        sorted_per_bin, overlaps = _evaluate(edges)
        scores = _scores(edges, overlaps)
        min_score = min(scores) if scores else 0.0
        console.print(
            f"  iter {it:3d}: bins={len(edges) - 1:2d}  min_high={min_score:6.2f}  "
            f"scores={[f'{s:.1f}' for s in scores]}  "
            f"action=split@k={k_worst} new_edge={new_edge:.3f}"
        )

    console.print(f"[yellow]✗ max_outer_iters={max_outer_iters} reached (min_high={min_score:.2f}%)[/yellow]")
    return edges, sorted_per_bin, overlaps


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
) -> None:
    """
    Save tau-binned opacity products to a structured .npy file.

    The saved file can be loaded with:
        data = np.load("tau_bin_opacities.npy")
        mixed = data["mixed"]
        temperature = data["T"]
        pressure = data["p"]
    """
    planck = np.asarray(tau_bin_results["kappa_planck"], dtype=np.float64)
    rosseland = np.asarray(tau_bin_results["kappa_rosseland"], dtype=np.float64)
    mixed = np.asarray(tau_bin_results["kappa_mixed"], dtype=np.float64)
    members = np.asarray(tau_bin_results["members_per_band"], dtype=np.int32)
    temperature = np.asarray(temperature_grid, dtype=np.float64)
    pressure = np.asarray(pressure_grid, dtype=np.float64)

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

    dtype = np.dtype(
        [
            ("planck", np.float64, (nt, n_pressure, n_bands)),
            ("rosseland", np.float64, (nt, n_pressure, n_bands)),
            ("mixed", np.float64, (nt, n_pressure, n_bands)),
            ("T", np.float64, (nt,)),
            ("p", np.float64, (n_pressure,)),
            ("members_per_band", np.int32, (n_bands,)),
        ]
    )

    packed = np.empty((), dtype=dtype)
    packed["planck"] = planck
    packed["rosseland"] = rosseland
    packed["mixed"] = mixed
    packed["T"] = temperature
    packed["p"] = pressure
    packed["members_per_band"] = members

    np.save(output_file, packed)
    console.print(f"[green]✓ Saved tau-bin opacities to {output_file}[/green]")


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
        # ] = [-0.63, -0.3, -0.15, -0.0, 0.25, 0.7, 1.5, 3.9, 7.0], # refined with mid bin
    ] = [-0.63, -0.4, -0.2375, -0.075, 0.15, 0.7, 1.5, 3.8, 7.0],  # refined without mid
    lambda_bin_edges: Annotated[
        list[float],
        typer.Option(
            "--lambda-bin-edges",
            help="List of wavelength edges to sort opacities at",
        ),
    ] = [3.0, 5.0],
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
    refine_mid: bool = typer.Option(
        True,
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

    if optimize_high_overlap:
        t0 = time.perf_counter()
        optimized_edges, final_sorted_per_bin, _ = optimize_tau_bin_edges(
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
            max_bins=8,
            refine_mid=refine_mid,
        )
        t1 = time.perf_counter()
        console.print(f"[dim]⏱  optimize_tau_bin_edges: {t1 - t0:.3f}s[/dim]")
        console.print(f"[green]optimized tau_bin_edges = {[round(e, 4) for e in optimized_edges]}[/green]")
        compute_bot_segment_overlap_per_tau_bin(
            final_sorted_per_bin,
            tau_bin_edges=list(optimized_edges),
            print_table=True,
            verbose=True,
            refine_mid=refine_mid,
        )
        return

    console.print("\n[cyan]Calculating tau-binned opacities...[/cyan]")

    bin_number = assign_tau_to_bin(
        tau_rosseland_at_tau_lambda_one,
        wavelength_grid_subbins_centers,
        tau_bin_edges=tau_bin_edges,
        lambda_bin_edges=lambda_bin_edges,
    )
    unique_bins = np.unique(bin_number[bin_number >= 0])
    unassigned = int(np.sum(bin_number < 0))
    console.print(f"assigned bands: {unique_bins}")
    console.print(f"unassigned wavelength points: {unassigned}/{len(bin_number)}")

    plot_tau_rosselend_at_tau_lambda_one_vs_wavelength(
        tau_rosseland_at_tau_lambda_one[skip_first_n_wavelengths:],
        wavelength_grid_subbins_centers[skip_first_n_wavelengths:],
        tau_bin_edges=tau_bin_edges,
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
        tau_bin_edges=tau_bin_edges,
        wavelength_grid_subbins_centers=wavelength_grid_subbins_centers,
        max_height_idx=max_height_idx,
    )
    t1 = time.perf_counter()
    n_nonempty = sum(1 for v in sorted_per_bin.values() if not v.get("empty", False))
    console.print(f"sorted opacity dist: {n_nonempty}/{len(sorted_per_bin)} non-empty tau-bins")
    for k, r in sorted_per_bin.items():
        if r.get("empty", False):
            console.print(f"  bin {k}: empty")
        else:
            console.print(
                f"  bin {k}: members={r['members']}, "
                f"T_top={r['T_top']:.1f}K (i={r['i_top']}), "
                f"T_bot={r['T_bot']:.1f}K (i={r['i_bot']})"
            )
    console.print(f"[dim]⏱  sort_weighted_opacity_per_tau_bin: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    plot_sorted_weighted_opacity_per_tau_bin(
        sorted_per_bin,
        tau_bin_edges=tau_bin_edges,
        refine_mid=refine_mid,
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  plot_sorted_weighted_opacity_per_tau_bin: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    compute_bot_segment_overlap_per_tau_bin(
        sorted_per_bin,
        tau_bin_edges=tau_bin_edges,
        refine_mid=refine_mid,
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  compute_bot_segment_overlap_per_tau_bin: {t1 - t0:.3f}s[/dim]")

    t0 = time.perf_counter()
    tau_bin_results = calculate_tau_bin_opacities(
        odf=odf,
        cont=cont,
        band_index=bin_number,
        n_bins=len(tau_bin_edges) - 1,
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

    t0 = time.perf_counter()
    save_tau_bin_opacities_npy(
        tau_bin_output,
        tau_bin_results,
        temperature_grid=np.power(10.0, odf.T),
        pressure_grid=np.power(10.0, odf.P),
    )
    t1 = time.perf_counter()
    console.print(f"[dim]⏱  save_tau_bin_opacities_npy: {t1 - t0:.3f}s[/dim]")


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


if __name__ == "__main__":
    app()
