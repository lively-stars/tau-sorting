#!/usr/bin/env python3
"""
Tau-sorting: Opacity binning for stellar atmospheres

This script reads atmospheric model data, opacity distribution functions (ODFs),
and continuum opacity data to calculate binned opacities for radiative transfer.
"""

from turtle import right

from ast import Raise

import typer
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from netCDF4 import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

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
        self.z = None  # height [cm]
        self.rho = None  # density [g/cm^3]
        self.p = None  # pressure [dyn/cm^2]
        self.T = None  # temperature [K]
        self.nlevels = 0

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
        self.ODF: NDArray[np.float64] | None = (
            None  # ODF values [nt, np, nbins, nsubbins]
        )
        self.wavelength_grid: NDArray[np.float64] | None = None  # Frequency grid edges
        self.P: NDArray[np.float64] | None = None  # Pressure grid
        self.T: NDArray[np.float64] | None = None  # Temperature grid
        self.subbin: NDArray[np.float64] | None = None  # Sub-bin weights
        self.vturb: NDArray[np.float64] | None = None  # Turbulent velocity

        # Dimensions
        self.np = 0
        self.nt = 0
        self.nbins = 0
        self.nsubbins = 0
        self.numfp = 0

    def __repr__(self):
        return (
            f"ODFData(nt={self.nt}, np={self.np}, "
            f"nbins={self.nbins}, nsubbins={self.nsubbins})"
        )


class ContinuumData:
    """Container for continuum opacity data"""

    def __init__(self):
        self.kappa_abs = None  # Absorption opacity

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
        console.print(
            f"  Pressure range: {atm.p.min():.2e} - {atm.p.max():.2e} dyn/cm²"
        )

        return atm

    except Exception as e:
        console.print(f"[red]Error reading atmospheric model: {e}[/red]")
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

            console.print(
                f"  Dimensions: nt={odf.nt}, np={odf.np}, "
                f"nbins={odf.nbins}, nsubbins={odf.nsubbins}"
            )

            # Read variables
            odf.ODF = 10 ** (
                nc.variables["ODF"][:] / 1000
            )  # short integer, convert: 10^(ODF/1000)
            odf.wavelength_grid = nc.variables["FreqG"][:]
            odf.P = nc.variables["P"][:]
            odf.T = nc.variables["T"][:]
            odf.subbin = nc.variables["subbin"][:]

            # Read global attributes
            if hasattr(nc, "vturb"):
                odf.vturb = nc.vturb
                console.print(f"  Turbulent velocity: {odf.vturb} km/s")

            console.print("  ✓ ODF loaded successfully")
            console.print(
                f"  Temperature grid: {odf.T.min():.1f} - {odf.T.max():.1f} K"
            )
            console.print(
                f"  Pressure grid: {odf.P.min():.2e} - {odf.P.max():.2e} dyn/cm²"
            )
            console.print(
                f"  Frequency range: {odf.wavelength_grid.min():.2e} - {odf.wavelength_grid.max():.2e} Hz"
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


def read_continuum_opacity(
    filepath: Path, n_bins: int, n_temperature: int, n_pressure: int
) -> np.ndarray:
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
            console.print(f"  [green]Using fast .npy format[/green]")
            kappa = np.load(npy_path)
        else:
            # Fall back to ASCII .dat file
            console.print(
                f"  [yellow]Loading ASCII (slow) - consider converting to .npy[/yellow]"
            )
            data = np.loadtxt(str(filepath))
            expected_size = n_bins * n_temperature * n_pressure

            if data.size != expected_size:
                console.print(
                    f"[yellow]Warning: Expected {expected_size} values, "
                    f"got {data.size}[/yellow]"
                )

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
    cont_scatter: float = 0.0,
) -> ContinuumData:
    """
    Read all continuum opacity data files.

    Args:
        abs_file: Path to absorption continuum file
        nlam: Number of wavelength bins
        nt: Number of temperature points
        n_pressure: Number of pressure points
        cont_scatter: Scattering contribution factor (default: 0.0)

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
            console.print(
                f"  ✗ Shape mismatch: continuum={cont.kappa_abs.shape}, "
                f"expected={expected_shape}"
            )

    # Summary
    console.print(
        f"\n[bold]Verification: {checks_passed}/{checks_total} checks passed[/bold]"
    )

    if checks_passed == checks_total:
        console.print("[green]✓ All data verified successfully![/green]")
    else:
        console.print("[yellow]⚠ Some verification checks failed[/yellow]")

    return checks_passed == checks_total


def planck_function(
    wavelength: NDArray[np.float64], temperature: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the Planck function B_lambda(T) for given wavelengths and temperatures.

    Args:
        wavelength: Wavelength array [cm]
        temperature: Temperature array [K]
    Returns:
        Planck function values B_lambda [erg/s/cm²/ster/cm]
    """

    B = (
        2
        * H
        * CVAC**2
        / wavelength**5
        / (np.exp(H * CVAC / wavelength / K_B / temperature) - 1)
    )

    return B


def planck_derivative(
    wavelength: NDArray[np.float64], temperature: NDArray[np.float64]
) -> NDArray[np.float64]:
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


def planck_derivative_analytic(
    wavelength: NDArray[np.float64], temperature: NDArray[np.float64]
) -> NDArray[np.float64]:
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


def plot_planck_and_derivatives(output_file: str = "planck_verification.png") -> None:
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
    cmap = plt.cm.plasma
    for i, T in enumerate(T_range):
        B = planck_function(wavelengths, T)
        color = cmap(i / len(T_range))
        ax1.plot(wavelengths_nm, B, linewidth=2, color=color, label=f"T={T} K")

    ax1.set_xlabel("Wavelength [nm]", fontsize=12)
    ax1.set_ylabel("Planck Function B [erg/s/cm²/ster/cm]", fontsize=12)
    ax1.set_title(
        "Planck Function at Various Temperatures", fontsize=13, fontweight="bold"
    )
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
        label=f"T={T_tangent+200}K",
    )
    ax2.plot(
        wavelengths_nm,
        B_minus_200,
        "k:",
        linewidth=1.5,
        alpha=0.3,
        label=f"T={T_tangent-200}K",
    )

    # Draw tangent lines at selected wavelengths
    for wl_nm, wl_cm, color in zip(
        selected_wavelengths_nm, selected_wavelengths, colors_deriv
    ):
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
        tangent_visual = (
            B_val + dB_dT * perturbation * 500
        )  # Scale by 500K for visibility

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
    ax2.set_title(
        f"dB/dT Tangent Lines at T={T_tangent}K", fontsize=13, fontweight="bold"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc="best", ncol=2)
    ax2.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())

    plt.suptitle(
        "Planck Function and Derivative Verification", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    console.print(f"[green]✓ Plot saved to {output_file}[/green]")

    # Print numerical verification at selected wavelengths
    console.print("\n[cyan]Numerical verification at selected wavelengths:[/cyan]")
    T_test = 5800  # K (Solar temperature)
    for wl_nm, wl_cm in zip(selected_wavelengths_nm, selected_wavelengths):
        B = planck_function(np.array([wl_cm]), T_test)[0]
        dB_dT = planck_derivative_analytic(np.array([wl_cm]), T_test)[0]
        wien_peak = 2.898e6 / T_test  # Wien's displacement law

        console.print(f"\n  λ = {wl_nm} nm, T = {T_test} K:")
        console.print(f"    B(λ,T)      = {B:.6e} erg/s/cm²/ster/cm")
        console.print(f"    dB/dT       = {dB_dT:.6e} erg/s/cm²/ster/cm/K")
        console.print(f"    Wien peak λ = {wien_peak:.1f} nm")

    console.print("\n[green]✓ Planck function verification complete![/green]\n")


def interpolate_kappa_to_atmosphere(
    odf: ODFData, cont: ContinuumData, atmo: AtmosphericData
) -> NDArray[np.float64]:
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
    total_kappa = (
        odf_kappa + continuum_kappa[..., np.newaxis]
    )  # shape: [nt, np, nbins, nsubbins]

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
                raise ValueError(
                    f"Data shape {data_grid.shape} does not match grids ({len(t_grid)}, {len(p_grid)})"
                )

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
                    f"Temperature query contains values out of bounds [{min_t}, {max_t}]. "
                    f"Violations: {bad_vals[:3]}..."
                )

            if np.any(p_pts < self.p_grid.min()) or np.any(p_pts > self.p_grid.max()):
                min_p, max_p = self.p_grid.min(), self.p_grid.max()
                bad_vals = p_pts[(p_pts < min_p) | (p_pts > max_p)]
                p_pts = np.clip(p_pts, a_min=min_p, a_max=max_p)
                console.print(
                    f"Pressure query contains values out of bounds [{min_p}, {max_p}]. "
                    f"Violations: {bad_vals[:3]}..."
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
        interpolated_opacity = interpolator.get_vectors(
            np.log10(atmo.T), np.log10(atmo.p)
        )

        # console.print(f"Interpolation successful. Output shape: {interpolated_opacity.shape}")
        return interpolated_opacity

    except ValueError as e:
        # Handle bounds errors (e.g., atmosphere is hotter than ODF table)
        # console.print(f"[red]Interpolation Error:[/red] {e}")
        raise e


def calculate_reference_opacities(
    odf: ODFData, cont: ContinuumData, kind: str = "rosseland"
) -> NDArray[np.float64]:
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

    console.print(
        f"shapes: continuum_kappa={continuum_kappa.shape}, odf_kappa={odf_kappa.shape}"
    )

    # add continuum kappa to each subbin based on the bin
    # if odf.nbins != continuum_kappa.shape[-2]:
    #     console.print(f"[red]Error: ODF nbins ({odf.nbins}) does not match continuum kappa shape ({continuum_kappa.shape})[/red]")
    #     raise ValueError("Inconsistent nbins between ODF and continuum data")

    total_kappa = (
        odf_kappa + continuum_kappa[..., np.newaxis]
    )  # shape: [nt, np, nbins, nsubbins]

    # for each T, p point from the ODF calculate the reference opacity using np.nditerate
    temperature_grid = odf.T  # shape: [nt]
    pressure_grid = odf.P  # shape: [np]
    console.print(f"Calculating {kind} reference opacities...")
    console.print(f"  Temperature grid: {temperature_grid.shape}")
    console.print(f"  Pressure grid: {pressure_grid.shape}")
    temperature_pressure_grid = np.array(
        [(temp, pres) for temp in temperature_grid for pres in pressure_grid]
    )
    console.print(f"  Temp-Pressure grid shape: {temperature_pressure_grid.shape}")
    for idx, (temperature, pressure) in tqdm(
        enumerate(temperature_pressure_grid), total=temperature_pressure_grid.shape[0]
    ):
        t_idx = np.where(odf.T == temperature)[0][0]
        p_idx = np.where(odf.P == pressure)[0][0]

        kappa_values = total_kappa[
            t_idx, p_idx, ...
        ].flatten()  # shape: [nbins * nsubbins]
        # verify kappa_values length matches expected
        expected_length = odf.nbins * odf.nsubbins
        if kappa_values.shape[0] != expected_length:
            console.print(
                f"[red]Error: kappa_values length ({kappa_values.shape[0]}) does not match expected ({expected_length})[/red]"
            )
            raise ValueError("Inconsistent kappa values length")

        # Assign sub-bin wavelengths based on ODF frequency grid and sub-bin weights
        wavelength_grid_bin_edges = odf.wavelength_grid * 1e-8  # cm
        wavelength_grid_bin_size = np.diff(wavelength_grid_bin_edges)
        # console.print(f"  Wavelength grid shape: {wavelength_grid_bin_edges.shape}")
        # console.print(f"  Wavelength grid values: {wavelength_grid_bin_edges[:10]}")

        wavelength_grid_subbin_weights = odf.subbin
        wavelength_grid_subbins_center = np.zeros_like(kappa_values)
        # console.print(f"odf.subbin shape: {odf.subbin.shape}")
        number_of_subbins: int = odf.subbin.shape[1]
        wavelength_grid_subbins_edges_shape = (
            odf.subbin.shape[0] * (number_of_subbins) + 1
        )
        wavelength_grid_subbins_edges = np.zeros(
            wavelength_grid_subbins_edges_shape, dtype=np.float64
        )
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
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = (
                    sub_bin_edges
                )
                # console.print(f"setting the subbins from {counter} to {counter+len(subbin_sizes)}")
                # console.print(f" sub_bin_edges: {sub_bin_edges}")
                # console.print(f"  First bin edges: {wavelength_grid_subbins_edges[:15]}")
                # console.print(f"  Bin {bin_idx} edges: {wavelength_grid_subbins_edges[counter:counter+len(subbin_sizes)+1]}")
                counter += len(subbin_weights)
            # for subsequent bins we set the left edge to the previous right edge
            else:
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = (
                    sub_bin_edges
                )
                # console.print(f"  Bin {bin_idx} edges: {wavelength_grid_subbins_edges[counter-1:counter+len(subbin_sizes)]}")
                counter += len(subbin_weights)
            # console.print(f" After bin {bin_idx}, counter={counter}")
            # console.print(f" Current wavelength_grid_subbins_edges: {wavelength_grid_subbins_edges[:counter+2]}")
        counter -= 1  # adjust for last increment
        if counter != len(kappa_values):
            console.print(
                "[red]Error: Mismatch in wavelength grid and kappa values length[/red]"
            )
            raise ValueError(
                f"Wavelength grid and kappa values length mismatch: counter={counter}, kappa_values={len(kappa_values)}"
            )

        wavelength_grid_subbins_centers = 0.5 * (
            wavelength_grid_subbins_edges[:-1] + wavelength_grid_subbins_edges[1:]
        )
        B_lambda = planck_function(wavelength_grid_subbins_centers, temperature)
        dB_dT = planck_derivative_analytic(wavelength_grid_subbins_centers, temperature)
        if kind == "rosseland":
            # Rosseland mean opacity
            if kappa_values.shape != wavelength_grid_subbins_centers.shape:
                console.print(
                    f"[red]Error: kappa_values shape {kappa_values.shape} does not match wavelength grid shape {wavelength_grid_subbins_centers.shape}[/red]"
                )
                raise ValueError(
                    "Inconsistent shapes for kappa values and wavelength grid"
                )
            integrand_num = np.trapezoid(
                (1.0 / kappa_values) * dB_dT, wavelength_grid_subbins_centers
            )
            integrand_den = np.trapezoid(dB_dT, wavelength_grid_subbins_centers)
            kappa_rosseland = (
                integrand_den / integrand_num if integrand_num != 0 else 0.0
            )
            total_kappa[t_idx, p_idx] = kappa_rosseland

        elif kind == "planck":
            # Planck mean opacity
            integrand_num = np.trapezoid(
                kappa_values * B_lambda, wavelength_grid_subbins_centers
            )
            integrand_den = np.trapezoid(B_lambda, wavelength_grid_subbins_centers)
            kappa_planck = integrand_num / integrand_den if integrand_den != 0 else 0.0
            total_kappa[t_idx, p_idx] = kappa_planck

        elif kind == "500nm":
            # Opacity at 500nm
            wl_500nm = 500e-7  # cm
            idx_500nm = np.argmin(np.abs(wavelength_grid - wl_500nm))
            total_kappa[t_idx, p_idx] = kappa_values[idx_500nm]

    return total_kappa


def calculate_reference_opacities_from_custom_tp_grid(
    atmo: AtmosphericData, reference_opacities: NDArray[np.float64], wavelength_grid: NDArray[np.float64], subbin: NDArray[np.float64], nbins: int, nsubbins: int, kind: str = "rosseland"
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
    # console.print(f" total_kappa shape: {total_kappa.shape}")

    temperature_grid = atmo.T  # shape: [nt]
    pressure_grid = atmo.p  # shape: [np]
    console.print(f"Calculating {kind} reference opacities...")
    console.print(f"  Temperature grid: {temperature_grid.shape}")
    console.print(f"  Pressure grid: {pressure_grid.shape}")
    temperature_pressure_grid = np.column_stack(
        (temperature_grid, pressure_grid))
    console.print(f"  Temp-Pressure grid shape: {temperature_pressure_grid.shape}")
    for atmosphere_depth_idx, (temperature, pressure) in tqdm(
        enumerate(temperature_pressure_grid), total=temperature_pressure_grid.shape[0]
    ):
        t_idx = np.where(atmo.T == temperature)[0][0]
        p_idx = np.where(atmo.p == pressure)[0][0]

        kappa_values = total_kappa[
            atmosphere_depth_idx, ...
        ].flatten()  # shape: [nbins * nsubbins]
        # verify kappa_values length matches expected
        # console.print(f" total_kappa shape: {total_kappa.shape}")
        expected_length = nbins * nsubbins
        if kappa_values.shape[0] != expected_length:
            console.print(
                f"[red]Error: kappa_values length ({kappa_values.shape[0]}) does not match expected ({expected_length})[/red]"
            )
            raise ValueError("Inconsistent kappa values length")

        # Assign sub-bin wavelengths based on ODF frequency grid and sub-bin weights
        wavelength_grid_bin_edges = wavelength_grid  # cm
        wavelength_grid_bin_size = np.diff(wavelength_grid_bin_edges)
        # console.print(f"  Wavelength grid shape: {wavelength_grid_bin_edges.shape}")
        # console.print(f"  Wavelength grid values: {wavelength_grid_bin_edges[:10]}")

        wavelength_grid_subbin_weights = subbin
        wavelength_grid_subbins_center = np.zeros_like(kappa_values)
        # console.print(f"odf.subbin shape: {odf.subbin.shape}")
        number_of_subbins: int = subbin.shape[1]
        wavelength_grid_subbins_edges_shape = (
            subbin.shape[0] * (number_of_subbins) + 1
        )
        wavelength_grid_subbins_edges = np.zeros(
            wavelength_grid_subbins_edges_shape, dtype=np.float64
        )
        counter = 1
        for bin_idx in range(nbins):
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
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = (
                    sub_bin_edges
                )
                # console.print(f"setting the subbins from {counter} to {counter+len(subbin_sizes)}")
                # console.print(f" sub_bin_edges: {sub_bin_edges}")
                # console.print(f"  First bin edges: {wavelength_grid_subbins_edges[:15]}")
                # console.print(f"  Bin {bin_idx} edges: {wavelength_grid_subbins_edges[counter:counter+len(subbin_sizes)+1]}")
                counter += len(subbin_weights)
            # for subsequent bins we set the left edge to the previous right edge
            else:
                wavelength_grid_subbins_edges[counter : counter + len(subbin_sizes)] = (
                    sub_bin_edges
                )
                # console.print(f"  Bin {bin_idx} edges: {wavelength_grid_subbins_edges[counter-1:counter+len(subbin_sizes)]}")
                counter += len(subbin_weights)
            # console.print(f" After bin {bin_idx}, counter={counter}")
            # console.print(f" Current wavelength_grid_subbins_edges: {wavelength_grid_subbins_edges[:counter+2]}")
        counter -= 1  # adjust for last increment
        if counter != len(kappa_values):
            console.print(
                "[red]Error: Mismatch in wavelength grid and kappa values length[/red]"
            )
            raise ValueError(
                f"Wavelength grid and kappa values length mismatch: counter={counter}, kappa_values={len(kappa_values)}"
            )

        wavelength_grid_subbins_centers = 0.5 * (
            wavelength_grid_subbins_edges[:-1] + wavelength_grid_subbins_edges[1:]
        )
        B_lambda = planck_function(wavelength_grid_subbins_centers, temperature)
        dB_dT = planck_derivative_analytic(wavelength_grid_subbins_centers, temperature)
        if kind == "rosseland":
            # Rosseland mean opacity
            if kappa_values.shape != wavelength_grid_subbins_centers.shape:
                console.print(
                    f"[red]Error: kappa_values shape {kappa_values.shape} does not match wavelength grid shape {wavelength_grid_subbins_centers.shape}[/red]"
                )
                raise ValueError(
                    "Inconsistent shapes for kappa values and wavelength grid"
                )
            integrand_num = np.trapezoid(
                (1.0 / kappa_values) * dB_dT, wavelength_grid_subbins_centers
            )
            integrand_den = np.trapezoid(dB_dT, wavelength_grid_subbins_centers)
            kappa_rosseland = (
                integrand_den / integrand_num if integrand_num != 0 else 0.0
            )
            opacity_at_tp_points[atmosphere_depth_idx] = kappa_rosseland

        elif kind == "planck":
            # Planck mean opacity
            integrand_num = np.trapezoid(
                kappa_values * B_lambda, wavelength_grid_subbins_centers
            )
            integrand_den = np.trapezoid(B_lambda, wavelength_grid_subbins_centers)
            kappa_planck = integrand_num / integrand_den if integrand_den != 0 else 0.0
            opacity_at_tp_points[atmosphere_depth_idx] = kappa_planck

        elif kind == "500nm":
            # Opacity at 500nm
            wl_500nm = 500e-7  # cm
            idx_500nm = np.argmin(np.abs(wavelength_grid - wl_500nm))
            opacity_at_tp_points[atmosphere_depth_idx] = kappa_values[idx_500nm]

    return opacity_at_tp_points

def compute_tau_rosseland(
    atmo: AtmosphericData, kappa_rosseland: NDArray[np.float64]
) -> NDArray[np.float64]:
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
    tau_rosseland = np.zeros(n_layers-1, dtype=np.float64)
    height = atmo.z[::-1]  # cm, reverse to go from top to bottom
    density = atmo.rho  # g/cm³, reverse to match

    console.print("Calculating Rosseland optical depth profile...")

    if kappa_rosseland.shape[0] != n_layers:
        console.print(
            f"[red]Error: kappa_rosseland shape {kappa_rosseland.shape} does not match atmospheric layers {n_layers}[/red]"
        )
        raise ValueError("Inconsistent shapes for kappa rosseland and atmospheric layers")
    
    for layer_idx, _ in enumerate(height[:-1]):
        console.print(f" Layer {layer_idx}: height={height[layer_idx]:.2e} cm, density={density[layer_idx]:.2e} g/cm³, kappa_rosseland={kappa_rosseland[layer_idx]:.2e} cm²/g")
        
        density_integrand = density[:layer_idx + 1]
        height_integrand = height[:layer_idx + 1]
        kappa_integrand = kappa_rosseland[:layer_idx + 1]
    
        console.print(f"Performing trapezoidal integration over {layer_idx} layers...")
    
        tau_rosseland[layer_idx] = np.trapezoid(kappa_integrand * density_integrand, height_integrand)
    
    console.print("Rosseland optical depth profile calculation complete.")

    return tau_rosseland


@app.command()
def main(
    atm_file: Path = typer.Option(
        "G2_1D.dat",
        "--atm",
        "-a",
        help="1D atmospheric model file (height, density, pressure, temperature)",
    ),
    odf_file: Path = typer.Option(
        "ODF_nc_format.nc", "--odf", "-o", help="ODF data in NetCDF format"
    ),
    continuum_abs: Path = typer.Option(
        "continuumabs.dat", "--cont-abs", help="Continuum absorption opacity file"
    ),
    continuum_scat: Path = typer.Option(
        "continuumscat.dat", "--cont-scat", help="Continuum scattering opacity file"
    ),
    continuum_all: Path = typer.Option(
        "continuumall.dat", "--cont-all", help="Combined continuum opacity file"
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-O",
        help="Output file for binned opacities (default: kappa_<nbands>_band.dat)",
    ),
    nbands: int = typer.Option(2, "--nbands", "-n", help="Number of opacity bands"),
    cont_scatter: float = typer.Option(
        0.0, "--scatter", "-s", help="Continuum scattering contribution factor"
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
    try:
        atm = read_atmospheric_model(atm_file)
    except Exception:
        raise typer.Exit(code=1)

    # Step 2: Read ODF data
    try:
        odf = read_odf_netcdf(odf_file)
    except Exception:
        raise typer.Exit(code=1)

    # Step 3: Read continuum data
    try:
        cont = read_continuum_data(
            continuum_abs,
            # continuum_scat,
            # continuum_all,
            odf.nbins,
            odf.nt,
            odf.np,
            cont_scatter,
        )
    except Exception:
        raise typer.Exit(code=1)

    # Step 4: Verify data consistency
    if not verify_data_consistency(atm, odf, cont):
        console.print(
            "\n[red]Data verification failed. Please check your input files.[/red]"
        )
        raise typer.Exit(code=1)

    console.print("\n[green]✓ All input data loaded and verified successfully![/green]")
    console.print("\n[cyan]Ready to proceed with opacity binning...[/cyan]")

    # using reference opacities calculate Kappa_reference for each T, p point
    # reference_opacities = calculate_reference_opacities(odf, cont, kind="rosseland")

    # reference_opacities.shape = (nt, np)

    # Interpolate reference opacities onto atmospheric model grid (T, p)
    interpolated_opacity = interpolate_kappa_to_atmosphere(odf, cont, atm)
    console.print(f"interpolated_opacity shape: {interpolated_opacity.shape}")


    console.print("Calculate kappa rosseland at each atmosphere T, p point...")
    
    kappa_on_atmosphere_tp = calculate_reference_opacities_from_custom_tp_grid(
        atm,
        interpolated_opacity,
        odf.wavelength_grid * 1e-8,  # convert to cm
        odf.subbin,
        odf.nbins,
        odf.nsubbins,
        kind="rosseland"
    )
    console.print(f"kappa_on_atmosphere_tp shape: {kappa_on_atmosphere_tp.shape}")
    
    tau_rosseland = compute_tau_rosseland(atm, kappa_on_atmosphere_tp)
    console.print(f"tau_rosseland shape: {tau_rosseland.shape}")
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    height_mm = atm.z / 1e8  # Convert cm to Mm
    ax.semilogy(height_mm[::-1][:-1], tau_rosseland, 'b-', linewidth=2, label='Rosseland optical depth')
    ax.set_xlabel('Height [Mm]', fontsize=12)
    ax.set_ylabel('Rosseland Optical Depth τ', fontsize=12)
    ax.set_title('Rosseland Optical Depth Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    plt.tight_layout()
    tau_plot_file = 'tau_rosseland_profile.png'
    plt.savefig(tau_plot_file, dpi=150, bbox_inches='tight')
    console.print(f"[green]✓ Tau Rosseland profile plot saved to {tau_plot_file}[/green]")
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    temperature = atm.T
    ax.semilogx(tau_rosseland, temperature[:-1], 'b-', linewidth=2, label='Rosseland optical depth')
    ax.set_ylabel('Temperature [K]', fontsize=12)
    ax.set_xlabel('Rosseland Optical Depth τ', fontsize=12)
    # ax.set_title('Rosseland Optical Depth Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    plt.tight_layout()
    # tau_plot_file = 'tau_rosseland_profile.png'
    # plt.savefig(tau_plot_file, dpi=150, bbox_inches='tight')
    console.print(f"[green]✓ Tau Rosseland profile plot saved to {tau_plot_file}[/green]")
    plt.show()
    
    # TODO: Implement the following steps:
    # - Initialize grids and interpolation
    # - Calculate reference opacities (Rosseland, 500nm, etc.)
    # - Perform tau-sorting
    # - Calculate band-averaged opacities
    # - Write output file

    console.print("\n[yellow]Processing implementation in progress...[/yellow]")


@app.command()
def verify_planck(
    output: str = typer.Option(
        "planck_verification.png",
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
