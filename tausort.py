#!/usr/bin/env python3
"""
Tau-sorting: Opacity binning for stellar atmospheres

This script reads atmospheric model data, opacity distribution functions (ODFs),
and continuum opacity data to calculate binned opacities for radiative transfer.
"""

from ast import Raise

import typer
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from netCDF4 import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

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
        self.ODF = None  # ODF values [nt, np, nbins, nsubbins]
        self.FreqG = None  # Frequency grid edges
        self.P = None  # Pressure grid
        self.T = None  # Temperature grid
        self.subbin = None  # Sub-bin weights
        self.vturb = None  # Turbulent velocity

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
        self.kappa_scat = None  # Scattering opacity
        self.kappa_all = None  # Total opacity

    def __repr__(self):
        return f"ContinuumData(shape={self.kappa_all.shape if self.kappa_all is not None else None})"


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
            odf.ODF = nc.variables["ODF"][:]  # short integer, convert: 10^(ODF/1000)
            odf.FreqG = nc.variables["FreqG"][:]
            odf.P = nc.variables["P"][:]
            odf.T = nc.variables["T"][:]
            odf.subbin = nc.variables["subbin"][:]

            # Read global attributes
            if hasattr(nc, "vturb"):
                odf.vturb = nc.vturb
                console.print(f"  Turbulent velocity: {odf.vturb} km/s")

            console.print(f"  ✓ ODF loaded successfully")
            console.print(
                f"  Temperature grid: {odf.T.min():.1f} - {odf.T.max():.1f} K"
            )
            console.print(
                f"  Pressure grid: {odf.P.min():.2e} - {odf.P.max():.2e} dyn/cm²"
            )
            console.print(
                f"  Frequency range: {odf.FreqG.min():.2e} - {odf.FreqG.max():.2e} Hz"
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
        scat_file: Path to scattering continuum file
        all_file: Path to combined continuum file
        nlam: Number of wavelength bins
        nt: Number of temperature points
        n_pressure: Number of pressure points
        cont_scatter: Scattering contribution factor (default: 0.0)

    Returns:
        ContinuumData object with all continuum opacities
    """
    cont = ContinuumData()

    # Try to read combined file first
    # if all_file.exists():
    #     cont.kappa_all = read_continuum_opacity(all_file, nlam, nt, n_pressure)

    # Read individual files
    if abs_file.exists():
        cont.kappa_abs = read_continuum_opacity(abs_file, nlam, nt, n_pressure)

    # if scat_file.exists():
    #     cont.kappa_scat = read_continuum_opacity(scat_file, nlam, nt, n_pressure)

    # If we don't have combined, calculate it
    if (
        cont.kappa_all is None
        and cont.kappa_abs is not None
        and cont.kappa_scat is not None
    ):
        console.print("[cyan]Calculating combined continuum opacity[/cyan]")
        cont.kappa_all = cont.kappa_abs + cont.kappa_scat * cont_scatter
        console.print(f"  ✓ Combined with scattering factor = {cont_scatter}")

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
        / wavelength ** 5
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
    B = planck_function(wavelength, temperature)

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
    with np.errstate(divide='ignore', invalid='ignore'):
        x = c2 / (wavelength * temperature)
    
    # Calculate the exponential term
    # We trap overflow warnings here for very small wavelengths/temps where exp(x) -> inf
    with np.errstate(over='ignore'):
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
    

def plot_planck_and_derivatives(
    output_file: str = "planck_verification.png"
) -> None:
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
        ax1.plot(wavelengths_nm, B, linewidth=2, color=color, label=f'T={T} K')
    
    ax1.set_xlabel('Wavelength [nm]', fontsize=12)
    ax1.set_ylabel('Planck Function B [erg/s/cm²/ster/cm]', fontsize=12)
    ax1.set_title('Planck Function at Various Temperatures', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())
    
    # === Right plot: Derivatives with tangent lines ===
    selected_wavelengths_nm = [100, 250, 500, 750]  # nm
    selected_wavelengths = np.array(selected_wavelengths_nm) * 1.0e-7  # convert to cm
    colors_deriv = ['red', 'blue', 'green', 'orange']
    
    # Plot main Planck function on right panel for tangent visualization
    T_tangent = 5800  # K (Solar temperature for tangent lines)
    B_for_tangent = planck_function(wavelengths, T_tangent)
    ax2.plot(wavelengths_nm, B_for_tangent, 'k-', linewidth=2, alpha=0.5, 
             label=f'B(λ) at T={T_tangent}K')
    
    # Add T±200K curves
    B_plus_200 = planck_function(wavelengths, T_tangent + 200)
    B_minus_200 = planck_function(wavelengths, T_tangent - 200)
    ax2.plot(wavelengths_nm, B_plus_200, 'k--', linewidth=1.5, alpha=0.3, 
             label=f'T={T_tangent+200}K')
    ax2.plot(wavelengths_nm, B_minus_200, 'k:', linewidth=1.5, alpha=0.3, 
             label=f'T={T_tangent-200}K')
    
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
        ax2.plot(wl_nm, B_val, 'o', color=color, markersize=10, 
                label=f'λ={wl_nm} nm', zorder=5)
        
        # Plot tangent line
        ax2.plot(wl_tangent_range, tangent_visual, '--', color=color, 
                alpha=0.8, linewidth=2.5)
    
    ax2.set_xlabel('Wavelength [nm]', fontsize=12)
    ax2.set_ylabel('Planck Function B [erg/s/cm²/ster/cm]', fontsize=12)
    ax2.set_title(f'dB/dT Tangent Lines at T={T_tangent}K', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=7, loc='best', ncol=2)
    ax2.set_xlim(wavelengths_nm.min(), wavelengths_nm.max())
    
    plt.suptitle('Planck Function and Derivative Verification', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
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

    nt, np_dim = odf.nt, odf.np
    result = np.zeros((nt, np_dim), dtype=np.float64)

    # first we add the continuum opacity to the ODF based on the bin
    continuum_kappa: NDArray[np.float64] | None = (
        cont.kappa_abs
    )  # shape: [nt, np, nbins]
    odf_kappa: NDArray[np.float64] = 10 ** (
        odf.ODF / 1000
    )  # shape: [nt, np, nbins, nsubbins]

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

    kappa_reference = np.trapz()

    return total_kappa


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

    reference_opacities = calculate_reference_opacities(odf, cont, kind="rosseland")

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
        "--output", "-o",
        help="Output filename for verification plot"
    )
):
    """
    Verify Planck function and derivative calculations.
    
    Creates plots of the Planck function and its temperature derivative
    at various temperatures for verification purposes.
    """
    plot_planck_and_derivatives(output)


if __name__ == "__main__":
    app()
