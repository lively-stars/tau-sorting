
from ast import Raise

import typer
from pathlib import Path
from typing import Optional
import numpy as np
from numpy.typing import NDArray
from netCDF4 import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import matplotlib.pyplot as plt

console = Console()
app = typer.Typer(help="Tau-sorting opacity binning tool")


# Physical constants
CVAC = 2.99792458e10  # cm/s
H = 6.626196e-27  # erg*sec
K_B = 1.380622e-16  # erg/K
TENLOG = 2.30258509299405
LN10 = 4.342944819032518e-1


def planck_function(
    wavelength: NDArray[np.float64], temperature: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the Planck function B_lambda(T) for given wavelengths and temperatures.

    Args:
        wavelength: Wavelength array [nm]
        temperature: Temperature array [K]
    Returns:
        Planck function values B_lambda [erg/s/cm²/ster/cm]
    """
    # Convert nm to cm
    wl_cm = wavelength * 1.0e-7

    B = (
        2
        * H
        * CVAC**2
        / wl_cm ** 5
        / (np.exp(H * CVAC / wl_cm / K_B / temperature) - 1)
    )

    return B

def planck_derivative(
    wavelength: NDArray[np.float64], temperature: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Calculate the derivative of the Planck function dB/dT.

    Args:
        wavelength: Wavelength array [nm]
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

    Uses CGS units:
    h = 6.626e-27 erg s
    c = 2.9979e10 cm/s
    k = 1.3806e-16 erg/K

    Args:
        wavelength: Wavelength array [nm]
        temperature: Temperature array [K]

    Returns:
        derivative: Derivative of the Planck function dB/dT [erg/s/cm²/ster/cm/K]
    """
    # Physical constants in CGS units
    h = H
    c = CVAC
    k_B = K_B
    
    # Convert nm to cm
    wl_cm = wavelength * 1.0e-7

    # Pre-compute constants
    c1 = 2.0 * h * c**2
    c2 = h * c / k_B

    # Calculate the dimensionless exponent term: x = hc / (lambda * k * T)
    # Ensure no division by zero if inputs contain 0
    with np.errstate(divide='ignore', invalid='ignore'):
        x = c2 / (wl_cm * temperature)
    
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
    
    term1 = c1 / (wl_cm**5)
    term2 = (x * exp_x) / (temperature * denom_factor)
    
    derivative = term1 * term2
    
    # Clean up NaNs resulting from 0/0 or inf/inf in extreme limits if necessary
    # (Optional: depends on input guarantees, here we replace NaNs with 0 for safety)
    derivative = np.nan_to_num(derivative)

    return derivative


def plot_planck_and_derivatives():
    """
    Plot the Planck function and its derivatives at 4 evenly spaced wavelengths.
    """
    # Set up wavelength range in nm (the function expects nm and converts internally)
    wavelengths = np.linspace(10, 2000, 1000)  # 10-2000 nm
    
    # Temperature (K)
    T = 5800  # Solar temperature
    T_extra = [5000, 5200, 5400, 5600, 6000, 6200, 6400, 6600]
    
    # Calculate Planck function
    B = planck_function(wavelengths, T)
    
    # Select 4 evenly spaced wavelengths for derivatives
    indices = np.linspace(0, len(wavelengths) - 1, 4, dtype=int)
    selected_wavelengths = wavelengths[indices]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Planck function for main temperature
    ax.plot(wavelengths, B, 'b-', linewidth=2, label=f'T={T} K')
    
    # Plot Planck functions for additional temperatures using viridis colormap
    cmap = plt.cm.viridis
    colors_extra = [cmap(i / len(T_extra)) for i in range(len(T_extra))]
    for i, T_val in enumerate(T_extra):
        B_extra = planck_function(wavelengths, T_val)
        ax.plot(wavelengths, B_extra, '--', color=colors_extra[i], 
                linewidth=1.5, label=f'T={T_val} K')
    
    if True:
        # Plot derivatives at selected wavelengths
        colors = ['red', 'green', 'orange', 'purple']
        for i, (idx, wl) in enumerate(zip(indices, selected_wavelengths)):
            # Calculate derivatives using both methods
            dB_numerical = planck_derivative(wl, T)
            dB_analytic = planck_derivative_analytic(wl, T)
            
            # Print derivative values for comparison
            console.print(f"[cyan]λ={wl:.0f} nm:[/cyan]")
            console.print(f"  Numerical dB/dT = {dB_numerical:.6e}")
            console.print(f"  Analytic dB/dT  = {dB_analytic:.6e}")
            console.print(f"  Difference      = {abs(dB_numerical - dB_analytic):.6e}")
            console.print(f"  Relative error  = {abs(dB_numerical - dB_analytic)/dB_analytic*100:.6f}%\n")
            
            # Get the Planck function value at this wavelength
            B_val = B[idx]
            
            # Create tangent line for visualization
            wl_range = np.linspace(wl - 200, wl + 200, 100)
            tangent_numerical = B_val + dB_numerical * (wl_range - wl)
            tangent_analytic = B_val + dB_analytic * (wl_range - wl)
            
            # Plot the point
            ax.plot(wl, B_val, 'o', color=colors[i], markersize=8, 
                    label=f'λ={wl:.0f} nm')
            
            # Plot tangent lines (numerical in dashed, analytic in dotted)
            ax.plot(wl_range, tangent_numerical, '--', color=colors[i], 
                    alpha=0.6, linewidth=1.5, label=f'Numerical derivative')
            ax.plot(wl_range, tangent_analytic, ':', color=colors[i], 
                    alpha=0.8, linewidth=2, label=f'Analytic derivative')
    
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Planck Function [erg/s/cm²/ster/cm]', fontsize=12)
    ax.set_title(f'Planck Function at Different Temperatures', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig('planck_derivatives.png', dpi=150)
    console.print(f"[green]Plot saved to planck_derivatives.png[/green]")
    plt.show()


if __name__ == "__main__":
    plot_planck_and_derivatives()
    
