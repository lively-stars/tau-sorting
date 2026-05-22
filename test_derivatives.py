import numpy as np

from planck import planck_derivative, planck_derivative_analytic, planck_function

# Test at a single wavelength and temperature
wl = 500.0  # nm
T = 5800.0  # K

# Calculate function value
B = planck_function(wl, T)
print(f"B({wl} nm, {T} K) = {B:.6e}")

# Calculate derivatives
dB_num = planck_derivative(wl, T)
dB_ana = planck_derivative_analytic(wl, T)

print(f"Numerical dB/dT = {dB_num:.6e}")
print(f"Analytic dB/dT = {dB_ana:.6e}")

# Check if they should be dB/dT by testing a small temperature change
dT = 1.0  # 1 K change
B_plus = planck_function(wl, T + dT)
expected_change = dB_num * dT
actual_change = B_plus - B

print(f"\nExpected change for dT={dT}K: {expected_change:.6e}")
print(f"Actual change: {actual_change:.6e}")
print(f"Ratio: {actual_change / expected_change:.6f}")
