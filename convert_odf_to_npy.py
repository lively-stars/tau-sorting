#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "netcdf4",
# ]
# ///
"""
Convert ODF NetCDF format to NumPy .npy format for faster loading.

This script reads the ODF_nc_format.nc file and converts it to a structured
NumPy array saved as ODF_format.npy for significantly faster I/O.
"""

import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def convert_odf_netcdf_to_npy(
    input_file: Path = Path("ODF_nc_format.nc"),
    output_file: Path = Path("ODF_format.npy"),
):
    """
    Convert ODF NetCDF file to NumPy structured array format.

    The structured array uses named fields for easy access while maintaining
    all data in a single .npy file.
    """
    print(f"Reading ODF data from {input_file}...")

    try:
        with Dataset(input_file, "r") as nc:
            # Read dimensions
            np_dim = nc.dimensions["np"].size
            nt_dim = nc.dimensions["nt"].size
            nbins = nc.dimensions["nbins"].size
            nsubbins = nc.dimensions["nsubbins"].size
            numfp = nc.dimensions["numfp"].size

            print(f"  Dimensions: nt={nt_dim}, np={np_dim}, nbins={nbins}, nsubbins={nsubbins}")

            # Read variables. The ODF is stored as a short integer and reconstructed as
            # 10**(short/1000), so its true precision is ~0.23% — float32 (~1e-7 relative) is
            # far finer than the source, so we store it as float32 to halve memory + disk with
            # no meaningful precision loss. (float32 also halves the conversion-time temporary.)
            odf_data = (10 ** (nc.variables["ODF"][:] / 1000)).astype(np.float32)  # short-int source
            wavelength_grid = nc.variables["FreqG"][:]
            pressure = nc.variables["P"][:]
            temperature = nc.variables["T"][:]
            subbin = nc.variables["subbin"][:]

            # Read global attributes
            vturb = nc.vturb if hasattr(nc, "vturb") else 0.0

            print(f"  Temperature range: {temperature.min():.1f} - {temperature.max():.1f} K")
            print(f"  Pressure range: {pressure.min():.2e} - {pressure.max():.2e} dyn/cm²")
            print(f"  Frequency range: {wavelength_grid.min():.2e} - {wavelength_grid.max():.2e} Hz")
            print(f"  Turbulent velocity: {vturb} km/s")

    except Exception as e:
        print(f"Error reading NetCDF file: {e}", file=sys.stderr)
        raise

    # Create structured array with named fields
    # This allows access like odf_array['ODF'], odf_array['T'], etc.
    print("\nCreating structured array...")

    # Create a single structured array element
    dtype = np.dtype(
        [
            ("ODF", f"({nt_dim},{np_dim},{nbins},{nsubbins})f4"),  # float32: source is ~0.23% precise
            ("wavelength_grid", f"({numfp},)f8"),
            ("P", f"({np_dim},)f8"),
            ("T", f"({nt_dim},)f8"),
            ("subbin", f"({nbins},{nsubbins})f8"),
            ("vturb", "f8"),
            ("nt", "i4"),
            ("np", "i4"),
            ("nbins", "i4"),
            ("nsubbins", "i4"),
            ("numfp", "i4"),
        ]
    )

    odf_structured = np.zeros(1, dtype=dtype)
    odf_structured["ODF"][0] = odf_data
    odf_structured["wavelength_grid"][0] = wavelength_grid
    odf_structured["P"][0] = pressure
    odf_structured["T"][0] = temperature
    odf_structured["subbin"][0] = subbin
    odf_structured["vturb"][0] = vturb
    odf_structured["nt"][0] = nt_dim
    odf_structured["np"][0] = np_dim
    odf_structured["nbins"][0] = nbins
    odf_structured["nsubbins"][0] = nsubbins
    odf_structured["numfp"][0] = numfp

    print(f"Saving to {output_file}...")
    np.save(output_file, odf_structured)

    # Verify the saved file
    print("\nVerifying saved file...")
    loaded = np.load(output_file, allow_pickle=True)
    print("  ✓ File saved successfully")
    print(f"  File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Original NetCDF size: {input_file.stat().st_size / 1024 / 1024:.2f} MB")

    # Quick verification
    assert loaded["nt"][0] == nt_dim
    assert loaded["np"][0] == np_dim
    assert loaded["nbins"][0] == nbins
    assert loaded["nsubbins"][0] == nsubbins
    assert np.allclose(loaded["T"][0], temperature)
    assert np.allclose(loaded["P"][0], pressure)
    print("  ✓ Verification passed")

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert ODF NetCDF format to NumPy .npy format")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("ODF_nc_format.nc"),
        help="Input NetCDF file (default: ODF_nc_format.nc)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("ODF_format.npy"),
        help="Output .npy file (default: ODF_format.npy)",
    )

    args = parser.parse_args()

    convert_odf_netcdf_to_npy(args.input, args.output)
    print("\n✓ Conversion complete!")
