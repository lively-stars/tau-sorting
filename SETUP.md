# Tau-Sorting Python Setup

## Environment Setup

This project uses `uv` for Python package management and `typer` for command-line interface.

### Installation

The project has been initialized with:
```bash
uv init --name tausort --no-workspace
uv add typer netcdf4 numpy scipy
```

### Dependencies

- **typer**: Modern CLI framework with automatic help generation
- **netcdf4**: Reading ODF data from NetCDF files
- **numpy**: Numerical computations
- **scipy**: Scientific computing (for interpolation, etc.)
- **rich**: Beautiful terminal output (included with typer)

## Usage

### Basic Usage

Run with default parameters:
```bash
uv run python tausort.py
```

### Command-Line Options

```bash
uv run python tausort.py --help
```

Available options:
- `--atm, -a PATH`: 1D atmospheric model file (default: G2_1D.dat)
- `--odf, -o PATH`: ODF data in NetCDF format (default: ODF_nc_format.nc)
- `--cont-abs PATH`: Continuum absorption opacity file (default: continuumabs.dat)
- `--cont-scat PATH`: Continuum scattering opacity file (default: continuumscat.dat)
- `--cont-all PATH`: Combined continuum opacity file (default: continuumall.dat)
- `--output, -O PATH`: Output file for binned opacities
- `--nbands, -n INTEGER`: Number of opacity bands (default: 2)
- `--scatter, -s FLOAT`: Continuum scattering contribution factor (default: 0.0)
- `--verbose, -v`: Verbose output

### Example Commands

```bash
# Use custom files
uv run python tausort.py -a my_atmosphere.dat -o my_odf.nc

# Set number of bands
uv run python tausort.py --nbands 4

# Include scattering contribution
uv run python tausort.py --scatter 1.0

# Specify output file
uv run python tausort.py -O output_opacities.dat
```

## Script Structure

### Data Classes

1. **AtmosphericData**: Contains 1D atmospheric structure
   - Height (z)
   - Density (rho)
   - Pressure (p)
   - Temperature (T)

2. **ODFData**: Opacity Distribution Function data from NetCDF
   - ODF array [nt, np, nbins, nsubbins]
   - Frequency grid (FreqG)
   - Temperature grid (T)
   - Pressure grid (P)
   - Sub-bin weights

3. **ContinuumData**: Continuum opacity data
   - Absorption opacity
   - Scattering opacity
   - Combined opacity

### Main Functions

1. `read_atmospheric_model()`: Reads G2_1D.dat ASCII file
2. `read_odf_netcdf()`: Reads ODF data from NetCDF format
3. `read_continuum_opacity()`: Reads continuum opacity ASCII files
4. `read_continuum_data()`: Coordinates reading of all continuum files
5. `verify_data_consistency()`: Validates loaded data

## Current Status

✅ **Completed:**
- Environment setup with uv
- Modular CLI with typer
- Input file reading and verification
- Data consistency checking
- Rich terminal output

🚧 **In Progress:**
- Grid initialization and interpolation
- Reference opacity calculations (Rosseland, 500nm)
- Tau-sorting algorithm
- Band-averaged opacity calculations
- Binary output file generation

## Data Files

The script expects the following input files:

- `G2_1D.dat`: 1D atmospheric model (height, density, pressure, temperature)
- `ODF_nc_format.nc`: ODF data in NetCDF format
- `continuumabs.dat`: Continuum absorption opacity
- `continuumscat.dat`: Continuum scattering opacity
- `continuumall.dat`: Combined continuum opacity

All files are successfully read and verified!
