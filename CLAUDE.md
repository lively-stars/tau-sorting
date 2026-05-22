# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tau-sorting is an opacity binning tool for stellar atmosphere radiative transfer. It reads opacity distribution functions (ODFs), continuum opacities, and a 1D atmospheric model, then sorts opacity sub-bins into tau-groups and calculates band-averaged (Planck/Rosseland mean) opacities. The Python implementation (`tausort.py`) is a port/evolution of the original C code (`tausort.c`).

Key physics reference: [Nordlund 1982 / Ludwig opacity binning](https://www.aanda.org/articles/aa/pdf/2004/26/aa0043.pdf) — equations 6, 11, 12. The threshold 0.35 switches between Planck and Rosseland mean (per `insturctions.md`).

## Commands

```bash
# Run the main tau-sorting tool
uv run python tausort.py
uv run python tausort.py --help          # see all CLI options
uv run python tausort.py --nbands 4      # set number of opacity bands

# Run tests
uv run python -m unittest test_kappa_band_reader.py
uv run python -m unittest test_plot_kap_mean_grid.py
uv run python test_derivatives.py        # quick script, not unittest-based

# Compile the C version (reference implementation)
make            # builds tausort.x
make clean

# Install dependencies
uv sync

# Lint + format (manual pre-commit; see "Pre-commit" section below)
./scripts/precommit.sh            # auto-fix + format the tree
./scripts/precommit.sh --check    # report-only, exit non-zero if changes needed
uv run ruff check .               # lint only
uv run ruff format .              # format only
```

## Pre-commit

This repo uses Jujutsu (`jj`) without a colocated git checkout, so the standard
`pre-commit` framework cannot install a real hook (it would target
`.git/hooks/`, which does not exist). Instead, run the checks manually before
`jj commit` / `jj describe`:

```bash
./scripts/precommit.sh            # ruff format + ruff check --fix + whitespace/EOF
```

The script lives at `scripts/precommit.sh` and mirrors what `.pre-commit-config.yaml`
would do (`ruff-check --fix`, `ruff-format`, trailing whitespace, end-of-file
fixer). The YAML file is kept for reference and for the day you decide to
colocate jj with git via `jj git init --colocate`; after that,
`uv run pre-commit install` will wire it up.

Ruff is configured in `pyproject.toml` under `[tool.ruff]` — line length 120,
target `py312`, rules `E,F,W,I,UP` with a small ignore list (`E501`, `E741`,
`UP007`).

A handy jj alias to run the script before describing/committing:

```toml
# in ~/.config/jj/config.toml
[aliases]
precommit = ["util", "exec", "--", "bash", "-c", "./scripts/precommit.sh"]
```

## Architecture

- **`tausort.py`** — Main entry point. Typer CLI app that orchestrates the full pipeline: read inputs, compute reference opacities (Rosseland mean), interpolate onto the atmospheric grid, sort sub-bins into tau-groups, compute band-averaged opacities, and write output. Contains dataclasses `AtmosphericData`, `ODFData`, `ContinuumData`.
- **`planck.py`** — Planck function B_lambda(T) and its temperature derivative (both numerical and analytic). Used for Rosseland/Planck mean weighting.
- **`group_derivatives.py`** — Analysis/visualization of how opacity group boundaries affect results. Reads grouped column files and computes derivatives via spline fits.
- **`kappa_band_reader.py`** — Reads the binary output file (`kappa_4_band_comparison.dat`) produced by tausort. Parses the layout header and reshapes the opacity data array.
- **`plot_kap_mean_grid.py`** — Generates grid plots of band-averaged mean opacities.
- **`convert_odf_to_npy.py`** — Converts the large `.dat` opacity files to `.npy` format for faster loading.
- **`tausort.c` / `global_tau.h`** — Original C reference implementation. The `diff_binning/` directory has alternative `global_tau.h` configs for different bin counts.

## Data Files (not in git, see .gitignore)

All `.dat`, `.npy`, `.nc` files are gitignored. The main inputs are:
- `G2_1D.dat` — 1D atmospheric model (height, density, pressure, temperature)
- `ODF_nc_format.nc` — ODF data: shape `[nt=300, np=150, nbins=328, nsubbins=12]`, stored as short integers (float = 10^(ODF/1000))
- `continuumabs.dat` / `continuumscat.dat` / `continuumall.dat` — Continuum opacities on the same (T, p, nbins) grid

## Key Conventions

- Uses `uv` for package management (Python 3.12+), not pip/conda.
- `tausort.py` has inline script metadata (`# /// script`) so it can also run standalone via `uv run --script`.
- ODF values are stored as short integers; convert to float via `10^(ODF/1000)`.
- Wavelength grids use `FreqG` (frequency edges) with 329 edges for 328 bins.
- Version control uses `jj` (Jujutsu), not git.
