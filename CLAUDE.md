# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tau-sorting is an opacity binning tool for stellar atmosphere radiative transfer. It reads opacity distribution functions (ODFs), continuum opacities, and a 1D atmospheric model, then sorts opacity sub-bins into tau-groups and calculates band-averaged (Planck/Rosseland mean) opacities. The Python implementation (`tausort.py`) is a port/evolution of the original C code (`tausort.c`).

Key physics reference: [Nordlund 1982 / Ludwig opacity binning](https://www.aanda.org/articles/aa/pdf/2004/26/aa0043.pdf) — equations 6, 11, 12. The threshold 0.35 switches between Planck and Rosseland mean (per `insturctions.md`).

## Commands

```bash
# Run the main tau-sorting tool (`main` subcommand is required)
uv run python tausort.py main
uv run python tausort.py main --help     # see all CLI options
# set tau-group edges (repeat the flag per edge; use = so negatives aren't parsed as flags):
uv run python tausort.py main --tau-bin-edges=-0.63 --tau-bin-edges=7.0
# find tau-group edges whose high-segment overlap clears a threshold, then stop (prints only):
uv run python tausort.py main --optimize-high-overlap
# optimizer sweep that also saves the table: grow from coarse edges to N tau-groups
# (threshold > 1 is unreachable, so the optimizer always grows to the --max-bins cap)
uv run python tausort.py main --optimize-high-overlap --save-after-optimize --max-bins 4 \
    --high-overlap-threshold 1.01 --tau-bin-edges=-0.63 --tau-bin-edges=7.0

# Run tests
uv run python -m unittest test_kappa_band_reader.py
uv run python -m unittest test_plot_kap_mean_grid.py
uv run python -m unittest test_build_split_band_index.py   # split-resolved band index
uv run python -m unittest test_kappa_dat_export.py         # parametrized .dat export
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

- **`tausort.py`** — Main entry point. Typer CLI app that orchestrates the full pipeline: read inputs, compute reference opacities (Rosseland mean), interpolate onto the atmospheric grid, sort sub-bins into tau-groups, split each tau-group into opacity segments, compute band-averaged opacities, and write outputs (see "Outputs" below). Contains dataclasses `AtmosphericData`, `ODFData`, `ContinuumData`. Key helpers: `assign_tau_to_bin` (sub-bin → tau-group), `sort_weighted_opacity_per_tau_bin` (per-group sorted κ·Δλ·B curves), `build_split_band_index` (tau-group → low/mid/high splits → combined band index), `calculate_tau_bin_opacities` (Planck/Rosseland means + 0.35 transition merge).
- **`planck.py`** — Planck function B_lambda(T) and its temperature derivative (both numerical and analytic). Used for Rosseland/Planck mean weighting.
- **`group_derivatives.py`** — Analysis/visualization of how opacity group boundaries affect results. Reads grouped column files and computes derivatives via spline fits.
- **`kappa_band_reader.py`** — Reads/writes the C-format binary (`kappa_*.dat`). `read_kappa_4_band_comparison` parses the 8-int header and reshapes the opacity arrays into a `KappaBandComparison`; `write_kappa_4_band_comparison` is its exact inverse and is what `tausort.py main` calls to emit the parametrized `.dat` (via `build_kappa_band_comparison`).
- **`plot_kap_mean_grid.py`** — Generates grid plots of band-averaged mean opacities.
- **`convert_odf_to_npy.py`** — Converts the large `.dat` opacity files to `.npy` format for faster loading.
- **`tausort.c` / `global_tau.h`** — Original C reference implementation. The `diff_binning/` directory has alternative `global_tau.h` configs for different bin counts.

## Data Files (not in git, see .gitignore)

All `.dat`, `.npy`, `.nc` files are gitignored. The main inputs are:
- `G2_1D.dat` — 1D atmospheric model (height, density, pressure, temperature)
- `ODF_nc_format.nc` — ODF data: shape `[nt=300, np=150, nbins=328, nsubbins=12]`, stored as short integers (float = 10^(ODF/1000))
- `continuumabs.dat` / `continuumscat.dat` / `continuumall.dat` — Continuum opacities on the same (T, p, nbins) grid

## Outputs (`tausort.py main`)

`main` places each wavelength sub-bin into a **group = (lambda cell, tau index)**, then splits each
group into `nSplits = 3` opacity segments (low/mid/high, via `analyze_group` →
`build_split_band_index`). A group is `g = build_group_index_maps` flattening of
`(lambda cell, tau index)`; the final band count is `nBands = nSplits * nGroups` where
`nGroups = Σ_cell nTau[cell]`. The band axis factorizes as
`band -> (group = band // nSplits, split = band % nSplits)`.

Lambda binning (`--lambda-bin-edges`, log10 Å) is a **real band dimension**: with the default
`[3.0, 5.0]` there is one lambda cell (`nGroups = nTauGroups`, output identical to before). With ≥3
edges, each lambda cell carves tau **independently** — its own `tau_edges_per_lambda[cell]` list, so
tau-group boundaries "jump" across the fixed lambda lines (visible in
`tau_rosseland_at_tau_lambda_one.jpg`). `--optimize-high-overlap` runs the greedy optimizer
separately per lambda cell, so cells can reach different tau-group counts. Two files are written
(both gitignored):

- **`tau_bin_opacities.npy`** (`save_tau_bin_opacities_npy`) — structured array with `planck`,
  `rosseland`, `mixed` each `[nT, nP, nBands]` (linear; `mixed` is the `2^(-τ/0.35)` Planck↔Rosseland
  blend), plus `T`/`p` (linear), `members_per_band`, `n_splits`, `tau_bin_edges` (cell-0 edges,
  single-cell convenience), and the ragged factorization `lambda_bin_edges` / `n_tau_per_lambda` /
  `tau_edges_concat` (split per cell with `cumsum(n_tau_per_lambda + 1)`). Empty bands are NaN.
- **`kappa_<nBands>band_tg<..>_sp<..>_tau_<edges>_lam_<edges>.dat`** (single lambda cell) or
  **`kappa_<nBands>band_lm<L>_tg<g0-g1-..>_sp<..>_lam_<edges>.dat`** (lambda-split; per-cell tau
  edges are ragged so only counts go in the name, full edges live in the `.npy`) — C-format binary
  (`build_kappa_band_comparison` + `write_kappa_4_band_comparison`). Matches `tausort.c output()`:
  `kap_mean = ln(mixed)` as `[nBands, NT, Np]`, `B_band = ln(...)`, axes
  `tab_T/tab_p = log10(T)/log10(p)` (= `odf.T`/`odf.P`). Round-trips through
  `read_kappa_4_band_comparison`; its `kap_mean` equals `ln(mixed)` from the `.npy`.

`nSplits` is fixed at 3 for the saved table regardless of `--refine-mid` (off by default; the flag
only affects the diagnostic overlap table and the `sorted_weighted_opacity_per_tau_bin.jpg` plot).
With `--optimize-high-overlap`, `main` prints optimized edges and returns early — **no files are
saved** — unless `--save-after-optimize` is also set, in which case the pipeline continues with the
optimized edges and saves both outputs (the `.dat` filename then encodes the optimized edges).
`--max-bins` caps how many tau-groups the optimizer may grow to (default 8).

## Key Conventions

- Uses `uv` for package management (Python 3.12+), not pip/conda.
- `tausort.py` has inline script metadata (`# /// script`) so it can also run standalone via `uv run --script`.
- ODF values are stored as short integers; convert to float via `10^(ODF/1000)`.
- Wavelength grids use `FreqG` (frequency edges) with 329 edges for 328 bins.
- Saved opacity bands factorize as `band = group * nSplits + split` (`nSplits = 3`: low/mid/high), where a `group` is a `(lambda cell, tau index)` pair (`nGroups = Σ_cell nTau[cell]`; with one lambda cell `group == tau index`). The `.npy` stores **linear** opacity with linear `T`/`p`; the `.dat` stores **natural-log** opacity (`kap_mean = ln(mixed)`) with `log10(T)`/`log10(p)` axes and a leading band axis `[nBands, NT, Np]`. Lambda edges are fixed; tau edges are per-lambda-cell (`tau_edges_per_lambda`) so the optimizer can grow each cell independently.
- Version control uses `jj` (Jujutsu), not git.
