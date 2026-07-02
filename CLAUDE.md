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
# split-lambda mode: shared tau binning + a per-tau-group 0/1 flag string (one digit per
# tau group) picking which groups subdivide into the lambda cells (mutually exclusive with
# --optimize-high-overlap). Here 8 tau groups, lambda cut at 3.8, split only groups 2-5:
uv run python tausort.py main --lambda-bin-edges 3 --lambda-bin-edges 3.8 --lambda-bin-edges 5 \
    --split-lambda 00111100

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

# Run the Q_rad explorer web app as a managed background server (uses ./.venv)
uv sync                 # create ./.venv + install deps (once)
make start              # start webapp/server.py detached -> http://localhost:8771
make status             # running?    (PID in .webapp.pid, logs in webapp.log)
make stop               # stop it
make restart            # stop + start

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

- **`tausort.py`** — Main entry point. Typer CLI app that orchestrates the full pipeline: read inputs, compute reference opacities (Rosseland mean), interpolate onto the atmospheric grid, sort sub-bins into tau-groups, split each tau-group into opacity segments, compute band-averaged opacities, and write outputs (see "Outputs" below). Contains dataclasses `AtmosphericData`, `ODFData`, `ContinuumData`. Key helpers: `assign_tau_to_bin` / `assign_split_lambda` / `assign_per_tau_lambda` (sub-bin → group, per-cell vs split-flag vs per-τ-group-λ modes), `build_group_specs_per_cell` / `build_group_specs_split_lambda` / `build_group_specs_per_tau` (per-group `(τ, λ)` descriptor that the downstream consumes; `_per_tau` gives each τ group its own λ edges), `sort_weighted_opacity_per_tau_bin` (per-group sorted κ·Δλ·B curves), `build_split_band_index` (group → low/mid/high splits → combined band index), `calculate_tau_bin_opacities` (Planck/Rosseland means + 0.35 transition merge).
- **`planck.py`** — Planck function B_lambda(T) and its temperature derivative (both numerical and analytic). Used for Rosseland/Planck mean weighting.
- **`group_derivatives.py`** — Analysis/visualization of how opacity group boundaries affect results. Reads grouped column files and computes derivatives via spline fits.
- **`kappa_band_reader.py`** — Reads/writes the C-format binary (`kappa_*.dat`). `read_kappa_4_band_comparison` parses the 8-int header and reshapes the opacity arrays into a `KappaBandComparison`; `write_kappa_4_band_comparison` is its exact inverse and is what `tausort.py main` calls to emit the parametrized `.dat` (via `build_kappa_band_comparison`).
- **`plot_kap_mean_grid.py`** — Generates grid plots of band-averaged mean opacities.
- **`convert_odf_to_npy.py`** — Converts the large `.dat` opacity files to `.npy` format for faster loading.
- **`rte.py`** — Standalone 1D radiative-transfer solver: `Solver` (short-characteristics formal solution, returns intensity/`J`/`F` and a blended heating rate `get_Q`), `compute_tau`, `ParallelSolver`. Only depends on numpy + multiprocessing.
- **`compare_Qrad_from_kappa.py`** — Q_rad validation/analysis (the *analyze* half, now in-repo). Reads kappa tables via `read_kappa_4_band_comparison`, interpolates onto a `models/` STAGGER atmosphere, solves the RTE with `rte.Solver`, and plots Q_rad vs the full-ODF reference (`Qrad_comparison.png`). Auto-discovers `kappa_*band_*.dat` in the repo root; `SELECT`/`LABELS` restrict the plotted cases. See README "Radiative-transfer Q_rad comparison".
- **`qrad_core.py`** — Shared Q_rad scoring core. Holds the edge-independent `precompute()`/`INV` (reads the ODF, interpolates opacity onto the atmosphere, Rosseland τ, τ at τ_λ=1), the per-star `reference_for_star()` (full-ODF/gray baselines + log₁₀τ axis, cached in `REF_CACHE`), `_qrad_from_table` (RTE solve → Q summed over bands), and `score_binning(tau_edges, lambda_edges, flags, star, *, lambda_edges_per_tau=None) -> dict` (the full binning→rms pipeline: assign → sort → split → band-average → RTE → residual). Passing `lambda_edges_per_tau` (one λ-edge list per τ group) switches to the **per-τ-group λ** grouping (`tausort.build_group_specs_per_tau` / `assign_per_tau_lambda`) where each τ group has its own wavelength split; otherwise the shared-λ + flags path is used (byte-identical to before). Both `webapp/server.py` and `qrad_optimize.py` import it; the webapp's `compute()` is now a thin display wrapper over `score_binning`.
- **`qrad_optimize.py`** — Q_rad-driven binning optimizer + typer CLI. Unlike `tausort`'s `optimize_tau_bin_edges` (maximizes the high-overlap *proxy*), this **minimizes the Q_rad rms residual directly** via `qrad_core.score_binning`. Block-coordinate search: coordinate-descend τ edges (`_coord_descent` or Nelder-Mead `_nelder_mead_edges` over a softmax-simplex reparameterization), greedy split-flag flips (`_flag_search`), coordinate-descend λ edges — to a fixed point (`_block_fixed_point`) — then grow the τ-group count (`_insert_edge`, accepted only if rms improves past `grow_tol`). Guardrails: `_valid_monotone` + per-axis min-gap + a multiplicative empty-band penalty (`make_evaluator`). `optimize_qrad(...)` is the public API (with `on_eval`/`on_progress`/`should_stop` hooks for live progress + graceful abort); `score_fn` is dependency-injected so `test_qrad_optimize.py` tests the search logic with no ODF. Run-and-wait (~2.5 s/eval), bounded by `--max-evals`/`--max-seconds`. The webapp exposes it as a background job (`POST /api/optimize_qrad` + `/api/optimize_qrad_status` polling + `/api/optimize_qrad_cancel`), surfaced as the "Optimize for Q_rad" button. With `--per-group-lambda` / the "per-group λ" checkbox it runs the per-τ-group variant (`_per_group_lambda_search` + `_block_fixed_point_pg` + `_insert_edge_pg`): each τ group independently keeps the better of *no split* or a *single λ cut* (position coordinate-descended), returning `lambda_edges_per_tau`; the binning diagram then shows the λ cuts jumping per τ band.
- **`models/`** — STAGGER 1D model atmospheres (`F/G/K/M_SSD`, binary float32) used by `compare_Qrad_from_kappa.py`, `qrad_core.py`, and the webapp.
- **`tausort.c` / `global_tau.h`** — Original C reference implementation. The `diff_binning/` directory has alternative `global_tau.h` configs for different bin counts.

## Data Files (not in git, see .gitignore)

All `.dat`, `.npy`, `.nc` files are gitignored. The main inputs are:
- `G2_1D.dat` — 1D atmospheric model (height, density, pressure, temperature)
- `ODF_nc_format.nc` — ODF data: shape `[nt=300, np=150, nbins=328, nsubbins=12]`, stored as short integers (float = 10^(ODF/1000))
- `continuumabs.dat` / `continuumscat.dat` / `continuumall.dat` — Continuum opacities on the same (T, p, nbins) grid

## Outputs (`tausort.py main`)

`main` places each wavelength sub-bin into a **group**, then splits each group into `nSplits = 3`
opacity segments (low/mid/high, via `analyze_group` → `build_split_band_index`). The final band
count is `nBands = nSplits * nGroups`, and the band axis factorizes as
`band -> (group = band // nSplits, split = band % nSplits)`. Every mode produces a per-group
**descriptor** — `group_tau_edges[nGroups, 2]` (-log10 τ lo/hi) and `group_lam_edges[nGroups, 2]`
(log10 λ/Å lo/hi) — which the shared downstream (`sort_weighted_opacity_per_tau_bin`,
`compute_bot_segment_overlap_per_tau_bin`, both plots, the saver) consumes. There are three
grouping modes:

- **Default / single lambda cell** (`--lambda-bin-edges 3 5`): `group = tau index`, output
  identical to the pre-lambda tool (byte-identical `.dat`).
- **Per-cell (uniform / optimized)**: with ≥3 lambda edges each lambda cell carves tau
  *independently* (`tau_edges_per_lambda[cell]`), so tau-group boundaries can "jump" across the
  fixed lambda lines. `--optimize-high-overlap` runs the greedy optimizer separately per lambda
  cell (cells may reach different tau-group counts). `build_group_specs_per_cell` builds the
  descriptor cell-major.
- **Split-flag (`--split-lambda`)**: a **shared** tau binning (`--tau-bin-edges`, same tau ranges
  at all wavelengths) plus a per-tau-group 0/1 flag string; flagged groups subdivide into the
  lambda cells, unflagged groups stay one band spanning all lambda
  (`nBands = nSplits · Σ_k (split[k] ? nLambda : 1)`). `build_group_specs_split_lambda` /
  `assign_split_lambda`. Mutually exclusive with `--optimize-high-overlap`; omit the flag to get
  the per-cell uniform behavior.

Two files are written (both gitignored):

- **`tau_bin_opacities.npy`** (`save_tau_bin_opacities_npy`) — structured array with `planck`,
  `rosseland`, `mixed` each `[nT, nP, nBands]` (linear; `mixed` is the `2^(-τ/0.35)` Planck↔Rosseland
  blend), plus `T`/`p` (linear), `members_per_band`, `n_splits`, the authoritative
  `group_tau_edges` / `group_lam_edges` descriptor, `lambda_bin_edges`, `split_along_lambda`
  (`int8[N]`, empty unless split-flag mode), and (per-cell mode) the ragged `tau_bin_edges` /
  `n_tau_per_lambda` / `tau_edges_concat`. Empty bands are NaN.
- **`.dat`** (C-format binary, `build_kappa_band_comparison` + `write_kappa_4_band_comparison`),
  named per mode: `kappa_<nBands>band_tg<..>_sp<..>_tau_<edges>_lam_<edges>.dat` (single cell),
  `kappa_<nBands>band_lm<L>_tg<g0-g1-..>_sp<..>_lam_<edges>.dat` (per-cell multi), or
  `kappa_<nBands>band_lm<L>_sl<10..>_sp<..>_tau_<edges>_lam_<edges>.dat` (split-flag; `sl` is the
  flags as a 1/0 string). Matches `tausort.c output()`: `kap_mean = ln(mixed)` as `[nBands, NT, Np]`,
  `B_band = ln(...)`, axes `tab_T/tab_p = log10(T)/log10(p)` (= `odf.T`/`odf.P`). Round-trips through
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
