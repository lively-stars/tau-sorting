# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tau-sorting is an opacity binning tool for stellar atmosphere radiative transfer. It reads opacity distribution functions (ODFs), continuum opacities, and a 1D atmospheric model, then sorts opacity sub-bins into tau-groups and calculates band-averaged (Planck/Rosseland mean) opacities. The Python implementation (`tausort.py`) is a port/evolution of the original C code (`tausort.c`).

Key physics reference: **Vögler, Bruls & Schüssler (2004), A&A 421, 741–754** — the opacity-binning method (building on Nordlund 1982; Ludwig), DOI [10.1051/0004-6361:20047043](https://doi.org/10.1051/0004-6361:20047043). Full HTML: [aa0043.right.html](https://www.aanda.org/articles/aa/full/2004/26/aa0043/aa0043.right.html); PDF: [aa0043.pdf](https://www.aanda.org/articles/aa/pdf/2004/26/aa0043.pdf). Equations 6, 11, 12 (Planck↔Rosseland blend), 16, 17 are implemented here. The threshold 0.35 (τ₀ in Eq. 12) switches between Planck and Rosseland mean (per `insturctions.md`). The local copy of the HTML full text is `papers/voegler_bruls_schuessler_2004_aa421_741_full.html`; note `papers/Simulations-of-magneto-convection-in-the-solar-photosphere.pdf` is the companion MURaM code paper (Vögler et al. 2005, A&A 429, 335), not this one.

## Commands

```bash
# Run the main tau-sorting tool (`main` subcommand is required)
uv run python tausort.py main
uv run python tausort.py main --help     # see all CLI options
# set tau-group edges (repeat the flag per edge; use = so negatives aren't parsed as flags):
uv run python tausort.py main --tau-bin-edges=-0.63 --tau-bin-edges=7.0
# split-lambda mode: shared tau binning + a per-tau-group 0/1 flag string (one digit per
# tau group) picking which groups subdivide into the lambda cells. Here 8 tau groups,
# lambda cut at 3.8, split only groups 2-5. (All three input modes seed the same
# guillotine-tree IR; see "Outputs" below.)
uv run python tausort.py main --lambda-bin-edges 3 --lambda-bin-edges 3.8 --lambda-bin-edges 5 \
    --split-lambda 00111100
# Q_rad-driven binning optimizer: minimize the rms residual directly. Beam search is the
# default (beam_width>=2); --beam-width 1 is the greedy grow (one midpoint split per round).
# The former --method cd|nm and its separate greedy optimizer paths are gone -- beam_width is
# the only structural knob; position refinement is always coordinate-descent.
uv run python qrad_optimize.py --model G2_1D.dat --max-seconds 600 --save-plot opt.png
# general-2D guillotine tree (--tree), writes the optimized kappa table:
uv run python qrad_optimize.py --tree --beam-width 3 --save-dat
# render a tau-lambda binning plot for every improved binning the search finds:
uv run python qrad_optimize.py --tree --beam-width 3 --max-seconds 600 --plot plots

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

# Convert inputs to the fast .npy caches that `main` reads (positional INPUT [OUTPUT])
uv run python tausort.py convert-odf ODF_nc_format.nc ODF_format.npy        # (or convert_odf_to_npy.py -i/-o)
uv run python tausort.py convert-continuum continuumabs.dat continuumabs.npy

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

- **`tausort.py`** — Main entry point. Typer CLI app that orchestrates the full pipeline: read inputs, compute reference opacities (Rosseland mean), interpolate onto the atmospheric grid, assign sub-bins to groups, split each group into opacity segments, compute band-averaged opacities, and write outputs (see "Outputs" below). Contains dataclasses `AtmosphericData`, `ODFData`, `ContinuumData`. The single grouping IR is a **guillotine tree** that recursively cuts the (−log10 τ, log10 λ) plane along `tau` or `lam`; its leaves are the `(τ, λ)` groups. `_iter_leaf_rects` enumerates leaves in τ-major DFS pre-order; `build_group_specs_tree` builds the per-group descriptor (`group_tau_edges` / `group_lam_edges`) every downstream stage consumes; `assign_tree` vectorizes sub-bin → leaf membership; `_resolve_grouping_inputs` turns the CLI options (`--split-lambda` / `--lambda-per-tau` / uniform) into the per-tau-group λ-edge list (`lpt`) that `qrad_optimize.tree_from_lpt` lifts to a tree (τ cuts at the root, a λ-chain per band). `main()` routes **every** mode through this tree — the 7 legacy grouping helpers (`assign_tau_to_bin` / `assign_split_lambda` / `assign_per_tau_lambda` + `build_group_specs_per_cell` / `_split_lambda` / `_per_tau`) and the `optimize_tau_bin_edges` high-overlap proxy are **deleted**. Also: `sort_weighted_opacity_per_tau_bin` (per-group sorted κ·Δλ·B curves), `build_split_band_index` (group → low/mid/high splits → combined band index), `calculate_tau_bin_opacities` (Planck/Rosseland means + 0.35 transition merge).
- **`planck.py`** — Planck function B_lambda(T) and its temperature derivative (both numerical and analytic). Used for Rosseland/Planck mean weighting.
- **`group_derivatives.py`** — Analysis/visualization of how opacity group boundaries affect results. Reads grouped column files and computes derivatives via spline fits.
- **`kappa_band_reader.py`** — Reads/writes the C-format binary (`kappa_*.dat`). `read_kappa_4_band_comparison` parses the 8-int header and reshapes the opacity arrays into a `KappaBandComparison`; `write_kappa_4_band_comparison` is its exact inverse and is what `tausort.py main` calls to emit the parametrized `.dat` (via `build_kappa_band_comparison`).
- **`plot_kap_mean_grid.py`** — Generates grid plots of band-averaged mean opacities.
- **`convert_odf_to_npy.py`** — Converts the large `.dat` opacity files to `.npy` format for faster loading.
- **`rte.py`** — Standalone 1D radiative-transfer solver: `Solver` (short-characteristics formal solution, returns intensity/`J`/`F` and a blended heating rate `get_Q`), `compute_tau`, `ParallelSolver`. Only depends on numpy + multiprocessing.
- **`compare_Qrad_from_kappa.py`** — Q_rad validation/analysis (the *analyze* half, now in-repo). Reads kappa tables via `read_kappa_4_band_comparison`, interpolates onto `models/G2_1D.dat`, solves the RTE with `rte.Solver`, and plots Q_rad vs the full-ODF reference (`Qrad_comparison.png`). Auto-discovers `kappa_*band_*.dat` in the repo root; `SELECT`/`LABELS` restrict the plotted cases. See README "Radiative-transfer Q_rad comparison".
- **`qrad_core.py`** — Shared Q_rad scoring core. The atmosphere is **selectable**: any validated 1D model under `models/` (default `G2_1D.dat`). `validate_model_file(path)` checks a file is a usable model — ASCII, 4 columns (z, ρ, p, T), ≥2 rows, and a **strictly decreasing** height column z (top first) — returning `{name, ok, error, n_rows, n_cols}` with a one-line reason on failure; `scan_models()` runs it over every file in `models/` and `valid_model_names()` returns the passing ones. Per-model **caches** (keyed by bare filename): `precompute(model)`/`_INV_CACHE` (reads the ODF, interpolates opacity onto that atmosphere, Rosseland τ, τ at τ_λ=1; the RTE gets it depth-from-top so `compute_tau` sees z increasing inward) and `reference(model)`/`_REF_CACHE` (full-ODF/gray baselines + log₁₀τ axis on that atmosphere; also reads an **optional** `data/kappa_goldenS.dat` "golden standard" into `q_golden` if present); `inv_for(model)` lazily precomputes on first use. Fills are guarded by `_CACHE_LOCK` (thread-safe). `_qrad_from_table(inv, …)` does the RTE solve → Q summed over bands. `score_binning(tau_edges, lambda_edges, flags, model=None, *, lambda_edges_per_tau=None, binning_tree=None, window=None, min_opacity_delta=1.0) -> dict` (`model` = which atmosphere; None → `DEFAULT_MODEL`) is the full binning→rms pipeline (assign → sort → split → band-average → RTE → residual). It is **tree-first**: precedence `binning_tree` (general-2D, any rectangular tiling) > `lambda_edges_per_tau` (per-τ-group λ) > shared-λ + `flags`; each is normalized to a guillotine tree via `qrad_optimize.tree_from_lpt` and consumed by `tausort.assign_tree` / `build_group_specs_tree` (the flat seed shims are retained for the manual-binning + compute/download surface). Both `webapp/server.py` and `qrad_optimize.py` import it; the webapp's `compute()` is a thin display wrapper over `score_binning`. `save_kappa_dat(..., model=None, *, binning_tree=None, lambda_edges_per_tau=None, ...)` runs the same pipeline but keeps the full opacity result, NaN-masks empty bands, and writes the C-format kappa `.dat` (via `tausort.build_kappa_band_comparison` / `write_kappa_4_band_comparison`) with a self-describing name.
- **`qrad_optimize.py`** — Q_rad-driven binning optimizer + Typer CLI. It **minimizes the Q_rad rms residual directly** via `qrad_core.score_binning` and is **tree-only**: every input (`binning_tree`, `lambda_edges_per_tau`, `per_group_lambda`, or shared-tau + `flags`) is normalized to a guillotine tree by `tree_from_lpt` and refined via one seed → grow → polish path. **Grow** is a non-greedy **beam search** by default (`_beam_grow_tree`, `beam_width >= 2` (default 3): keeps rival topologies in parallel, tries several split positions per leaf, dedupes by leaf-rectangle signature `_tree_signature`), or greedy (`_grow_tree`, `beam_width == 1`: one midpoint split per round, committed immediately); **polish** sweeps every cut's position (`_block_fixed_point_tree` / `_tree_position_search` / `_refine_node`). The `flags` / `per_group_lambda` / `lambda_edges_per_tau` arguments survive only as **seed shims** (they no longer drive separate optimizer modes), and `opt_tau`/`opt_lambda`/`opt_flags` are retained for API compatibility but are inert on the tree path. Guardrails: `_valid_monotone` + per-axis min-gap + a multiplicative empty-band penalty (`make_evaluator`). `optimize_qrad(...)` is the public API (with `on_eval`/`on_progress`/`on_improve`/`should_stop` hooks for live progress, per-improvement tiling snapshots, and graceful abort); `score_fn` is dependency-injected so `test_qrad_optimize.py` tests the search logic with no ODF. Run-and-wait (~2.5 s/eval), bounded by `--max-evals`/`--max-seconds`; `--model <file>` picks which validated atmosphere under `models/` to optimize on (default `G2_1D.dat`). CLI grow/topology knobs: `--beam-width` (1 = greedy, >= 2 = beam, default 3), `--tree/--no-tree` (general-2D guillotine), `--beam-leaves`/`--beam-positions`, `--max-groups`, `--plot DIR` (render the tau-lambda tiling of every improved binning found during the search). The webapp exposes it as a background job (`POST /api/optimize_qrad` + `/api/optimize_qrad_status` polling + `/api/optimize_qrad_cancel`), surfaced as the "Optimize for Q_rad" button (it runs on the dropdown's selected model; the result is always a general-2D guillotine tree). The backend has `GET /api/models` (the Refresh button's re-scan) and includes the model list in `GET /api/init`; every compute/optimize/download request carries the chosen `model` and is rejected (400) if it fails validation. `--save-dat` writes the optimized binning's kappa table (via `qrad_core.save_kappa_dat`); the webapp exposes the same as a **download** (`POST /api/kappa_dat` streams the current binning's `.dat`, "Download kappa table" button).
- **`models/`** — holds the 1D model atmospheres (ASCII: `z[cm] ρ p T`, top-of-atmosphere first / z descending). `models/G2_1D.dat` is the tracked default and ships with the repo; drop additional models here and they become selectable in the webapp once they pass validation (`qrad_core.validate_model_file`: 4 columns, ≥2 rows, strictly decreasing z). Any validated model is the atmosphere the binning **and** the RTE run on (`score_binning(..., model=...)`); the webapp's **Model atmosphere** dropdown + **Refresh** button list them and report what's wrong with any file that fails. `models/G2_1D.dat` is the `--atm` default for `tausort.py main`; `compare_Qrad_from_kappa.py` runs on it. (The former per-star STAGGER models were removed.)
- **`tausort.c` / `global_tau.h`** — Original C reference implementation. The `diff_binning/` directory has alternative `global_tau.h` configs for different bin counts.

## Data Files (mostly not in git, see .gitignore)

`.dat`/`.npy`/`.nc` are gitignored **except** `models/G2_1D.dat` (un-ignored via `!models/*.dat`, so it ships with the repo). The main inputs:
- `models/G2_1D.dat` — the tracked default 1D atmospheric model (ASCII: height/density/pressure/temperature). Additional models dropped in `models/` are auto-detected + validated (selectable in the webapp).
- `ODF_nc_format.nc` — ODF data: shape `[nt=300, np=150, nbins=328, nsubbins=12]`, stored as short integers (float = 10^(ODF/1000))
- `continuumabs.dat` / `continuumscat.dat` / `continuumall.dat` — Continuum opacities on the same (T, p, nbins) grid

## Outputs (`tausort.py main`)

`main` resolves its CLI options (`--tau-bin-edges` / `--lambda-bin-edges` / `--split-lambda` /
`--lambda-per-tau`) into a **guillotine tree** (via `_resolve_grouping_inputs` →
`qrad_optimize.tree_from_lpt`), assigns each wavelength sub-bin to a **tree leaf** — a `(τ, λ)`
**group** (`assign_tree`) — then splits each group into `nSplits = 3` opacity segments
(low/mid/high, via `analyze_group` → `build_split_band_index`). The final band count is
`nBands = nSplits * nGroups`; the band axis factorizes as `band -> (group = band // nSplits,
split = band % nSplits)` and the leaves are numbered in τ-major DFS pre-order (τ cuts at the
root, a λ-chain per τ band) — the canonical band order for **every** mode. Every mode produces
the same per-group **descriptor** — `group_tau_edges[nGroups, 2]` (-log10 τ lo/hi) and
`group_lam_edges[nGroups, 2]` (log10 λ/Å lo/hi) — which the shared downstream
(`sort_weighted_opacity_per_tau_bin`, `compute_bot_segment_overlap_per_tau_bin`, both plots,
the saver) consumes. The three input modes all seed this one tree:

- **Uniform** (`--lambda-bin-edges 3 5`, default): one λ window, `group == τ index`.
- **Split-flag (`--split-lambda 00111100`)**: shared τ binning + a per-τ-group 0/1 flag string;
  flagged groups subdivide into the λ cells, the rest span the whole window
  (`nBands = nSplits · Σ_k (split[k] ? nLambda : 1)`).
- **Per-tau-group λ (`--lambda-per-tau`)**: shared τ binning where each τ group carries its
  **own** λ edges (repeat the flag once per τ group; 2 edges = that group unsplit); all groups
  share the outer `[min,max]` window, the interior cut(s) vary. (This is the seed shape
  `qrad_optimize --per-group-lambda` produces.)

`--split-lambda` and `--lambda-per-tau` are mutually exclusive.

Two files are written (both gitignored):

- **`tau_bin_opacities.npy`** (`save_tau_bin_opacities_npy`) — structured array with `planck`,
  `rosseland`, `mixed` each `[nT, nP, nBands]` (linear; `mixed` is the `2^(-τ/0.35)` Planck↔Rosseland
  blend), plus `T`/`p` (linear), `members_per_band`, `n_splits`, the authoritative
  `group_tau_edges` / `group_lam_edges` descriptor, `lambda_bin_edges`, `split_along_lambda`
  (`int8[N]`, empty unless split-flag mode), and (uniform mode with ≥2 λ cells) the ragged
  `tau_bin_edges` / `n_tau_per_lambda` / `tau_edges_concat`. Empty bands are NaN.
- **`.dat`** (C-format binary, `build_kappa_band_comparison` + `write_kappa_4_band_comparison`),
  named per mode (the variants are not yet unified): `kappa_<nBands>band_tg<..>_sp<..>_tau_<edges>_lam_<edges>.dat`
  (single λ cell), `kappa_<nBands>band_lm<L>_tg<g0-g1-..>_sp<..>_lam_<edges>.dat` (uniform, ≥2 λ
  cells), `kappa_<nBands>band_lm<L>_sl<10..>_sp<..>_tau_<edges>_lam_<edges>.dat` (split-flag; `sl`
  is the flags as a 1/0 string), or `kappa_<nBands>band_pt_sp<..>_tau_<edges>_lam_<min>_<max>_cuts_<g0>-<g1>-...dat`
  (per-tau-group λ; `x` = unsplit). The optimizer/download path, which writes via an explicit
  `binning_tree`, emits `kappa_<nBands>band_tree<Nleaves>_sp<..>_<short-hash>.dat` (leaf count + a
  blake2s hash of the canonical tree; the full tree lives in the `.npy`). Matches `tausort.c output()`:
  `kap_mean = ln(mixed)` as `[nBands, NT, Np]`, `B_band = ln(...)`, axes `tab_T/tab_p = log10(T)/log10(p)`
  (= `odf.T`/`odf.P`). Round-trips through `read_kappa_4_band_comparison`; its `kap_mean` equals
  `ln(mixed)` from the `.npy`.

`nSplits` is fixed at 3 for the saved table regardless of `--refine-mid` (off by default; the flag
only affects the diagnostic overlap table and the `sorted_weighted_opacity_per_tau_bin.jpg` plot).
`--min-opacity-delta` (default 1.0) gates the low/mid/high split: a group whose bottom
weighted-opacity dynamic range (max/min) is below it stays a single band. The tree optimizer's
leaf-count cap is `--max-groups` in `qrad_optimize.py` (default 8).

## Key Conventions

- Uses `uv` for package management (Python 3.12+), not pip/conda.
- `tausort.py` has inline script metadata (`# /// script`) so it can also run standalone via `uv run --script`.
- ODF values are stored as short integers; convert to float via `10^(ODF/1000)`.
- Wavelength grids use `FreqG` (frequency edges) with 329 edges for 328 bins.
- Saved opacity bands factorize as `band = group * nSplits + split` (`nSplits = 3`: low/mid/high), where a `group` is a **guillotine-tree leaf** in τ-major DFS pre-order (τ cuts at the root, a λ-chain per τ band) — the canonical band order for **every** mode (the former uniform-per-cell layout is relabeled into this τ-major order). The `.npy` stores **linear** opacity with linear `T`/`p`; the `.dat` stores **natural-log** opacity (`kap_mean = ln(mixed)`) with `log10(T)`/`log10(p)` axes and a leading band axis `[nBands, NT, Np]`. The authoritative grouping is the per-group `group_tau_edges` / `group_lam_edges` descriptor; `tau_edges_per_lambda` / `split_along_lambda` in the `.npy` only record which CLI mode produced the tree.
- Version control uses `jj` (Jujutsu), not git.
- **Commit messages carry no AI authorship attribution** — never add `Co-Authored-By:` lines or "Generated with …" footers (the Claude Code / Codex boilerplate). Plain commit messages only.
