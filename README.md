# Readme

This repository contains the source code and documentation for tau-sorting.

## Data files

```
├── G2_1D.dat                  - 1D atmospheric model data (height - ascending, density, pressure, temperature)
├── Makefile                   - Makefile to compile the c version tau-sorting code
├── ODF_nc_format.nc           - ODF data in netCDF format (p, T, n_bins, n_subbins)
├── continuumabs.dat           - Continuum absorption data (p, T, n_bins)
├── continuumscat.dat          - Continuum scattering data
├── continuumall.dat           - Continuum absorption + scattering data
├── diff_binning               - ignored directory
│   ├── global_tau.h_12bins
│   ├── global_tau.h_15bin
│   ├── global_tau.h_2bins
│   ├── global_tau.h_4bins
│   └── global_tau.h_grey
├── global_tau.h               - Header file with global variables for tau-sorting
├── p00big3.bdf                - Line absorption data file
└── tausort.c                  - C source code for tau-sorting
```

```
❯ ncdump -h ODF_nc_format.nc
netcdf ODF_nc_format {
dimensions:
	np = 150 ;                            - number of pressure points
	nt = 300 ;                            - number of temperature points
	nbins = 328 ;                         - number of lambda bins
	nsubbins = 12 ;                       - number of sub-bins per lambda bin
	numfp = 329 ;                         - number of lambda edges (lambda!!)
variables:
	short ODF(nt, np, nbins, nsubbins) ;  - ODF values as short integers - float value = 10^(ODF/1000)
	double FreqG(numfp) ;
	double P(np) ;
	double T(nt) ;
	double subbin(nbins, nsubbins) ;

// global attributes:
		:vturb = 2. ;
}
```

## Python implementation

### Overview

1. Inputs:
    - ODF_nc_format.nc - kappa (T, p, N_b, N_s)
    - continuumall.dat - continuum opacity (T, p, N_b)
    - G2_1D.dat - atmospheric model (height, rho, p, T)
2. Calculate reference kappa (rosseland, 500nm...) as
    $$ \kappa_\text{all}(T,p) = f(\kappa_{ODF} + \kappa_{cont}) $$
    $$ \kappa_\text{ross} = \frac{\integrate_0^\inf\kappa_\text{all} \frac{dB_\lambda}{dT} d\lambda}{\integrate_0^\inf \frac{dB_\lambda}{dT} d\lambda} $$
3. Interpolate kappa calcualted on the T,p grid from ODFs to the T,p grid of the atmospheric model

## Tau-bin edge optimization and segmentation flags

`tausort.py main` exposes a small number of new CLI flags that control how
the tau-bin edges are chosen and how the per-bin sorted-opacity curves are
segmented.

### `--optimize-high-overlap` / `--high-overlap-threshold`

```
--optimize-high-overlap         (default: off)
--high-overlap-threshold FLOAT  (default: 0.70)
```

When `--optimize-high-overlap` is passed, the tool runs a greedy optimizer
over `--tau-bin-edges` instead of producing the usual per-tau-bin plot.

The optimizer:

1. Computes the *high segment* (large-tau tail of the sorted-opacity curve)
   overlap for every bin defined by the current edge list.
2. While any bin has a high-segment overlap below `--high-overlap-threshold`,
   the optimizer greedily adjusts an existing edge — or inserts a new one —
   to lift the worst-offending bin above the threshold.
3. Iteration stops once every bin clears the threshold (or the cap of
   8 bins is reached). The final edge list and per-bin overlap table are
   printed and the tool exits **before** the sorted-opacity plot is written.

Use the optimizer to *find* good edges, then re-run with the printed
`--tau-bin-edges ...` (no `--optimize-high-overlap`) to actually produce
the sorted-opacity plot.

### `--refine-mid` / `--no-refine-mid`

```
--refine-mid     (default)
--no-refine-mid
```

This flag is forwarded straight to `analyze_group` (in
`group_derivatives.py`) which segments each bin's sorted-opacity curve
into `low / mid / high` (and optionally `mid1 / mid2`) pieces.

* With `--refine-mid` (default), `analyze_group` runs
  `iterative_refine_breaks`, which can split the middle segment in two by
  setting `seg["split_mid"] = True` and choosing a `seg["b_mid"]` index.
  When that happens, `plot_sorted_weighted_opacity_per_tau_bin` draws an
  additional dashed vertical line at `b_mid` for that bin.
* With `--no-refine-mid`, `iterative_refine_breaks` is skipped, so
  `seg["split_mid"]` stays `False`, no `b_mid` is computed, and no
  `b_mid` line is drawn — only the outer `b1` and `b2` break lines remain.

The `compute_bot_segment_overlap_per_tau_bin` table also reflects the
choice: with `--refine-mid` the `mid` row may be replaced by separate
`mid1` and `mid2` rows.

## Sorted-opacity plots

Four representative `sorted_weighted_opacity_per_tau_bin` runs are committed
in `plots/`. All four use explicit `--tau-bin-edges` (no optimizer), so the
sorted-opacity plot is produced; the only differences are the number of
tau-bins and whether `--refine-mid` is on or off.

| Plot | Bins | `refine-mid` |
| --- | --- | --- |
| `plots/sorted_4bin_refinemid.jpg` | 4 | on (default) |
| `plots/sorted_4bin_no_refinemid.jpg` | 4 | off |
| `plots/sorted_8bin_refinemid.jpg` | 8 | on (default) |
| `plots/sorted_8bin_no_refinemid.jpg` | 8 | off |

![4 tau-bins, refine-mid on](plots/sorted_4bin_refinemid.jpg)

*4 tau-bins, `--refine-mid`: the middle segment of each bin is allowed to
split, so a dashed `b_mid` line appears wherever `iterative_refine_breaks`
finds a better 4-segment (low / mid1 / mid2 / high) fit.*

![4 tau-bins, refine-mid off](plots/sorted_4bin_no_refinemid.jpg)

*4 tau-bins, `--no-refine-mid`: same edges, but `split_mid` is forced
`False` for every bin, so only the outer `b1` and `b2` break lines are
drawn and the middle segment is a single piece.*

![8 tau-bins, refine-mid on](plots/sorted_8bin_refinemid.jpg)

*8 tau-bins (edges from a previous `--optimize-high-overlap` run), with
mid-segment refinement enabled. With more, narrower tau-bins the
sorted-opacity curves are flatter, but several still benefit from a
`b_mid` split.*

![8 tau-bins, refine-mid off](plots/sorted_8bin_no_refinemid.jpg)

*8 tau-bins, `--no-refine-mid`: the same 8-bin edges, but with no
mid-segment refinement; useful as a baseline for comparing the segmentation
quality against the refined version above.*

### Reproducing the plots

```bash
# 4 tau-bins, refine-mid (default)
uv run python tausort.py main \
  --tau-bin-edges -0.63 --tau-bin-edges -0.1 --tau-bin-edges 1.5 \
  --tau-bin-edges 3.8 --tau-bin-edges 7.0 \
  --refine-mid

# 8 tau-bins, no-refine-mid
uv run python tausort.py main \
  --tau-bin-edges -0.63 --tau-bin-edges -0.3 --tau-bin-edges -0.15 \
  --tau-bin-edges 0.0  --tau-bin-edges 0.25 --tau-bin-edges 0.7 \
  --tau-bin-edges 1.5  --tau-bin-edges 3.9  --tau-bin-edges 7.0 \
  --no-refine-mid
```

Each run writes `sorted_weighted_opacity_per_tau_bin.jpg` to the CWD;
rename/move it into `plots/` to reproduce the files above.

## Radiative-transfer Q_rad comparison

`compare_Qrad_from_kappa.py` validates the binned-opacity tables by computing the
radiative heating rate Q_rad from each table and comparing it against the full-ODF
reference. It is **self-contained** in this repo — the RT solver (`rte.py`) and the
1D model atmospheres (`models/`) live here, no external dependencies.

```bash
uv run python compare_Qrad_from_kappa.py
```

For each opacity table it:

1. reads the table with `kappa_band_reader.read_kappa_4_band_comparison`,
2. interpolates `ln κ` / `ln B` onto a 1D STAGGER atmosphere (`models/G_SSD`),
3. solves the 1D RTE per band (short characteristics, `rte.Solver`),
4. sums the per-band heating rate to Q_rad and plots `Q/ρ` and the residual
   `(Q − Q_full)/ρ` vs `log10 τ_ross`, writing `Qrad_comparison.png`.

Inputs it expects (all gitignored — provide or regenerate them):

- `data/kappa_grey.dat`, `data/kappa_fullodf.dat`, `data/kappa_12_band.dat` — the
  gray / full-ODF / C-12-band reference tables (full-ODF is the residual baseline).
- `kappa_<…>band_…_lam_….dat` in the repo root — any tables emitted by
  `tausort.py main`; they are auto-discovered and labelled by their binning
  (`tg…` single-cell, `lm…_tg…` per-cell λ-split, `lm…_sl…` split-flag).

Restrict/relabel the plotted cases via the `SELECT` / `LABELS` lists near the top of
the script (empty `SELECT` plots gray, full, 12band, and every discovered table).

Relevant files: `compare_Qrad_from_kappa.py` (driver), `rte.py` (RT solver),
`models/{F,G,K,M}_SSD` (STAGGER 1D atmospheres).
