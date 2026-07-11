# Manuscript: Direct optimization of the (τ, λ) opacity-binning partition

A&A (Astronomy & Astrophysics) manuscript for the project's core contribution:
**optimizing the (τ, λ) opacity-binning partition directly against the radiative
heating-rate residual `Q_rad`**, instead of hand-placing τ-edges and λ-cuts.

Built on the method in `qrad_optimize.py` / `qrad_core.py`, with the RT solver
`rte.py`, all in the repo root.

## Files

| file | what |
| --- | --- |
| `ms.tex` | the manuscript |
| `refs.bib` | bibliography (entries for Vögler+2004 and its predecessors taken verbatim from that paper's reference list) |
| `aa.cls`, `aa.bst` | A&A LaTeX class v9.0 and bibliography style (from the EDP/CTAN distribution) |
| `aa_example.tex` | the official A&A template kept for reference (not part of the manuscript) |
| `figures/` | `fig1` binning diagram, `fig2` Q_rad comparison, `fig3` before/after, `fig4` sorted-opacity |

## Build

Needs a TeX distribution (TeX Live / MacTeX) — none is installed on this machine
yet, so build on [Overleaf](https://www.overleaf.com) (upload this folder) or
after installing TeX:

```bash
pdflatex ms
bibtex   ms
pdflatex ms
pdflatex ms
```

or, with `latexmk`:
```bash
latexmk -pdf ms
```

## Pre-submission TODOs (flagged in `ms.tex` with `% TODO:`)

1. **Authors / affiliations / emails** — placeholder block at the top of `ms.tex`.
2. **Finalize the results table** — the rms values in Table 1 are the development
   numbers from the README (produced on the STAGGER-era atmosphere). Re-run on the
   present `models/G2_1D.dat` for the published figures:
   ```bash
   uv run python compare_Qrad_from_kappa.py        # -> Qrad_comparison.png  (fig2)
   uv run python qrad_optimize.py ... --save-plot fig3   # before/after      (fig3)
   ```
   The **relative ordering** is atmosphere-robust; only the absolute numbers need
   refreshing. Replace `figures/fig2_*` / `figures/fig3_*` and update Table 1.
3. **Vectorize figures** — A&A prefers PDF/EPS. Regenerate the figures as PDF
   (the plotting scripts use matplotlib) and update the `\includegraphics` paths.
4. **`\date{Received …; accepted …}`** — fill in on submission.
5. Consider extending the validation to a 3D-RHD mean-opacity table (see
   Discussion caveats) before the production version.

## References provenance

`refs.bib` keys `nordlund1982 … vogler2004` reproduce the reference list of
Vögler, Bruls & Schüssler (2004), the paper cited in `tausort.py` for the
Planck↔Rosseland blend (its Eq. 12) and the band means (Eqs. 16–17). The local
copy of that paper's full HTML is
`../papers/voegler_bruls_schuessler_2004_aa421_741_full.html`.
