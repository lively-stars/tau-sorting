"""
compare_Qrad_from_kappa.py
==========================

Lightweight radiative-heating-rate (Qrad) comparison.

Instead of running the full opacity-binning pipeline
    load_binning -> calc_formation_heights -> sort_bins -> calc_kappa
(which needs the giant ODF data set resident in memory), this script reads
*precomputed* binned-opacity tables -- one per case -- as written by
``tausort.save_kappa``, interpolates them onto a 1D model atmosphere, solves
the RTE, and computes the radiative heating rate Q.

That makes the working set tiny: all we ever hold is a handful of small
(nband, nT, np) tables.

Real tables are read with ``kappa_band_reader.read_kappa_4_band_comparison``
(this repo). ``USE_MOCK = True`` swaps in synthetic tables of the same shape for
a quick plumbing test without any ``.dat`` files present.
"""

import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kappa_band_reader import read_kappa_4_band_comparison
from rte import Solver, compute_tau

# Resolve every input/output relative to this script so it runs from any cwd.
_HERE = Path(__file__).resolve().parent

# Stefan-Boltzmann constant + bilinear interpolation, inlined so this analysis
# script is self-contained (they used to come from an RT-side `tausort` module
# in a separate repo).
_cvac = 3.0e10  # speed of light [cm/s]
_h = 2.0 * np.pi * 1.0546e-27  # Planck constant [erg s]
_kB = 1.3807e-16  # Boltzmann constant [erg/K]
sbc = (2 * np.pi**5 * _kB**4) / (15 * _cvac**2 * _h**3)  # Stefan-Boltzmann [erg/cm^2/s/K^4]


def bilin_interp(dat, x, y, mode="1D"):
    """Bilinear interpolation of `dat` at fractional indices (x, y).

    mode='1D': x and y are equal-length -> a 1-D map; mode='2D': different
    lengths -> a 2-D map.
    """
    nx, ny = dat.shape
    if mode == "1D":
        x = np.clip(x, 0, nx - 2)
        y = np.clip(y, 0, ny - 2)
    elif mode == "2D":
        x = np.clip(x, 0, nx - 2)[:, None]
        y = np.clip(y, 0, ny - 2)[None, :]
    else:
        raise ValueError("bilin_interp mode must be '1D' or '2D'")
    wx, wy = x % 1.0, y % 1.0
    xi, yi = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    return (
        (1 - wx) * (1 - wy) * dat[xi, yi]
        + wx * (1 - wy) * dat[xi + 1, yi]
        + (1 - wx) * wy * dat[xi, yi + 1]
        + wx * wy * dat[xi + 1, yi + 1]
    )


# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
USE_MOCK = False  # mock fallback for a quick no-data plumbing test

REF_CASE = "full"  # case used as the reference in the residual panel


def _discover_our_tables():
    """Map case label -> path for every tau-sorting kappa table.

    Recognises the layouts emitted by tausort.main:
      * single lambda cell:  kappa_<n>band_tg<g>_sp<s>_tau_..._lam_....dat
      * per-cell lambda-split:kappa_<n>band_lm<L>_tg<g0-g1-...>_sp<s>_lam_....dat
      * split-flag mode:      kappa_<n>band_lm<L>_sl<10..>_sp<s>_tau_..._lam_....dat
    Single-cell tables are labelled 'tg<g>', the others 'lm<L>_<tg|sl><...>@<cut>';
    tables are ordered by band count and a colliding label gets a '*' suffix.
    """
    found = []
    for path in glob.glob(str(_HERE / "kappa_*band_*sp*_lam_*.dat")):
        base = os.path.basename(path)
        m = re.match(r"kappa_(\d+)band_(?:lm(\d+)_)?(tg[\d-]+|sl[01]+)_sp(\d+)_", base)
        if not m:
            continue
        nbands = int(m.group(1))
        n_lambda = int(m.group(2)) if m.group(2) else 1
        kind = m.group(3)  # "tg<counts>" or "sl<flags>"
        # interior lambda cut position(s), so a sweep over the cut stays distinguishable
        interior = base.split("_lam_")[-1][:-4].split("_")[1:-1]
        label = kind if n_lambda == 1 else f"lm{n_lambda}_{kind}@{'_'.join(interior)}"
        found.append((nbands, label, path))
    cases = {}
    for _nbands, label, path in sorted(found):
        lab = label
        while lab in cases:
            lab += "*"
        cases[lab] = path
    return cases


OUR_TABLES = _discover_our_tables()
KAPPA_FILES = {
    "gray": str(_HERE / "data" / "kappa_grey.dat"),
    "full": str(_HERE / "data" / "kappa_fullodf.dat"),
    "12band": str(_HERE / "data" / "kappa_12_band.dat"),  # C tausort 12-bin binning
    **OUR_TABLES,
}

# SELECT restricts the plot to specific case labels, in this order; empty -> plot
# gray/full/12band plus every discovered kappa table. LABELS overrides the legend
# text for any case.
SELECT: list[str] = ["gray", "full", "12band", "tg8", "lm2_sl1111@3.8", "lm2_sl1000@3.8"]
LABELS: dict[str, str] = {
    "tg8": "8τ, no λ-split (24 band)",
    "lm2_sl1111@3.8": "4τ × λ@3.8, all split (24 band)",
    "lm2_sl1000@3.8": "4τ × λ@3.8, group 0 only (15 band)",
}

if SELECT:
    CASES = [c for c in SELECT if c in KAPPA_FILES]
    missing = [c for c in SELECT if c not in KAPPA_FILES]
    if missing:
        print(f"SELECT: skipping unknown cases {missing}; available: {sorted(KAPPA_FILES)}")
else:
    CASES = ["gray", "full", "12band"] + list(OUR_TABLES)


# ---------------------------------------------------------------------------
# 1. Load a 1D model atmosphere (first index = top of the box)
# ---------------------------------------------------------------------------
# The single 1D atmosphere: models/G2_1D.dat (ASCII columns z rho p T, top-of-atmosphere first).
_atm = np.loadtxt(_HERE / "models" / "G2_1D.dat")
nz = _atm.shape[0]
dz = 1e6  # uniform grid spacing [cm], for the int-Q weight
# The RTE wants z increasing into the atmosphere; G2_1D.dat stores z as height (descending),
# so use depth-from-top. rho/p/T stay in the file's (top-first) order.
z = _atm[0, 0] - _atm[:, 0]
rho = _atm[:, 1]
pre = _atm[:, 2]
tem = _atm[:, 3]


# ---------------------------------------------------------------------------
# 2. Opacity-table reader (mock or real)
# ---------------------------------------------------------------------------
def _mock_kappa_table(case):
    """Synthetic stand-in for a real kappa table; identical return signature.

    Returns (kap_tab, B_tab, ttab, ptab, header) with:
      kap_tab : (nband, nT, np)   natural-log band opacity
      B_tab   : (nband, nT)       natural-log band-integrated Planck function
      ttab    : (nT,)             log10(T)   axis  (as save_kappa stores it)
      ptab    : (np,)             log10(p)   axis
      header  : int(8) -> [tau5000, nT, np, nband, pt_rhot, fullodf, scat, bh]

    The numbers are physically plausible but NOT real -- just enough to keep
    the RTE well behaved.  Delete this once real tables exist.
    """
    nT, nP = 100, 30
    ttab = np.linspace(np.log10(1500.0), np.log10(60000.0), nT)  # log10(T)
    ptab = np.linspace(-2.0, 8.0, nP)  # log10(p)
    nband = {"gray": 1, "multibin": 12, "full": 20}[case]

    LT, LP = np.meshgrid(ttab, ptab, indexing="ij")  # (nT, np)
    Tgrid = 10.0**ttab  # (nT,)
    lt_n = (LT - LT.min()) / (LT.max() - LT.min())  # normalised 0..1
    lp_n = (LP - LP.min()) / (LP.max() - LP.min())

    kap_tab = np.empty((nband, nT, nP))
    B_tab = np.empty((nband, nT))
    for i in range(nband):
        offset = 0.4 * (i - nband / 2.0) / max(nband, 1)  # bands differ a bit
        # smooth, bounded ln(kappa): rises with pressure, falls with temperature
        kap_tab[i] = np.log(1.0e-2) + 3.0 * lp_n - 1.5 * lt_n + offset
        # band Planck: equal slice of sigma*T^4/pi  (crude, but stable)
        B_tab[i] = np.log(sbc * Tgrid**4 / np.pi / nband)

    header = np.asarray([0, nT, nP, nband, 0, int(case == "full"), 0, 0], dtype=int)
    return kap_tab, B_tab, ttab, ptab, header


def _load_dat(path, label):
    """Read a kappa .dat table via read_kappa_4_band_comparison.

    kap_mean/B_band come back as ln(values) with a leading band axis and
    tab_T/tab_p as log10 axes, so we only adapt the dataclass to the
    (kap_tab, B_tab, ttab, ptab, header) 5-tuple.
    It also handles the grey file's tau5000 section and the full-ODF file's
    trailing nuout via the header flags.

    Sanitization, both of which would otherwise poison the RTE:
      * empty bands (all-NaN kappa, e.g. an empty (tau-group, split)) are
        dropped with a warning;
      * non-finite ln(B) entries (ln(0) = -inf for zero-flux sub-bins at
        extreme T in the full-ODF table) become -700, i.e. B = 0 after exp().

    Also prints the band-summed Planck coverage vs sigma*T^4/pi -- flux the
    binning left unassigned is simply absent, and this quantifies it.
    """
    kbc = read_kappa_4_band_comparison(path)
    kap_tab = np.asarray(kbc.kap_mean)  # (nband, nT, np)  ln(kappa)
    B_tab = np.asarray(kbc.B_band, dtype=np.float64)  # (nband, nT)      ln(B)
    ttab = np.asarray(kbc.tab_T, dtype=np.float64)  # (nT,)  log10(T)
    ptab = np.asarray(kbc.tab_p, dtype=np.float64)  # (np,)  log10(p)

    keep = ~np.isnan(kap_tab).all(axis=(1, 2))
    if not keep.all():
        print(f"    {label!r}: dropping {(~keep).sum()} empty band(s): {np.flatnonzero(~keep)}")
        kap_tab, B_tab = kap_tab[keep], B_tab[keep]
    B_tab = np.where(np.isfinite(B_tab), B_tab, -700.0)  # ln(B) -> B = 0
    if not np.isfinite(kap_tab).all():
        print(f"    {label!r}: WARNING -- non-finite kappa inside kept bands; Q will be NaN")

    # spectral completeness of the binned source function
    coverage = np.exp(B_tab).sum(axis=0) / (sbc * (10.0**ttab) ** 4 / np.pi)
    print(
        f"    {label!r}: {kap_tab.shape[0]} bands; sum(B_band)/(sigma*T^4/pi) = "
        f"{coverage.min():.3f} .. {coverage.max():.3f} over the T grid"
    )

    header = np.asarray(
        [kbc.tau5000bin, kbc.NT, kbc.Np, kap_tab.shape[0], kbc.pp_axis, kbc.full_odf, kbc.scatter_on, kbc.back_heating],
        dtype=int,
    )
    return kap_tab, B_tab, ttab, ptab, header


def load_kappa_table(case):
    """Return (kap_tab, B_tab, ttab, ptab, header) for a case.

    Real tables are read with the tau-sorting reader; USE_MOCK switches the
    cases the mock knows about to synthetic tables for quick plumbing tests.
    """
    if USE_MOCK and case in ("gray", "full", "multibin"):
        return _mock_kappa_table(case)
    return _load_dat(KAPPA_FILES[case], case)


# ---------------------------------------------------------------------------
# 3. Read a table -> interpolate onto the atmosphere -> solve RTE -> get Q
# ---------------------------------------------------------------------------
def Q_from_kappa(case, z, rho, tem, pre, nmu=4):
    """Compute the heating rate for one case from its (binned) opacity table.

    Returns (Q, k_z, rt):
      Q   : heating rate, shape (nband, nz)   [erg/cm^3/s]
      k_z : band opacities on the atmosphere, shape (nband, nz)
      rt  : the RTE Solver (so the caller can grab rt.F etc.)
    """
    kap_tab, B_tab, ttab, ptab, _header = load_kappa_table(case)
    nband = kap_tab.shape[0]

    # The table axes are stored in LOG10 (note: not natural log), so index with
    # log10(T), log10(p).  These give *fractional* table indices for bilin_interp.
    lt = np.interp(np.log10(tem), ttab, np.arange(ttab.size))
    lp = np.interp(np.log10(pre), ptab, np.arange(ptab.size))

    k_z = np.zeros([nband, nz])
    B_z = np.zeros([nband, nz])
    for i in range(nband):
        # tables hold ln(values); exp() recovers physical opacity / Planck fn
        B_z[i] = np.exp(np.interp(np.log10(tem), ttab, B_tab[i]))
        k_z[i] = np.exp(bilin_interp(kap_tab[i], lt, lp))

    rt = Solver(z=z, rho=rho, kappa=k_z, S=B_z, nmu=nmu)
    rt.solve_rte()
    return rt.get_Q(), k_z, rt


# ---------------------------------------------------------------------------
# 4. Loop over cases
# ---------------------------------------------------------------------------
Qs = {}
F_full = None
tau_ref = None
for case in CASES:
    print("solving:", case)
    Q, k_z, rt = Q_from_kappa(case, z, rho, tem, pre)
    Qs[case] = Q
    if case == "full":
        F_full = rt.F[..., 0]  # outgoing flux (per band), if needed later
    if case == "gray":
        # the gray opacity IS the Rosseland mean, so tau built from it is a good
        # reference optical-depth axis -- a stand-in for the pipeline's tau_ross
        tau_ref = np.maximum(compute_tau(z, k_z, rho)[0], 1e-20)

if tau_ref is None:
    raise RuntimeError("need the 'gray' case in CASES to build the reference tau axis")
ltau = np.log10(tau_ref)


# ---------------------------------------------------------------------------
# 5. Metrics vs the reference case, then plot Q/rho and the residuals
# ---------------------------------------------------------------------------
Q_ref = Qs[REF_CASE].sum(axis=0)

# metrics over the physically meaningful window (= the plotted range); the
# topmost point (tau = 0, ltau clipped to -20) is a boundary artefact that
# would otherwise dominate max|dQ/rho| for every case
in_win = (ltau >= -5.0) & (ltau <= 4.0)

print("\nmetrics over log10(tau_ross) in [-5, 4]:")
print(f"{'case':>8s} {'file':<28s} {'int Q':>10s} {'dint%':>7s} {'max|dQ/rho|':>12s} {'@ltau':>6s} {'rms dQ/rho':>12s}")
for case in CASES:
    Qtot = Qs[case].sum(axis=0)
    dq = ((Qtot - Q_ref) / rho)[in_win]
    lt_w = ltau[in_win]
    i_max = int(np.abs(dq).argmax())
    int_q = -Qs[case].sum() * dz / 1e10
    int_ref = -Qs[REF_CASE].sum() * dz / 1e10
    name = os.path.basename(KAPPA_FILES[case])[:28] if case in KAPPA_FILES else "mock"
    print(
        f"{case:>8s} {name:<28s} {int_q:10.4f} {100 * (int_q - int_ref) / int_ref:7.2f} "
        f"{np.abs(dq).max():12.3e} {lt_w[i_max]:6.2f} {np.sqrt(np.mean(dq**2)):12.3e}"
    )

# gray/full/12band keep fixed colors; every other ("ours") table shares the magma
# colormap as a gradient (dark -> bright in plot order), which reads as a sweep
# whether the swept axis is the tau-group count or the lambda-cut position.
fixed = {"gray": "tab:blue", "full": "tab:green", "12band": "tab:cyan"}
ours = [c for c in CASES if c not in fixed]
colors = dict(fixed)
colors.update(zip(ours, plt.cm.magma(np.linspace(0.1, 0.85, max(len(ours), 2)))))
tg_cases = [c for c in CASES if c.startswith("tg")]
# The "Nr. tau bins:" legend header only makes sense for a bare tg-number sweep;
# suppress it when explicit LABELS are in play (a focused, mixed comparison).
show_tau_header = bool(tg_cases) and not LABELS


def _label(case):
    if case in LABELS:
        return LABELS[case]
    return case[2:] if case.startswith("tg") else case


plt.figure(figsize=(7, 7))
for case in CASES:
    Qtot = Qs[case].sum(axis=0)
    label = _label(case)

    plt.subplot(2, 1, 1)
    if show_tau_header and case == tg_cases[0]:
        plt.plot([], [], " ", label="Nr. tau bins:")  # legend section header
    plt.plot(ltau, Qtot / rho, label=label, color=colors.get(case))

    plt.subplot(2, 1, 2)
    plt.plot(ltau, (Qtot - Q_ref) / rho, label=label, color=colors.get(case))

plt.subplot(2, 1, 1)
plt.xlim([4, -5])
plt.axhline(y=0, linestyle=":")
plt.axvline(x=0, linestyle=":")
plt.legend(loc="upper left", fontsize=8)
plt.ylabel(r"$Q/\rho$ [erg/g/s]")

plt.subplot(2, 1, 2)
plt.xlim([4, -5])
plt.axhline(y=0, linestyle=":")
plt.axvline(x=0, linestyle=":")
# Auto-fit y to the residuals inside the visible tau window only, so the panel
# is not stretched by the off-screen tau=0 boundary point (ltau ~ -20).
_res_win = np.concatenate([((Qs[c].sum(axis=0) - Q_ref) / rho)[in_win] for c in CASES])
_lo, _hi = float(np.nanmin(_res_win)), float(np.nanmax(_res_win))
_pad = 0.05 * (_hi - _lo) if _hi > _lo else 1.0
plt.ylim(_lo - _pad, _hi + _pad)
plt.ylabel(r"$(Q - Q_{\rm " + REF_CASE + r"})/\rho$ [erg/g/s]")
plt.xlabel(r"$\log_{10}\tau_{\rm ross}$")

plt.tight_layout()
plt.savefig(str(_HERE / "Qrad_comparison.png"), dpi=150)
print("wrote Qrad_comparison.png")
plt.show()
