"""Q_rad scoring core — shared by the webapp and the Q_rad optimizer.

Holds the edge-INDEPENDENT precompute (reading the ODF, interpolating opacity onto
the atmosphere, the Rosseland tau profile, tau at tau_lambda=1, and the per-star
full-ODF / gray reference heating rates) and a single `score_binning()` that maps a
binning (tau edges, lambda edges, per-tau-group split flags, star) to the resulting
Q_rad and its residual against the full-ODF reference.

This is the exact pipeline the webapp used to inline in `compute()`; it lives here so
both `webapp/server.py` and `qrad_optimize.py` can call it without an HTTP layer.
`precompute()` is expensive (~10-30 s, reads the ODF) and runs once; each
`score_binning()` then only re-runs the cheap per-edge pipeline (assign -> sort ->
split -> band-average -> RTE, ~3 s).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import tausort as ts  # noqa: E402
from kappa_band_reader import read_kappa_4_band_comparison  # noqa: E402
from rte import Solver, compute_tau  # noqa: E402

# --- physical constant + interpolation helper (same as compare_Qrad_from_kappa) ---
_cvac = 3.0e10
_h = 2.0 * np.pi * 1.0546e-27
_kB = 1.3807e-16
_SBC = (2 * np.pi**5 * _kB**4) / (15 * _cvac**2 * _h**3)

DZ = 1.0e6  # STAGGER grid spacing [cm]
NMU = 4
WINDOW = (-5.0, 4.0)  # log10(tau_ross) window for the rms metric
SKIP = 1440  # first N (far-UV) sub-bins skipped in the binning-diagram scatter

INV: dict = {}  # edge-independent invariants (filled by precompute)
REF_CACHE: dict = {}  # per-star reference Q_rad / axis


def resolve_flags(split_lambda, n_tau: int) -> list[bool]:
    """Per-tau-group lambda-split flags. Falls back to all-True (uniform split) when
    `split_lambda` is missing or the wrong length; with one lambda cell it's a no-op."""
    if split_lambda is not None and len(split_lambda) == n_tau:
        return [bool(b) for b in split_lambda]
    return [True] * n_tau


def _bilin_interp(dat, x, y):
    nx, ny = dat.shape
    x = np.clip(x, 0, nx - 2)
    y = np.clip(y, 0, ny - 2)
    wx, wy = x % 1.0, y % 1.0
    xi, yi = np.asarray(x, dtype=int), np.asarray(y, dtype=int)
    return (
        (1 - wx) * (1 - wy) * dat[xi, yi]
        + wx * (1 - wy) * dat[xi + 1, yi]
        + (1 - wx) * wy * dat[xi, yi + 1]
        + wx * wy * dat[xi + 1, yi + 1]
    )


def _load_star(star: str):
    """Read a models/<star> STAGGER atmosphere -> (z, rho, pre, tem)."""
    tvar = np.fromfile(str(_REPO / "models" / star), dtype=np.float32)
    tvar = tvar[4:].reshape([int(tvar[0]), int(tvar[1])])
    nz = tvar.shape[-1]
    z = np.arange(nz) * DZ
    rho = np.flip(tvar[0])
    pre = np.flip(tvar[2])
    tem = np.flip(tvar[3])
    return z, rho, pre, tem


def _qrad_from_table(kap_tab, b_tab, ttab, ptab, star):
    """Q_rad(z) summed over bands, mirroring compare_Qrad's Q_from_kappa.

    kap_tab: [nband, nT, np] ln(kappa); b_tab: [nband, nT] ln(B); ttab/ptab log10.
    Empty (all-NaN) bands are dropped; non-finite ln(B) -> B=0.
    """
    z, rho, pre, tem = REF_CACHE[star]["atm"]
    keep = ~np.isnan(kap_tab).all(axis=(1, 2))
    kap_tab = kap_tab[keep]
    b_tab = np.where(np.isfinite(b_tab[keep]), b_tab[keep], -700.0)
    nband = kap_tab.shape[0]
    nz = z.size

    lt = np.interp(np.log10(tem), ttab, np.arange(ttab.size))
    lp = np.interp(np.log10(pre), ptab, np.arange(ptab.size))
    k_z = np.zeros((nband, nz))
    b_z = np.zeros((nband, nz))
    for i in range(nband):
        b_z[i] = np.exp(np.interp(np.log10(tem), ttab, b_tab[i]))
        k_z[i] = np.exp(_bilin_interp(kap_tab[i], lt, lp))
    rt = Solver(z=z, rho=rho, kappa=k_z, S=b_z, nmu=NMU)
    rt.solve_rte()
    return rt.get_Q().sum(axis=0), k_z


def reference_for_star(star: str):
    """Precompute (and cache) the full-ODF Q_rad, gray Q_rad, and the log10(tau) axis."""
    if star in REF_CACHE:
        return REF_CACHE[star]
    REF_CACHE[star] = {"atm": _load_star(star)}
    z, rho, pre, tem = REF_CACHE[star]["atm"]

    # gray reference -> tau_ross axis
    g = read_kappa_4_band_comparison(str(_REPO / "data" / "kappa_grey.dat"))
    q_gray, k_gray = _qrad_from_table(
        np.asarray(g.kap_mean), np.asarray(g.B_band, dtype=np.float64), np.asarray(g.tab_T), np.asarray(g.tab_p), star
    )
    tau_ref = np.maximum(compute_tau(z, k_gray, rho)[0], 1e-20)
    ltau = np.log10(tau_ref)

    # full-ODF reference (the residual baseline)
    f = read_kappa_4_band_comparison(str(_REPO / "data" / "kappa_fullodf.dat"))
    q_full, _ = _qrad_from_table(
        np.asarray(f.kap_mean), np.asarray(f.B_band, dtype=np.float64), np.asarray(f.tab_T), np.asarray(f.tab_p), star
    )
    REF_CACHE[star].update(ltau=ltau, q_full=q_full, q_gray=q_gray, rho=rho)
    return REF_CACHE[star]


def precompute():
    """Edge-independent setup: mirror the front of tausort.main once."""
    t0 = time.perf_counter()
    atm = ts.read_atmospheric_model(_REPO / "G2_1D.dat")
    npy = _REPO / "ODF_format.npy"
    odf = ts.read_odf_npy(npy) if npy.exists() else ts.read_odf_netcdf(_REPO / "ODF_nc_format.nc")
    cont = ts.read_continuum_data(_REPO / "continuumabs.dat", odf.nbins, odf.nt, odf.np)
    interpolated_opacity = ts.interpolate_kappa_to_atmosphere(odf, cont, atm)
    kappa_on_atm, wl_centers = ts.calculate_reference_opacities_from_custom_tp_grid(
        atm, interpolated_opacity, odf.wavelength_grid, odf.subbin, odf.nbins, odf.nsubbins, kind="rosseland"
    )
    tau_ross = ts.compute_tau_rosseland(atm, kappa_on_atm)
    h_idx, _hh = ts.get_depth_at_tau_values_from_full_opacity(
        atm, interpolated_opacity, wl_centers, tau_values=[0.1, 1.0]
    )
    max_height_idx = int(np.max(h_idx[:, 1]))
    tau_at_lam1 = tau_ross[h_idx[:, -1]]

    # binning-diagram coordinates (edge-independent): per sub-bin log10 lambda[A]
    # and -log10 tau_Ros(tau_lambda=1), matching plot_tau_rosselend_at_tau_lambda_one.
    bin_x_all = np.log10(np.asarray(wl_centers) * 1e8)
    bin_y_all = -np.log10(np.clip(tau_at_lam1, 1e-300, None))

    INV.update(
        atm=atm,
        odf=odf,
        cont=cont,
        interpolated_opacity=interpolated_opacity,
        tau_ross=tau_ross,
        wl_centers=wl_centers,
        max_height_idx=max_height_idx,
        tau_at_lam1=tau_at_lam1,
        bin_x_all=bin_x_all,
        bin_y_all=bin_y_all,
        n_subbins=len(wl_centers),
    )
    # warm the default STAGGER atmosphere reference
    reference_for_star("G_SSD")
    print(f"[precompute] ready in {time.perf_counter() - t0:.1f}s (odf {odf.nt}x{odf.np}, {INV['n_subbins']} sub-bins)")


def score_binning(tau_edges, lambda_edges, flags, star, *, n_splits=3, lambda_edges_per_tau=None) -> dict:
    """Map a binning to Q_rad + residual metrics against the full-ODF reference.

    Requires `precompute()` to have run. Two grouping modes:
      - shared lambda + flags (default): `flags` resolved (one bool per tau group).
      - per-tau-group lambda: pass `lambda_edges_per_tau` (one lambda-edge list per tau
        group); `lambda_edges`/`flags` are then ignored. Each tau group gets its own
        wavelength split (see `build_group_specs_per_tau`).

    Returns raw full-length arrays (`q`, `resid`, `ltau`, `rho`, `q_full`, `q_gray`) plus
    scalar metrics and the descriptor/membership the binning diagram needs.
    """
    odf, cont, atm = INV["odf"], INV["cont"], INV["atm"]
    ref = reference_for_star(star)
    clamped = list(tau_edges)
    clamped[0] = float(-np.log10(INV["tau_ross"][INV["max_height_idx"]] + 0.2))

    if lambda_edges_per_tau is not None:
        # per-tau-group lambda: membership from un-clamped tau, descriptor from clamped
        _gt0, _gl0, offs = ts.build_group_specs_per_tau(tau_edges, lambda_edges_per_tau)
        band_index = ts.assign_per_tau_lambda(
            INV["tau_at_lam1"], INV["wl_centers"], tau_edges, lambda_edges_per_tau, offs
        )
        group_tau_edges, group_lam_edges, _offc = ts.build_group_specs_per_tau(clamped, lambda_edges_per_tau)
    else:
        # shared-tau + per-group split flags
        _gt0, _gl0, s2cg, s2sg = ts.build_group_specs_split_lambda(tau_edges, lambda_edges, flags)
        band_index = ts.assign_split_lambda(INV["tau_at_lam1"], INV["wl_centers"], tau_edges, lambda_edges, s2cg, s2sg)
        group_tau_edges, group_lam_edges, _a, _b = ts.build_group_specs_split_lambda(clamped, lambda_edges, flags)
    n_groups = int(group_tau_edges.shape[0])

    sorted_per_bin = ts.sort_weighted_opacity_per_tau_bin(
        atm=atm,
        odf=odf,
        interpolated_opacity=INV["interpolated_opacity"],
        tau_rosseland=INV["tau_ross"],
        band_index=band_index,
        group_tau_edges=group_tau_edges,
        wavelength_grid_subbins_centers=INV["wl_centers"],
        write_debug_json=False,
        verbose=False,
    )
    split_band_index = ts.build_split_band_index(
        sorted_per_bin, n_subbin_points=len(band_index), n_groups=n_groups, n_splits=n_splits
    )
    n_bands = n_groups * n_splits
    res = ts.calculate_tau_bin_opacities(odf=odf, cont=cont, band_index=split_band_index, n_bins=n_bands)
    mixed = np.asarray(res["kappa_mixed"])  # [nT, nP, nBands]
    b_band = np.asarray(res["B_band"])  # [nT, nBands]
    members = np.asarray(res["members_per_band"])

    with np.errstate(divide="ignore", invalid="ignore"):
        kap_tab = np.log(np.where(mixed > 0, mixed, np.nan)).transpose(2, 0, 1)  # [nBands, nT, nP]
        b_tab = np.log(np.where(b_band > 0, b_band, np.nan)).T  # [nBands, nT]
    kap_tab[members == 0] = np.nan

    q, _kz = _qrad_from_table(kap_tab, b_tab, np.asarray(odf.T), np.asarray(odf.P), star)

    ltau = ref["ltau"]
    rho = ref["rho"]
    q_full = ref["q_full"]
    resid = (q - q_full) / rho
    in_win = (ltau >= WINDOW[0]) & (ltau <= WINDOW[1])
    rms = float(np.sqrt(np.mean((resid[in_win]) ** 2)))
    max_abs = float(np.abs(resid[in_win]).max())
    int_q = float(-q.sum() * DZ / 1e10)
    int_full = float(-q_full.sum() * DZ / 1e10)

    return {
        "rms": rms,
        "max_abs": max_abs,
        "int_q_pct": (int_q - int_full) / int_full * 100.0,
        "q": q,
        "resid": resid,
        "ltau": ltau,
        "rho": rho,
        "q_full": q_full,
        "q_gray": ref["q_gray"],
        "n_groups": int(n_groups),
        "n_bands": int(n_bands),
        "n_empty": int((members == 0).sum()),
        "assigned": int((band_index >= 0).sum()),
        "total_subbins": int(len(band_index)),
        "members": members,
        "band_index": band_index,
        "group_tau_edges": group_tau_edges,
        "group_lam_edges": group_lam_edges,
    }


def _kappa_dat_name(tau_edges, lambda_edges, flags, lambda_edges_per_tau, clamped_tau, n_bands, n_splits):
    """Self-describing .dat filename for a binning (delegates to tausort per mode)."""
    if lambda_edges_per_tau is not None:
        return ts.build_kappa_dat_filename(
            nbands=n_bands,
            n_splits=n_splits,
            lambda_bin_edges=[lambda_edges_per_tau[0][0], lambda_edges_per_tau[0][-1]],
            tau_bin_edges=clamped_tau,
            lambda_edges_per_tau=lambda_edges_per_tau,
        )
    return ts.build_kappa_dat_filename(
        nbands=n_bands,
        n_splits=n_splits,
        lambda_bin_edges=lambda_edges,
        tau_bin_edges=clamped_tau,
        split_along_lambda=list(flags),
    )


def save_kappa_dat(tau_edges, lambda_edges, flags, star, *, lambda_edges_per_tau=None, n_splits=3, path=None):
    """Build the binning's C-format kappa table and write it to disk.

    Runs the same grouping + sort + band-average pipeline as `score_binning` but keeps the
    full opacity result, NaN-masks empty bands (as `tausort.main` does before saving), packs it
    with `tausort.build_kappa_band_comparison`, and writes it with `write_kappa_4_band_comparison`
    (`kap_mean = ln(mixed)` in `[nBands, NT, Np]`). `flags` may be None in per-tau mode. `path`
    overrides the output path; otherwise a self-describing name is used (in the CWD). `star` is
    unused (the table is star-independent) but kept for signature symmetry with `score_binning`.
    Returns `(written_path, descriptive_name)`.
    """
    odf, cont, atm = INV["odf"], INV["cont"], INV["atm"]
    clamped = list(tau_edges)
    clamped[0] = float(-np.log10(INV["tau_ross"][INV["max_height_idx"]] + 0.2))

    if lambda_edges_per_tau is not None:
        _gt0, _gl0, offs = ts.build_group_specs_per_tau(tau_edges, lambda_edges_per_tau)
        band_index = ts.assign_per_tau_lambda(
            INV["tau_at_lam1"], INV["wl_centers"], tau_edges, lambda_edges_per_tau, offs
        )
        group_tau_edges, _gl, _o = ts.build_group_specs_per_tau(clamped, lambda_edges_per_tau)
    else:
        _gt0, _gl0, s2cg, s2sg = ts.build_group_specs_split_lambda(tau_edges, lambda_edges, flags)
        band_index = ts.assign_split_lambda(INV["tau_at_lam1"], INV["wl_centers"], tau_edges, lambda_edges, s2cg, s2sg)
        group_tau_edges, _gl, _a, _b = ts.build_group_specs_split_lambda(clamped, lambda_edges, flags)
    n_groups = int(group_tau_edges.shape[0])

    sorted_per_bin = ts.sort_weighted_opacity_per_tau_bin(
        atm=atm,
        odf=odf,
        interpolated_opacity=INV["interpolated_opacity"],
        tau_rosseland=INV["tau_ross"],
        band_index=band_index,
        group_tau_edges=group_tau_edges,
        wavelength_grid_subbins_centers=INV["wl_centers"],
        write_debug_json=False,
        verbose=False,
    )
    split_band_index = ts.build_split_band_index(
        sorted_per_bin, n_subbin_points=len(band_index), n_groups=n_groups, n_splits=n_splits
    )
    n_bands = n_groups * n_splits
    res = ts.calculate_tau_bin_opacities(odf=odf, cont=cont, band_index=split_band_index, n_bins=n_bands)

    members = np.asarray(res["members_per_band"])
    empty = members == 0
    for key in ("kappa_planck", "kappa_rosseland", "kappa_mixed"):
        res[key][:, :, empty] = np.nan

    comparison = ts.build_kappa_band_comparison(res, odf)
    name = _kappa_dat_name(tau_edges, lambda_edges, flags, lambda_edges_per_tau, clamped, n_bands, n_splits)
    written = str(path) if path is not None else name
    ts.write_kappa_4_band_comparison(written, comparison)
    return written, name
