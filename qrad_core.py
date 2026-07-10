"""Q_rad scoring core — shared by the webapp and the Q_rad optimizer.

Holds the edge-INDEPENDENT precompute (reading the ODF, interpolating opacity onto
the atmosphere, the Rosseland tau profile, tau at tau_lambda=1, and the full-ODF /
gray reference heating rates) and a `score_binning()` that maps a binning (tau edges,
lambda edges, per-tau-group split flags, model) to the resulting Q_rad and its
residual against the full-ODF reference.

The atmosphere is selectable: any validated 1D model under `models/` (see
`scan_models` / `validate_model_file`). Per-model invariants and reference curves are
cached (`inv_for` / `reference`, keyed by filename), so the first use of a model pays
the ~10-30 s precompute and later `score_binning()` calls only re-run the cheap
per-edge pipeline (assign -> sort -> split -> band-average -> RTE, ~3 s). `None`
selects `DEFAULT_MODEL` (G2_1D.dat).

This is the exact pipeline the webapp used to inline in `compute()`; it lives here so
both `webapp/server.py` and `qrad_optimize.py` can call it without an HTTP layer.
"""

from __future__ import annotations

import sys
import threading
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

DZ = 1.0e6  # atmosphere grid spacing [cm] (uniform; matches models/G2_1D.dat) for the int-Q weight
NMU = 4
WINDOW = (-5.0, 2.5)  # log10(tau_ross) window for the rms metric
SKIP = 1440  # first N (far-UV) sub-bins skipped in the binning-diagram scatter

MODELS_DIR = _REPO / "models"  # 1D atmospheres live here (ASCII: z rho p T, one row per depth)
DEFAULT_MODEL = "G2_1D.dat"  # the model used when a caller doesn't pick one

# Per-model caches. Each model's edge-independent invariants (INV) and reference Q_rad
# curves (REF) are computed once (precompute is ~10-30 s) and reused. Keyed by bare filename.
_INV_CACHE: dict[str, dict] = {}
_REF_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()  # serialize the (expensive, memory-heavy) cache fills


def _model_name(model) -> str:
    """Normalize a model selector to a bare filename under models/ (None -> the default)."""
    if model is None or model == "":
        return DEFAULT_MODEL
    return Path(model).name


# --- model-file discovery + validation ------------------------------------------
def validate_model_file(path) -> dict:
    """Check that `path` is a usable 1D atmosphere and report exactly what's wrong.

    A usable model is an ASCII table with 4 columns (z, rho, pressure, temperature),
    at least 2 rows, and a strictly monotonically DECREASING height column z (top of
    the atmosphere first). Returns a dict: {name, ok, error, n_rows, n_cols}. `error`
    is None when ok, else a human-readable one-liner naming the first problem found.
    """
    path = Path(path)
    info: dict = {"name": path.name, "ok": False, "error": None, "n_rows": None, "n_cols": None}
    try:
        data = np.loadtxt(path)
    except Exception as e:  # non-numeric, ragged rows, unreadable, etc.
        info["error"] = f"not readable as a numeric table ({type(e).__name__}: {e})"
        return info
    data = np.atleast_2d(data)
    n_rows, n_cols = int(data.shape[0]), int(data.shape[1])
    info["n_rows"], info["n_cols"] = n_rows, n_cols
    if n_cols != 4:
        info["error"] = f"{n_cols} columns, need 4 (z, rho, pressure, temp)"
        return info
    if n_rows < 2:
        info["error"] = f"{n_rows} row(s), need >= 2"
        return info
    z = data[:, 0]
    dz = np.diff(z)
    if not np.all(dz < 0):
        i = int(np.argmax(dz >= 0))  # first non-decreasing step (dz[i] between rows i and i+1)
        info["error"] = f"z (col 1) not strictly decreasing: row {i + 1} z={z[i + 1]:.6g} >= row {i} z={z[i]:.6g}"
        return info
    info["ok"] = True
    return info


def scan_models() -> list[dict]:
    """Validate every file in models/. Returns per-file reports sorted by name (see
    `validate_model_file`). Directories and dot-files are skipped; everything else is
    reported (ok or not) so a bad file surfaces with its reason."""
    reports = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.iterdir()):
            if p.is_file() and not p.name.startswith("."):
                reports.append(validate_model_file(p))
    return reports


def valid_model_names() -> list[str]:
    """Names of the models under models/ that pass validation (selectable atmospheres)."""
    return [r["name"] for r in scan_models() if r["ok"]]


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


def _atm_rt(inv):
    """A model atmosphere in the orientation the RTE wants.

    Returns (z, rho, pre, tem) top-of-atmosphere first. The models are top-first but store z
    as height (descending), while `compute_tau`/`Solver` want z increasing into the atmosphere
    (tau=0 at index 0), so z = depth-from-top = atm.z[0] - atm.z; rho/p/T are in file order.
    """
    atm = inv["atm"]
    return atm.z[0] - atm.z, atm.rho, atm.p, atm.T


def _qrad_from_table(inv, kap_tab, b_tab, ttab, ptab):
    """Q_rad(z) summed over bands on `inv`'s atmosphere, mirroring compare_Qrad's Q_from_kappa.

    kap_tab: [nband, nT, np] ln(kappa); b_tab: [nband, nT] ln(B); ttab/ptab log10.
    Empty (all-NaN) bands are dropped; non-finite ln(B) -> B=0.
    """
    z, rho, pre, tem = _atm_rt(inv)
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


def reference(model=None):
    """Compute (once, per model) the gray/full-ODF (and optional goldenS) Q_rad references
    and the log10(tau) axis on the chosen atmosphere. Cached in `_REF_CACHE[name]`."""
    name = _model_name(model)
    if name in _REF_CACHE:
        return _REF_CACHE[name]
    inv = inv_for(name)
    with _CACHE_LOCK:
        if name in _REF_CACHE:  # filled while we waited for the lock
            return _REF_CACHE[name]
        z, rho, _pre, _tem = _atm_rt(inv)

        # gray reference -> tau axis
        g = read_kappa_4_band_comparison(str(_REPO / "data" / "kappa_grey.dat"))
        q_gray, k_gray = _qrad_from_table(
            inv,
            np.asarray(g.kap_mean),
            np.asarray(g.B_band, dtype=np.float64),
            np.asarray(g.tab_T),
            np.asarray(g.tab_p),
        )
        tau_ref = np.maximum(compute_tau(z, k_gray, rho)[0], 1e-20)
        ltau = np.log10(tau_ref)

        # full-ODF reference (the residual baseline)
        f = read_kappa_4_band_comparison(str(_REPO / "data" / "kappa_fullodf.dat"))
        q_full, _ = _qrad_from_table(
            inv,
            np.asarray(f.kap_mean),
            np.asarray(f.B_band, dtype=np.float64),
            np.asarray(f.tab_T),
            np.asarray(f.tab_p),
        )

        # optional "golden standard" reference table, plotted when data/kappa_goldenS.dat is present.
        q_golden = None
        golden_path = _REPO / "data" / "kappa_goldenS.dat"
        if golden_path.exists():
            gd = read_kappa_4_band_comparison(str(golden_path))
            q_golden, _ = _qrad_from_table(
                inv,
                np.asarray(gd.kap_mean),
                np.asarray(gd.B_band, dtype=np.float64),
                np.asarray(gd.tab_T),
                np.asarray(gd.tab_p),
            )

        ref = dict(ltau=ltau, q_full=q_full, q_gray=q_gray, q_golden=q_golden, rho=rho)
        _REF_CACHE[name] = ref
    return ref


def inv_for(model=None) -> dict:
    """Return the cached edge-independent invariants for a model, computing them on first use."""
    name = _model_name(model)
    if name not in _INV_CACHE:
        precompute(name)
    return _INV_CACHE[name]


def precompute(model=None) -> dict:
    """Edge-independent setup for a model: mirror the front of tausort.main once and cache it.

    Reads the chosen atmosphere (models/<name>) and the ODF, interpolates opacity onto that
    atmosphere, and derives the Rosseland tau profile, tau at tau_lambda=1, and the binning-diagram
    coordinates. Cached in `_INV_CACHE[name]`; also warms that model's reference()."""
    name = _model_name(model)
    if name in _INV_CACHE:
        return _INV_CACHE[name]
    with _CACHE_LOCK:
        if name in _INV_CACHE:  # filled while we waited for the lock
            return _INV_CACHE[name]
        t0 = time.perf_counter()
        atm = ts.read_atmospheric_model(MODELS_DIR / name)
        npy = _REPO / "ODF_format.npy"
        odf = ts.read_odf_npy(npy) if npy.exists() else ts.read_odf_netcdf(_REPO / "ODF_nc_format.nc")
        cont = ts.read_continuum_data(_REPO / "continuumabs.dat", odf.nbins, odf.nt, odf.np)
        # Upcast to float64 (only ~16 MB) so the tau integration / Rosseland-mean reference /
        # RTE path stays float64 even when the ODF+continuum are stored as float32 — cumulative
        # sums there are precision-sensitive. The big float32 win is in the band-averaging path
        # (calculate_tau_bin_opacities), which reads odf.ODF/cont directly and accumulates in float64.
        interpolated_opacity = np.asarray(ts.interpolate_kappa_to_atmosphere(odf, cont, atm), dtype=np.float64)
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

        inv = dict(
            model=name,
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
        _INV_CACHE[name] = inv
        print(
            f"[precompute] {name} ready in {time.perf_counter() - t0:.1f}s (odf {odf.nt}x{odf.np}, {len(wl_centers)} sub-bins)"
        )
    reference(name)  # warm the model's gray/full/golden reference + tau axis (outside the lock)
    return inv


def score_binning(
    tau_edges, lambda_edges, flags, model=None, *, n_splits=3, lambda_edges_per_tau=None, binning_tree=None, window=None
) -> dict:
    """Map a binning to Q_rad + residual metrics against the full-ODF reference.

    `model` selects the atmosphere the binning + RTE run on (a bare filename under models/;
    None -> DEFAULT_MODEL). It is precomputed on first use. Grouping modes (highest priority first):
      - general 2D guillotine: pass `binning_tree` (a {window_tau, window_lam, root} tree); the
        other edge args are ignored. Any rectangular tiling (see `build_group_specs_tree`).
      - per-tau-group lambda: pass `lambda_edges_per_tau` (one lambda-edge list per tau group).
      - shared lambda + flags (default): `flags` resolved (one bool per tau group).

    Returns raw full-length arrays (`q`, `resid`, `ltau`, `rho`, `q_full`, `q_gray`) plus
    scalar metrics and the descriptor/membership the binning diagram needs.
    """
    inv = inv_for(model)
    odf, cont, atm = inv["odf"], inv["cont"], inv["atm"]
    ref = reference(model)
    clamped_top = float(-np.log10(inv["tau_ross"][inv["max_height_idx"]] + 0.2))

    if binning_tree is not None:
        # guillotine tree: membership from the raw window, descriptor with the clamped top edge
        tw, lw, root = binning_tree["window_tau"], binning_tree["window_lam"], binning_tree["root"]
        band_index = ts.assign_tree(inv["tau_at_lam1"], inv["wl_centers"], root, tw, lw)
        group_tau_edges, group_lam_edges = ts.build_group_specs_tree(root, [clamped_top, float(tw[1])], lw)
    elif lambda_edges_per_tau is not None:
        # per-tau-group lambda: membership from un-clamped tau, descriptor from clamped
        clamped = list(tau_edges)
        clamped[0] = clamped_top
        _gt0, _gl0, offs = ts.build_group_specs_per_tau(tau_edges, lambda_edges_per_tau)
        band_index = ts.assign_per_tau_lambda(
            inv["tau_at_lam1"], inv["wl_centers"], tau_edges, lambda_edges_per_tau, offs
        )
        group_tau_edges, group_lam_edges, _offc = ts.build_group_specs_per_tau(clamped, lambda_edges_per_tau)
    else:
        clamped = list(tau_edges)
        clamped[0] = clamped_top
        # shared-tau + per-group split flags
        _gt0, _gl0, s2cg, s2sg = ts.build_group_specs_split_lambda(tau_edges, lambda_edges, flags)
        band_index = ts.assign_split_lambda(inv["tau_at_lam1"], inv["wl_centers"], tau_edges, lambda_edges, s2cg, s2sg)
        group_tau_edges, group_lam_edges, _a, _b = ts.build_group_specs_split_lambda(clamped, lambda_edges, flags)
    n_groups = int(group_tau_edges.shape[0])

    sorted_per_bin = ts.sort_weighted_opacity_per_tau_bin(
        atm=atm,
        odf=odf,
        interpolated_opacity=inv["interpolated_opacity"],
        tau_rosseland=inv["tau_ross"],
        band_index=band_index,
        group_tau_edges=group_tau_edges,
        wavelength_grid_subbins_centers=inv["wl_centers"],
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

    q, _kz = _qrad_from_table(inv, kap_tab, b_tab, np.asarray(odf.T), np.asarray(odf.P))

    ltau = ref["ltau"]
    rho = ref["rho"]
    q_full = ref["q_full"]
    resid = (q - q_full) / rho
    # rms / max_abs are scored over the log10(tau_Ross) window (default WINDOW=(-5,2.5)); the
    # caller can narrow it to focus the fit on a depth slice. int_q stays whole-atmosphere.
    win = WINDOW if window is None else (float(window[0]), float(window[1]))
    in_win = (ltau >= min(win)) & (ltau <= max(win))
    if int(in_win.sum()) < 2:
        raise ValueError(f"scoring window {win} (log10 tau_Ross) covers < 2 atmosphere points; widen it")
    rms = float(np.sqrt(np.mean((resid[in_win]) ** 2)))
    max_abs = float(np.abs(resid[in_win]).max())
    int_q = float(-q.sum() * DZ / 1e10)
    int_full = float(-q_full.sum() * DZ / 1e10)

    return {
        "rms": rms,
        "max_abs": max_abs,
        "int_q_pct": (int_q - int_full) / int_full * 100.0,
        "window": (float(min(win)), float(max(win))),
        "q": q,
        "resid": resid,
        "ltau": ltau,
        "rho": rho,
        "q_full": q_full,
        "q_gray": ref["q_gray"],
        "q_golden": ref.get("q_golden"),
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


def _kappa_dat_name(
    tau_edges, lambda_edges, flags, lambda_edges_per_tau, clamped_tau, n_bands, n_splits, binning_tree=None
):
    """Self-describing .dat filename for a binning (delegates to tausort per mode)."""
    if binning_tree is not None:
        return ts.build_kappa_dat_filename(nbands=n_bands, n_splits=n_splits, binning_tree=binning_tree)
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


def save_kappa_dat(
    tau_edges, lambda_edges, flags, model=None, *, lambda_edges_per_tau=None, binning_tree=None, n_splits=3, path=None
):
    """Build the binning's C-format kappa table and write it to disk.

    Runs the same grouping + sort + band-average pipeline as `score_binning` but keeps the
    full opacity result, NaN-masks empty bands (as `tausort.main` does before saving), packs it
    with `tausort.build_kappa_band_comparison`, and writes it with `write_kappa_4_band_comparison`
    (`kap_mean = ln(mixed)` in `[nBands, NT, Np]`). `flags` may be None in per-tau mode. `path`
    overrides the output path; otherwise a self-describing name is used (in the CWD). `model`
    selects the atmosphere the binning runs on (bare filename under models/; None -> default).
    Returns `(written_path, descriptive_name)`.
    """
    inv = inv_for(model)
    odf, cont, atm = inv["odf"], inv["cont"], inv["atm"]
    clamped_top = float(-np.log10(inv["tau_ross"][inv["max_height_idx"]] + 0.2))
    clamped = None

    if binning_tree is not None:
        tw, lw, root = binning_tree["window_tau"], binning_tree["window_lam"], binning_tree["root"]
        band_index = ts.assign_tree(inv["tau_at_lam1"], inv["wl_centers"], root, tw, lw)
        group_tau_edges, _gl = ts.build_group_specs_tree(root, [clamped_top, float(tw[1])], lw)
    elif lambda_edges_per_tau is not None:
        clamped = list(tau_edges)
        clamped[0] = clamped_top
        _gt0, _gl0, offs = ts.build_group_specs_per_tau(tau_edges, lambda_edges_per_tau)
        band_index = ts.assign_per_tau_lambda(
            inv["tau_at_lam1"], inv["wl_centers"], tau_edges, lambda_edges_per_tau, offs
        )
        group_tau_edges, _gl, _o = ts.build_group_specs_per_tau(clamped, lambda_edges_per_tau)
    else:
        clamped = list(tau_edges)
        clamped[0] = clamped_top
        _gt0, _gl0, s2cg, s2sg = ts.build_group_specs_split_lambda(tau_edges, lambda_edges, flags)
        band_index = ts.assign_split_lambda(inv["tau_at_lam1"], inv["wl_centers"], tau_edges, lambda_edges, s2cg, s2sg)
        group_tau_edges, _gl, _a, _b = ts.build_group_specs_split_lambda(clamped, lambda_edges, flags)
    n_groups = int(group_tau_edges.shape[0])

    sorted_per_bin = ts.sort_weighted_opacity_per_tau_bin(
        atm=atm,
        odf=odf,
        interpolated_opacity=inv["interpolated_opacity"],
        tau_rosseland=inv["tau_ross"],
        band_index=band_index,
        group_tau_edges=group_tau_edges,
        wavelength_grid_subbins_centers=inv["wl_centers"],
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
    name = _kappa_dat_name(
        tau_edges, lambda_edges, flags, lambda_edges_per_tau, clamped, n_bands, n_splits, binning_tree=binning_tree
    )
    written = str(path) if path is not None else name
    ts.write_kappa_4_band_comparison(written, comparison)
    return written, name
