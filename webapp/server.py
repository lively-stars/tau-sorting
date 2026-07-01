"""Interactive Q_rad explorer — backend.

A tiny stdlib HTTP server that lets you play with tau / lambda bin edges (and
per-tau-group lambda-split flags) and see how the resulting binned-opacity table
reproduces the radiative heating rate Q_rad against the full-ODF reference.

The expensive, edge-independent work (reading the ODF, interpolating opacity onto
the atmosphere, the Rosseland tau profile, tau at tau_lambda=1, and the full-ODF /
gray reference heating rates per STAGGER atmosphere) is done ONCE at startup and
cached. Each /api/compute request then only re-runs the cheap per-edge pipeline:
assign -> sort -> split -> band-average -> RTE, so it responds in a few seconds.

Run:  uv run python webapp/server.py      (then open http://localhost:8765)
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import tausort as ts  # noqa: E402
from kappa_band_reader import read_kappa_4_band_comparison  # noqa: E402
from rte import Solver, compute_tau  # noqa: E402

# --- physical constant + interpolation helper (same as compare_Qrad_from_kappa) ---
_cvac = 3.0e10
_h = 2.0 * np.pi * 1.0546e-27
_kB = 1.3807e-16
_SBC = (2 * np.pi**5 * _kB**4) / (15 * _cvac**2 * _h**3)

PORT = 8771
DZ = 1.0e6  # STAGGER grid spacing [cm]
NMU = 4
WINDOW = (-5.0, 4.0)  # log10(tau_ross) window for the rms metric

_LOCK = threading.Lock()
INV: dict = {}  # edge-independent invariants
REF_CACHE: dict = {}  # per-star reference Q_rad / axis


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


def _reference_for_star(star: str):
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
    h_idx, _h = ts.get_depth_at_tau_values_from_full_opacity(
        atm, interpolated_opacity, wl_centers, tau_values=[0.1, 1.0]
    )
    max_height_idx = int(np.max(h_idx[:, 1]))
    tau_at_lam1 = tau_ross[h_idx[:, -1]]

    INV.update(
        atm=atm,
        odf=odf,
        cont=cont,
        interpolated_opacity=interpolated_opacity,
        tau_ross=tau_ross,
        wl_centers=wl_centers,
        max_height_idx=max_height_idx,
        tau_at_lam1=tau_at_lam1,
        n_subbins=len(wl_centers),
    )
    # warm the default STAGGER atmosphere reference
    _reference_for_star("G_SSD")
    print(f"[precompute] ready in {time.perf_counter() - t0:.1f}s (odf {odf.nt}x{odf.np}, {INV['n_subbins']} sub-bins)")


def compute(tau_edges, lambda_edges, split_lambda, star):
    """Run the per-edge pipeline and return the Q_rad curves + metrics."""
    with _LOCK:
        odf, cont = INV["odf"], INV["cont"]
        atm = INV["atm"]
        n_tau = len(tau_edges) - 1
        # A single code path: shared-tau + per-group lambda flags. All-True (default)
        # reproduces a uniform split; n_lambda == 1 makes the flags a no-op.
        if split_lambda and len(split_lambda) == n_tau:
            flags = [bool(b) for b in split_lambda]
        else:
            flags = [True] * n_tau

        ref = _reference_for_star(star)

        # membership from the un-clamped edges
        _gt0, _gl0, s2cg, s2sg = ts.build_group_specs_split_lambda(tau_edges, lambda_edges, flags)
        band_index = ts.assign_split_lambda(INV["tau_at_lam1"], INV["wl_centers"], tau_edges, lambda_edges, s2cg, s2sg)
        # atmosphere-top clamp on the first tau edge for the descriptor used downstream
        clamped = list(tau_edges)
        clamped[0] = float(-np.log10(INV["tau_ross"][INV["max_height_idx"]] + 0.2))
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
        n_splits = 3
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

        # sort by the tau axis, and drop the off-screen tau=0 boundary point so it
        # doesn't dominate the plot's y-range (metrics above already use [-5, 4]).
        order = np.argsort(ltau)
        lo = ltau[order]
        disp = (lo >= WINDOW[0] - 1.0) & (lo <= WINDOW[1] + 1.0)
        idx = order[disp]
        return {
            "ltau": ltau[idx].tolist(),
            "q_over_rho": (q[idx] / rho[idx]).tolist(),
            "q_full_over_rho": (q_full[idx] / rho[idx]).tolist(),
            "q_gray_over_rho": (ref["q_gray"][idx] / rho[idx]).tolist(),
            "residual": resid[idx].tolist(),
            "n_bands": int(n_bands),
            "n_groups": int(n_groups),
            "n_empty": int((members == 0).sum()),
            "rms": rms,
            "max_abs": max_abs,
            "int_q_pct": (int_q - int_full) / int_full * 100.0,
            "assigned": int((band_index >= 0).sum()),
            "total_subbins": int(len(band_index)),
        }


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # quiet

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (Path(__file__).resolve().parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/init":
            stars = sorted(p.name for p in (_REPO / "models").glob("*_SSD"))
            self._send(
                200,
                json.dumps(
                    {
                        "stars": stars,
                        "default_star": "G_SSD" if "G_SSD" in stars else (stars[0] if stars else ""),
                        "default_tau_edges": [-0.63, 0.3488, 1.2275, 2.885, 7.0],
                        "default_lambda_edges": [3.0, 3.8, 5.0],
                    }
                ),
            )
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/api/compute":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            tau_edges = [float(x) for x in req["tau_edges"]]
            lambda_edges = [float(x) for x in req["lambda_edges"]]
            split_lambda = req.get("split_lambda") or None
            star = req.get("star") or "G_SSD"
            if len(tau_edges) < 2:
                raise ValueError("need at least 2 tau edges")
            if len(lambda_edges) < 2:
                raise ValueError("need at least 2 lambda edges")
            if any(tau_edges[i] >= tau_edges[i + 1] for i in range(len(tau_edges) - 1)):
                raise ValueError("tau edges must be strictly increasing")
            if any(lambda_edges[i] >= lambda_edges[i + 1] for i in range(len(lambda_edges) - 1)):
                raise ValueError("lambda edges must be strictly increasing")
            t0 = time.perf_counter()
            out = compute(tau_edges, lambda_edges, split_lambda, star)
            out["elapsed"] = round(time.perf_counter() - t0, 2)
            self._send(200, json.dumps(out))
        except Exception as e:
            traceback.print_exc()
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    print("[startup] precomputing invariants (reads the ODF; ~10-30s)...")
    precompute()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[ready] Q_rad explorer at http://localhost:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
