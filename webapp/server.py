"""Interactive Q_rad explorer — backend.

A tiny stdlib HTTP server that lets you play with tau / lambda bin edges (and
per-tau-group lambda-split flags) and see how the resulting binned-opacity table
reproduces the radiative heating rate Q_rad against the full-ODF reference.

The expensive, edge-independent work (reading the ODF, interpolating opacity onto the
atmosphere, the reference heating rates) is done ONCE at startup by
`qrad_core.precompute()` and cached; each /api/compute then re-runs only the cheap
per-edge pipeline via `qrad_core.score_binning` (assign -> sort -> split -> band-average
-> RTE, ~3 s). The scoring core lives in `qrad_core` so the Q_rad optimizer can share it.

Run:  uv run python webapp/server.py      (then open http://localhost:8771)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import qrad_core as qc  # noqa: E402
import qrad_optimize as qopt  # noqa: E402
import tausort as ts  # noqa: E402

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8771"))
# Cap how many tau-groups the Q_rad optimizer may grow to. Each extra group enlarges the
# per-eval opacity arrays; on RAM-constrained hosts lower this (e.g. 5) to keep the peak in
# budget. Env-tunable so it can be changed via compose without rebuilding the image.
MAX_GROUPS = int(os.environ.get("QRAD_MAX_GROUPS", "8"))
SKIP = qc.SKIP
WINDOW = qc.WINDOW

_LOCK = threading.Lock()


def _resolve_model(req):
    """Validate the request's chosen atmosphere and return its bare filename (default when
    absent). Raises ValueError with the reason if the file isn't a usable 1D model."""
    name = qc._model_name(req.get("model") or None)
    rep = qc.validate_model_file(qc.MODELS_DIR / name)
    if not rep["ok"]:
        raise ValueError(f"model '{name}' is not usable: {rep['error']}")
    return name


def _window(req):
    """Optional (lo, hi) log10(tau_Ross) scoring window from window_lo/window_hi; None if absent
    (-> qrad_core.WINDOW). Raises if the two are given but don't span a sensible range."""
    lo, hi = req.get("window_lo"), req.get("window_hi")
    if lo is None or hi is None:
        return None
    lo, hi = float(lo), float(hi)
    if hi - lo < 0.5:
        raise ValueError("scoring window (log10 tau_Ross) must span at least 0.5; widen it")
    return (lo, hi)


# --- Q_rad optimizer: one background job at a time, polled by the UI ---
_QOPT_LOCK = threading.Lock()
_QOPT: dict = {
    "running": False,
    "cancel": False,
    "history": [],
    "result": None,
    "error": None,
    "t0": 0.0,
    "n_evals": 0,
    "best": None,
    "rms0": None,
    "groups": 0,
    "model": "",
    "diagram": None,  # binning-diagram data of the current best (for the live top-plot preview)
}


def _run_qrad_opt(tau_edges, lambda_edges, flags, model, opt):
    """Thread target: run the Q_rad optimizer, streaming progress into _QOPT."""

    def on_eval(n, cost, r):
        _QOPT["n_evals"] = n
        _QOPT["groups"] = int(r.get("n_groups", 0))
        rms = float(r["rms"])
        if _QOPT["best"] is None or rms < _QOPT["best"]:
            _QOPT["best"] = rms
            # snapshot the current best binning's diagram (sub-bin group + boxes) — cheap, no RTE —
            # so the UI can live-preview the top plot. Assigned atomically; the poller reads it.
            bi, gte, gle = r.get("band_index"), r.get("group_tau_edges"), r.get("group_lam_edges")
            if bi is not None and gte is not None and gle is not None:
                _QOPT["diagram"] = {
                    "rms": rms,
                    "bin_group": np.asarray(bi)[SKIP:].astype(int).tolist(),
                    "group_tau_edges": np.asarray(gte).tolist(),
                    "group_lam_edges": np.asarray(gle).tolist(),
                }

    def on_progress(tag, value, groups, n):
        if tag == "start":
            _QOPT["rms0"] = float(value)
        _QOPT["history"].append(
            {
                "tag": tag,
                "rms": float(value),
                "groups": int(groups),
                "n_evals": int(n),
                "t": round(time.perf_counter() - _QOPT["t0"], 1),
            }
        )

    try:
        _QOPT["result"] = qopt.optimize_qrad(
            tau_edges,
            lambda_edges,
            flags=flags,
            model=model,
            opt_tau=opt["opt_tau"],
            opt_lambda=opt["opt_lambda"],
            opt_flags=opt["opt_flags"],
            grow=opt["grow"],
            metric=opt["metric"],
            method=opt["method"],
            max_seconds=opt["max_seconds"],
            max_evals=opt["max_evals"],
            max_groups=opt["max_groups"],
            min_gap_tau=opt["min_gap_tau"],
            min_gap_lam=opt["min_gap_lam"],
            grow_tol_rel=opt["grow_tol_rel"],
            window=opt["window"],
            target_rms=opt["target_rms"],
            plateau_evals=opt["plateau_evals"],
            plateau_rel=opt["plateau_rel"],
            per_group_lambda=opt["per_group_lambda"],
            lambda_edges_per_tau=opt["lambda_edges_per_tau"],
            tree=opt["tree"],
            binning_tree=opt["binning_tree"],
            min_opacity_delta=opt["min_opacity_delta"],
            on_eval=on_eval,
            on_progress=on_progress,
            should_stop=lambda: _QOPT["cancel"],
        )
    except Exception as e:
        traceback.print_exc()
        _QOPT["error"] = f"{type(e).__name__}: {e}"
    finally:
        _QOPT["running"] = False


def compute(
    tau_edges,
    lambda_edges,
    split_lambda,
    model,
    lambda_edges_per_tau=None,
    binning_tree=None,
    window=None,
    min_opacity_delta=1.0,
):
    """Run the per-edge pipeline (via qrad_core) and shape the Q_rad curves + metrics for the UI.

    `model` selects the atmosphere the binning + RTE run on (validated file under models/).
    `window` (log10 tau_Ross (lo,hi)) narrows the rms/max_abs scoring range (None -> default).
    Grouping (highest priority first): `binning_tree` (general 2D guillotine), then
    `lambda_edges_per_tau` (per-tau-group lambda), else the shared-lambda + split-flag model.
    """
    with _LOCK:
        if binning_tree is not None:
            r = qc.score_binning(
                None, None, None, model, binning_tree=binning_tree, window=window, min_opacity_delta=min_opacity_delta
            )
        elif lambda_edges_per_tau is not None:
            r = qc.score_binning(
                tau_edges,
                None,
                None,
                model,
                lambda_edges_per_tau=lambda_edges_per_tau,
                window=window,
                min_opacity_delta=min_opacity_delta,
            )
        else:
            flags = qc.resolve_flags(split_lambda, len(tau_edges) - 1)
            r = qc.score_binning(
                tau_edges, lambda_edges, flags, model, window=window, min_opacity_delta=min_opacity_delta
            )
        inv = qc.inv_for(model)

        ltau, rho = r["ltau"], r["rho"]
        q, q_full, q_gray, resid = r["q"], r["q_full"], r["q_gray"], r["resid"]

        # sort by the tau axis, and drop the off-screen tau=0 boundary point. The display slice
        # is the UNION of the default view and the user's scoring window (each +-1 dex), so a
        # narrowed window never crops the plot below today's view.
        effwin = r["window"]
        order = np.argsort(ltau)
        lo = ltau[order]
        disp = (lo >= min(WINDOW[0], effwin[0]) - 1.0) & (lo <= max(WINDOW[1], effwin[1]) + 1.0)
        idx = order[disp]

        # binning diagram: sub-bin scatter (colored by group downstream) + group boxes
        bg = r["band_index"][SKIP:]
        out = {
            "ltau": ltau[idx].tolist(),
            "q_over_rho": (q[idx] / rho[idx]).tolist(),
            "q_full_over_rho": (q_full[idx] / rho[idx]).tolist(),
            "q_gray_over_rho": (q_gray[idx] / rho[idx]).tolist(),
            "residual": resid[idx].tolist(),
            "n_bands": r["n_bands"],
            "n_groups": r["n_groups"],
            "n_empty": r["n_empty"],
            "rms": r["rms"],
            "max_abs": r["max_abs"],
            "int_q_pct": r["int_q_pct"],
            "window": list(effwin),  # effective scoring window (log10 tau_Ros) -> UI guides
            "assigned": r["assigned"],
            "total_subbins": r["total_subbins"],
            # binning diagram (log10 lambda[A] vs -log10 tau_Ros); scatter is invariant,
            # bin_group + boxes track the current edges.
            "bin_x": inv["bin_x_all"][SKIP:].tolist(),
            "bin_y": inv["bin_y_all"][SKIP:].tolist(),
            "bin_group": bg.astype(int).tolist(),
            "group_tau_edges": r["group_tau_edges"].tolist(),
            "group_lam_edges": r["group_lam_edges"].tolist(),
            # per-band Q/ρ (signed) over the displayed depth slice — drives the 4th panel's
            # stacked-area decomposition; the band axis aligns 1:1 with (group, split) via n_splits.
            "q_per_band": (r["q_per_band"][:, idx] / rho[idx]).tolist(),
            "n_splits": int(r.get("n_splits", 3)),
        }
        # optional golden-standard reference curve (only when data/kappa_goldenS.dat exists)
        q_golden = r.get("q_golden")
        if q_golden is not None:
            out["q_golden_over_rho"] = (q_golden[idx] / rho[idx]).tolist()
        return out


def optimize_edges(tau_window, lambda_edges, max_bins, threshold, model=None):
    """Greedy high-overlap optimizer -> an optimized *shared* tau binning.

    Runs tausort's optimizer over a single lambda cell spanning the whole given
    lambda range, so it returns one tau-edge list (not per-cell) that the webapp's
    shared-tau + split-flag model can consume directly. Starts from the outer
    [top, bottom] window and grows interior edges up to `max_bins` groups, nudging
    them to maximize the worst group's high-segment overlap. With threshold >= ~1
    (unreachable) it always grows to exactly `max_bins` optimally-placed groups.
    """
    with _LOCK:
        inv = qc.inv_for(model)
        window_lambda = [float(lambda_edges[0]), float(lambda_edges[-1])]
        outer = [float(tau_window[0]), float(tau_window[-1])]
        per_cell = ts.optimize_tau_bin_edges(
            atm=inv["atm"],
            odf=inv["odf"],
            interpolated_opacity=inv["interpolated_opacity"],
            tau_rosseland=inv["tau_ross"],
            tau_rosseland_at_tau_lambda_one=inv["tau_at_lam1"],
            wavelength_grid_subbins_centers=inv["wl_centers"],
            max_height_idx=inv["max_height_idx"],
            initial_tau_bin_edges=outer,
            lambda_bin_edges=window_lambda,
            threshold=float(threshold),
            max_bins=int(max_bins),
            refine_mid=False,
        )
        return [round(float(e), 4) for e in per_cell[0]]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_download(self, data, filename):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # quiet

    def _atmosphere(self):
        """GET /api/atmosphere?model=<name> — the selected atmosphere's structure (z, rho, p, T),
        read straight from the model .dat. It's just the four columns — no ODF, no precompute, no
        RTE — so it's instant even for a model that was never binned."""
        try:
            q = parse_qs(urlparse(self.path).query)
            model = (q.get("model") or [None])[0]
            name = _resolve_model({"model": model})  # validates: 4 cols, >=2 rows, decreasing z
            data = np.atleast_2d(np.loadtxt(qc.MODELS_DIR / name))
            z, rho, pres, tem = (data[:, i].astype(float) for i in range(4))
            out = {
                "model": name,
                "n_levels": int(z.size),
                "height_mm": (z / 1e8).tolist(),  # height [Mm] (as stored, top-of-atmosphere first)
                "depth_mm": ((z[0] - z) / 1e8).tolist(),  # depth from top [Mm], increasing inward
                "rho": rho.tolist(),  # density [g/cm^3]
                "p": pres.tolist(),  # pressure [dyn/cm^2]
                "T": tem.tolist(),  # temperature [K]
            }
            self._send(200, json.dumps(out))
        except ValueError as e:
            self._send(400, json.dumps({"error": str(e)}))
        except Exception as e:
            traceback.print_exc()
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, (Path(__file__).resolve().parent / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/init":
            self._send(
                200,
                json.dumps(
                    {
                        "default_tau_edges": [-0.63, 0.3488, 1.2275, 2.885, 7.0],
                        "default_lambda_edges": [3.0, 3.8, 5.0],
                        "default_window": [WINDOW[0], WINDOW[1]],
                        "max_groups_ceiling": MAX_GROUPS,
                        "default_model": qc.DEFAULT_MODEL,
                        "models": qc.scan_models(),
                    }
                ),
            )
        elif self.path == "/api/models":
            # "Refresh models" button: re-scan models/ and report each file (ok or why not).
            self._send(200, json.dumps({"default_model": qc.DEFAULT_MODEL, "models": qc.scan_models()}))
        elif urlparse(self.path).path == "/api/atmosphere":
            self._atmosphere()
        elif self.path == "/api/optimize_qrad_status":
            self._send(
                200,
                json.dumps(
                    {
                        "running": _QOPT["running"],
                        "n_evals": _QOPT["n_evals"],
                        "best": _QOPT["best"],
                        "rms0": _QOPT["rms0"],
                        "groups": _QOPT["groups"],
                        "elapsed": round(time.perf_counter() - _QOPT["t0"], 1) if _QOPT["t0"] else 0.0,
                        "history": _QOPT["history"][-12:],
                        "result": _QOPT["result"],
                        "error": _QOPT["error"],
                        "diagram": _QOPT["diagram"],
                    }
                ),
            )
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/optimize_qrad_cancel":
            _QOPT["cancel"] = True
            self._send(200, json.dumps({"cancelled": True}))
            return
        if self.path not in ("/api/compute", "/api/optimize", "/api/optimize_qrad", "/api/kappa_dat"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            tau_edges = [float(x) for x in req["tau_edges"]]
            lambda_edges = [float(x) for x in req["lambda_edges"]]
            if len(tau_edges) < 2:
                raise ValueError("need at least 2 tau edges")
            if len(lambda_edges) < 2:
                raise ValueError("need at least 2 lambda edges")
            if any(tau_edges[i] >= tau_edges[i + 1] for i in range(len(tau_edges) - 1)):
                raise ValueError("tau edges must be strictly increasing")
            if any(lambda_edges[i] >= lambda_edges[i + 1] for i in range(len(lambda_edges) - 1)):
                raise ValueError("lambda edges must be strictly increasing")
            model = _resolve_model(req)
            t0 = time.perf_counter()
            if self.path == "/api/optimize":
                max_bins = int(req.get("max_bins", 4))
                if not 2 <= max_bins <= 12:
                    raise ValueError("target tau groups must be between 2 and 12")
                # unreachable threshold -> grow to exactly max_bins, optimally placed
                edges = optimize_edges(tau_edges, lambda_edges, max_bins, threshold=1.01, model=model)
                out = {"tau_edges": edges, "elapsed": round(time.perf_counter() - t0, 2)}
                self._send(200, json.dumps(out))
                return
            if self.path == "/api/optimize_qrad":
                with _QOPT_LOCK:
                    if _QOPT["running"]:
                        raise ValueError("a Q_rad optimization is already running")
                    _QOPT.update(
                        running=True,
                        cancel=False,
                        history=[],
                        result=None,
                        error=None,
                        t0=time.perf_counter(),
                        n_evals=0,
                        best=None,
                        rms0=None,
                        groups=0,
                        model=model,
                        diagram=None,
                    )
                try:
                    flags = qc.resolve_flags(req.get("split_lambda") or None, len(tau_edges) - 1)
                    qc.reference(model)  # warm the chosen model's reference before threading
                    metric = req.get("metric", "rms")
                    if metric not in ("rms", "maxabs", "int_q"):
                        raise ValueError(f"metric must be rms|maxabs|int_q, got {metric!r}")
                    method = req.get("method", "cd")
                    if method not in ("cd", "nm"):
                        raise ValueError(f"method must be cd|nm, got {method!r}")
                    target = req.get("target_rms")
                    opt = {
                        "opt_tau": bool(req.get("opt_tau", True)),
                        "opt_lambda": bool(req.get("opt_lambda", True)),
                        "opt_flags": bool(req.get("opt_flags", True)),
                        "grow": bool(req.get("grow", True)),
                        "per_group_lambda": bool(req.get("per_group_lambda", False)),
                        # per-group-lambda warm start (re-running keeps refining the current cuts)
                        "lambda_edges_per_tau": req.get("lambda_edges_per_tau") or None,
                        # general 2D guillotine mode + warm start (a {window_tau, window_lam, root} tree)
                        "tree": bool(req.get("tree", False)),
                        "binning_tree": req.get("binning_tree") or None,
                        "metric": metric,
                        "method": method,
                        "max_seconds": float(req.get("max_seconds", 300.0)),
                        "max_evals": int(req.get("max_evals", 5000)),
                        # user may ask for fewer than the host ceiling, never more.
                        "max_groups": max(2, min(MAX_GROUPS, int(req.get("max_groups", MAX_GROUPS)))),
                        "window": _window(req),
                        "target_rms": (float(target) if target else None),
                        "plateau_evals": max(0, int(req.get("plateau_evals", 0) or 0)),
                        "plateau_rel": float(req.get("plateau_rel", 0.005) or 0.005),
                        "grow_tol_rel": float(req.get("grow_tol_rel", 0.01) or 0.01),
                        "min_gap_tau": float(req.get("min_gap_tau", 0.15) or 0.15),
                        "min_gap_lam": float(req.get("min_gap_lam", 0.10) or 0.10),
                        "min_opacity_delta": float(req.get("min_opacity_delta", 1.0) or 1.0),
                    }
                    threading.Thread(
                        target=_run_qrad_opt, args=(tau_edges, lambda_edges, flags, model, opt), daemon=True
                    ).start()
                except Exception:
                    _QOPT["running"] = False
                    raise
                self._send(200, json.dumps({"started": True}))
                return
            if self.path == "/api/kappa_dat":
                # Build the current binning's kappa table and stream it back as a download.
                lpt = req.get("lambda_edges_per_tau") or None
                btree = req.get("binning_tree") or None
                min_od = float(req.get("min_opacity_delta", 1.0) or 1.0)
                fd, tmp = tempfile.mkstemp(suffix=".dat")
                os.close(fd)
                try:
                    with _LOCK:
                        if btree is not None:
                            _w, name = qc.save_kappa_dat(
                                None, None, None, model, binning_tree=btree, path=tmp, min_opacity_delta=min_od
                            )
                        elif lpt is not None:
                            _w, name = qc.save_kappa_dat(
                                tau_edges,
                                None,
                                None,
                                model,
                                lambda_edges_per_tau=lpt,
                                path=tmp,
                                min_opacity_delta=min_od,
                            )
                        else:
                            flags = qc.resolve_flags(req.get("split_lambda") or None, len(tau_edges) - 1)
                            _w, name = qc.save_kappa_dat(
                                tau_edges, lambda_edges, flags, model, path=tmp, min_opacity_delta=min_od
                            )
                        data = Path(tmp).read_bytes()
                finally:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                self._send_download(data, name)
                return
            split_lambda = req.get("split_lambda") or None
            lpt = req.get("lambda_edges_per_tau") or None
            btree = req.get("binning_tree") or None
            out = compute(
                tau_edges,
                lambda_edges,
                split_lambda,
                model,
                lambda_edges_per_tau=lpt,
                binning_tree=btree,
                window=_window(req),
                min_opacity_delta=float(req.get("min_opacity_delta", 1.0) or 1.0),
            )
            out["elapsed"] = round(time.perf_counter() - t0, 2)
            self._send(200, json.dumps(out))
        except Exception as e:
            traceback.print_exc()
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    print("[startup] precomputing invariants (reads the ODF; ~10-30s)...")
    qc.precompute()
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[ready] Q_rad explorer at http://localhost:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
