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
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

import qrad_core as qc  # noqa: E402
import tausort as ts  # noqa: E402

PORT = 8771
SKIP = qc.SKIP
WINDOW = qc.WINDOW

_LOCK = threading.Lock()


def compute(tau_edges, lambda_edges, split_lambda, star):
    """Run the per-edge pipeline (via qrad_core) and shape the Q_rad curves + metrics for the UI."""
    with _LOCK:
        flags = qc.resolve_flags(split_lambda, len(tau_edges) - 1)
        r = qc.score_binning(tau_edges, lambda_edges, flags, star)

        ltau, rho = r["ltau"], r["rho"]
        q, q_full, q_gray, resid = r["q"], r["q_full"], r["q_gray"], r["resid"]

        # sort by the tau axis, and drop the off-screen tau=0 boundary point so it
        # doesn't dominate the plot's y-range (metrics already use [-5, 4]).
        order = np.argsort(ltau)
        lo = ltau[order]
        disp = (lo >= WINDOW[0] - 1.0) & (lo <= WINDOW[1] + 1.0)
        idx = order[disp]

        # binning diagram: sub-bin scatter (colored by group downstream) + group boxes
        bg = r["band_index"][SKIP:]
        return {
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
            "assigned": r["assigned"],
            "total_subbins": r["total_subbins"],
            # binning diagram (log10 lambda[A] vs -log10 tau_Ros); scatter is invariant,
            # bin_group + boxes track the current edges.
            "bin_x": qc.INV["bin_x_all"][SKIP:].tolist(),
            "bin_y": qc.INV["bin_y_all"][SKIP:].tolist(),
            "bin_group": bg.astype(int).tolist(),
            "group_tau_edges": r["group_tau_edges"].tolist(),
            "group_lam_edges": r["group_lam_edges"].tolist(),
        }


def optimize_edges(tau_window, lambda_edges, max_bins, threshold):
    """Greedy high-overlap optimizer -> an optimized *shared* tau binning.

    Runs tausort's optimizer over a single lambda cell spanning the whole given
    lambda range, so it returns one tau-edge list (not per-cell) that the webapp's
    shared-tau + split-flag model can consume directly. Starts from the outer
    [top, bottom] window and grows interior edges up to `max_bins` groups, nudging
    them to maximize the worst group's high-segment overlap. With threshold >= ~1
    (unreachable) it always grows to exactly `max_bins` optimally-placed groups.
    """
    with _LOCK:
        window_lambda = [float(lambda_edges[0]), float(lambda_edges[-1])]
        outer = [float(tau_window[0]), float(tau_window[-1])]
        per_cell = ts.optimize_tau_bin_edges(
            atm=qc.INV["atm"],
            odf=qc.INV["odf"],
            interpolated_opacity=qc.INV["interpolated_opacity"],
            tau_rosseland=qc.INV["tau_ross"],
            tau_rosseland_at_tau_lambda_one=qc.INV["tau_at_lam1"],
            wavelength_grid_subbins_centers=qc.INV["wl_centers"],
            max_height_idx=qc.INV["max_height_idx"],
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
        if self.path not in ("/api/compute", "/api/optimize"):
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
            t0 = time.perf_counter()
            if self.path == "/api/optimize":
                max_bins = int(req.get("max_bins", 4))
                if not 2 <= max_bins <= 12:
                    raise ValueError("target tau groups must be between 2 and 12")
                # unreachable threshold -> grow to exactly max_bins, optimally placed
                edges = optimize_edges(tau_edges, lambda_edges, max_bins, threshold=1.01)
                out = {"tau_edges": edges, "elapsed": round(time.perf_counter() - t0, 2)}
                self._send(200, json.dumps(out))
                return
            split_lambda = req.get("split_lambda") or None
            star = req.get("star") or "G_SSD"
            out = compute(tau_edges, lambda_edges, split_lambda, star)
            out["elapsed"] = round(time.perf_counter() - t0, 2)
            self._send(200, json.dumps(out))
        except Exception as e:
            traceback.print_exc()
            self._send(400, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    print("[startup] precomputing invariants (reads the ODF; ~10-30s)...")
    qc.precompute()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[ready] Q_rad explorer at http://localhost:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
