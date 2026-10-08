"""Q_rad-driven binning optimizer.

Searches over the opacity binning to directly MINIMIZE the Q_rad rms residual against
the full-ODF reference — unlike tausort's `optimize_tau_bin_edges`, which maximizes a
proxy (per-group high-segment overlap). As we saw in the webapp, the proxy and the real
target disagree, so this optimizes the metric that actually matters.

Decision variables (chosen scope): the interior tau/lambda cut positions of a guillotine
binnings tree (every grid is a guillotine tree), whose leaf count may grow. The outer
tau/lambda window is held fixed. The atmosphere is selectable via `model` (a validated 1D
model under models/; None -> DEFAULT_MODEL).

The optimizer is **tree-only**: every grouping (an explicit guillotine tree, per-tau-group
lambda edges, or shared tau + split flags) is normalized to a guillotine tree and refined via
a single seed/grow/polish path. `beam_width` is the structural knob: `>= 2` (default 3) runs
a non-greedy beam search over tree topologies (keeps rival tilings alive, tries several split
positions per leaf); `1` falls back to the greedy grow (one midpoint split per round,
committed immediately). Each evaluation is a full RTE solve (~3 s via `qrad_core.score_binning`),
so this is a run-and-wait / batch tool, bounded by an eval + wall-clock budget.

CLI:  uv run python qrad_optimize.py --help
"""

from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer

_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

import qrad_core  # noqa: E402
from tausort import parse_split_lambda  # noqa: E402


def _rss_mb() -> float:
    """Current resident set size in MB (Linux /proc/self/status); falls back to ru_maxrss."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _deep_size_mb(obj, _seen=None) -> float:
    """Approximate deep memory (MB) of a structure of dicts/lists/tuples/sets/scalars/ndarrays.
    Shares (same sub-node referenced by many parents) are counted once via id-dedup."""
    import sys as _sys

    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0.0
    _seen.add(oid)
    if isinstance(obj, np.ndarray):
        return obj.nbytes / 1e6
    s = _sys.getsizeof(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            s += _deep_size_mb(k, _seen) + _deep_size_mb(v, _seen)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            s += _deep_size_mb(v, _seen)
    return s / 1e6


# --- defaults -------------------------------------------------------------------
MIN_GAP_TAU = 0.15  # min spacing between tau edges [-log10 tau], keeps groups from collapsing
MIN_GAP_LAM = 0.10  # min spacing between lambda edges [log10 A]
EMPTY_PENALTY = 0.75  # multiplicative penalty per empty band (see make_evaluator)
ADJUST_STEPS = (0.10, 0.05, 0.02)


# --- guardrails -----------------------------------------------------------------
def _valid_monotone(edges) -> bool:
    return all(edges[i] < edges[i + 1] for i in range(len(edges) - 1))


def _min_gap_ok(edges, min_gap: float) -> bool:
    return all((edges[i + 1] - edges[i]) >= min_gap - 1e-12 for i in range(len(edges) - 1))


def _check_feasible(edges, min_gap: float, name: str) -> None:
    if not _valid_monotone(edges):
        raise ValueError(f"{name} edges must be strictly increasing, got {edges}")
    span = edges[-1] - edges[0]
    need = (len(edges) - 1) * min_gap
    if span < need - 1e-9:
        raise ValueError(
            f"{len(edges) - 1} {name} groups need span >= {need:.3f} but only {span:.3f} "
            f"available (min_gap={min_gap}); reduce groups or min_gap"
        )


# --- objective ------------------------------------------------------------------
def make_evaluator(
    model,
    *,
    metric="rms",
    empty_penalty=EMPTY_PENALTY,
    score_fn=None,
    on_eval=None,
    window=None,
    min_opacity_delta=None,
):
    """Return (evaluate, state). `evaluate(*, binning_tree) -> (cost, raw_dict)`.

    cost = base_metric * (1 + empty_penalty * n_empty). The multiplicative empty-band
    penalty counteracts the fact that an empty band is silently dropped from the Q sum
    (which would otherwise *lower* rms and let the search collapse groups). `model` selects
    the atmosphere passed through to `score_fn`; `window` (log10 tau_Ross (lo,hi)) narrows the
    rms/max_abs scoring range and is forwarded only when set, so an injected `score_fn` keeps
    its 4-positional signature by default. `score_fn` is dependency-injected (defaults to
    qrad_core.score_binning) so the search logic is unit-testable against an analytic objective
    with no ODF. `state["n_evals"]` counts calls. `on_eval(n_evals, cost, raw_dict)` (optional)
    fires after every evaluation — used by the webapp for a live progress ticker.
    """
    score = score_fn if score_fn is not None else qrad_core.score_binning
    state = {"n_evals": 0}
    _key = {"rms": "rms", "maxabs": "max_abs", "int_q": "int_q_pct"}[metric]
    _kw = {} if window is None else {"window": (float(window[0]), float(window[1]))}
    if min_opacity_delta is not None:
        _kw["min_opacity_delta"] = float(min_opacity_delta)

    def evaluate(*, binning_tree):
        # Tree-only: the optimizer always calls this with a guillotine tree. The positional
        # (tau, lambda, flags) / lambda_edges_per_tau branches are gone — those groupings are now
        # normalized to a tree before evaluation (see optimize_qrad's seeding shim).
        r = score(None, None, None, model, binning_tree=binning_tree, **_kw)
        base = abs(float(r[_key]))
        cost = base * (1.0 + empty_penalty * int(r.get("n_empty", 0)))
        state["n_evals"] += 1
        if on_eval is not None:
            on_eval(state["n_evals"], cost, r)
        return cost, r

    return evaluate, state


@dataclass
class _Budget:
    max_evals: int
    max_seconds: float
    state: dict
    t0: float
    should_stop: object = None  # optional callable -> True to abort at the next check
    target_rms: float | None = None  # stop once the best RAW rms drops to/below this
    plateau_evals: int = 0  # stop if the best rms hasn't improved over this many evals (0 = off)
    plateau_rel: float = 0.005  # "improvement" = best rms fell by >= this fraction of the reference
    best_rms: float = float("inf")  # running best RAW rms (r["rms"], independent of the metric)
    stop_reason: str = ""  # which condition ended the search (for reporting)
    _pl_ref_rms: float = float("inf")  # plateau reference rms
    _pl_ref_n: int = 0  # eval count at which _pl_ref_rms was last set

    def record(self, rms: float, n: int) -> None:
        """Track the running best rms + plateau reference. Called after every evaluation."""
        if rms < self.best_rms:
            self.best_rms = rms
        # reset the plateau window whenever we see a meaningful (>= plateau_rel) improvement
        if self._pl_ref_rms == float("inf") or (self._pl_ref_rms - rms) >= self.plateau_rel * self._pl_ref_rms:
            self._pl_ref_rms, self._pl_ref_n = rms, n

    def reset_plateau(self, n: int) -> None:
        """Restart the plateau window (e.g. after an accepted grow) so it isn't cut short."""
        self._pl_ref_rms, self._pl_ref_n = self.best_rms, n

    def exhausted(self) -> bool:
        if self.should_stop is not None and self.should_stop():
            self.stop_reason = self.stop_reason or "cancelled"
            return True
        n = self.state["n_evals"]
        if n >= self.max_evals:
            self.stop_reason = "max_evals"
            return True
        if (time.perf_counter() - self.t0) >= self.max_seconds:
            self.stop_reason = "max_seconds"
            return True
        if self.target_rms is not None and self.best_rms <= self.target_rms:
            self.stop_reason = "target_rms"
            return True
        if self.plateau_evals > 0 and self._pl_ref_n > 0 and (n - self._pl_ref_n) >= self.plateau_evals:
            self.stop_reason = "plateau"
            return True
        return False


@dataclass
class _Cfg:
    min_gap_tau: float = MIN_GAP_TAU
    min_gap_lam: float = MIN_GAP_LAM
    adjust_steps: tuple = ADJUST_STEPS
    # Few sweeps per tau visit + many rounds => the expensive tau block yields, so the
    # cheap flags/lambda blocks aren't starved of budget (they interleave each round).
    max_sweeps: int = 2
    max_block_rounds: int = 12
    block_tol: float = 1e-4  # relative improvement to keep iterating blocks


# --- general 2D guillotine tree search ------------------------------------------
# A binning tree is {"window_tau": [tlo,thi], "window_lam": [llo,lhi], "root": <node>},
# node = leaf {"leaf": True} or internal {"axis": "tau"|"lam", "at": float, "lo": .., "hi": ..}.
def _is_leaf(node) -> bool:
    return node.get("leaf", False) or "axis" not in node


def _count_leaves(node) -> int:
    return 1 if _is_leaf(node) else _count_leaves(node["lo"]) + _count_leaves(node["hi"])


def _n_leaves(tree) -> int:
    return _count_leaves(tree["root"])


def _leaf_rects(node, rect):
    """(tlo,thi,llo,lhi) per leaf, lo-before-hi DFS."""
    if _is_leaf(node):
        yield rect
        return
    tlo, thi, llo, lhi = rect
    at = float(node["at"])
    if node["axis"] == "tau":
        yield from _leaf_rects(node["lo"], (tlo, at, llo, lhi))
        yield from _leaf_rects(node["hi"], (at, thi, llo, lhi))
    else:
        yield from _leaf_rects(node["lo"], (tlo, thi, llo, at))
        yield from _leaf_rects(node["hi"], (tlo, thi, at, lhi))


def _root_rect(tree):
    wt, wl = tree["window_tau"], tree["window_lam"]
    return (float(wt[0]), float(wt[1]), float(wl[0]), float(wl[1]))


def _tree_feasible(tree, min_gap_tau, min_gap_lam) -> bool:
    """Every leaf rectangle must be non-inverted and respect the per-axis min gap."""
    for tlo, thi, llo, lhi in _leaf_rects(tree["root"], _root_rect(tree)):
        if thi <= tlo or lhi <= llo:
            return False
        if (thi - tlo) < min_gap_tau - 1e-12 or (lhi - llo) < min_gap_lam - 1e-12:
            return False
    return True


def _iter_internal(node, rect):
    """(node_ref, axis_lo, axis_hi) for each internal node (axis span = the range its `at` may move in)."""
    if _is_leaf(node):
        return
    tlo, thi, llo, lhi = rect
    at = float(node["at"])
    if node["axis"] == "tau":
        yield (node, tlo, thi)
        yield from _iter_internal(node["lo"], (tlo, at, llo, lhi))
        yield from _iter_internal(node["hi"], (at, thi, llo, lhi))
    else:
        yield (node, llo, lhi)
        yield from _iter_internal(node["lo"], (tlo, thi, llo, at))
        yield from _iter_internal(node["hi"], (tlo, thi, at, lhi))


def _iter_leaves_with_path(node, rect, path=()):
    """(path, rect) per leaf — path is a tuple of 'lo'/'hi' from the root, for in-place grow."""
    if _is_leaf(node):
        yield (path, rect)
        return
    tlo, thi, llo, lhi = rect
    at = float(node["at"])
    if node["axis"] == "tau":
        yield from _iter_leaves_with_path(node["lo"], (tlo, at, llo, lhi), (*path, "lo"))
        yield from _iter_leaves_with_path(node["hi"], (at, thi, llo, lhi), (*path, "hi"))
    else:
        yield from _iter_leaves_with_path(node["lo"], (tlo, thi, llo, at), (*path, "lo"))
        yield from _iter_leaves_with_path(node["hi"], (tlo, thi, at, lhi), (*path, "hi"))


def _node_at_path(root, path):
    n = root
    for step in path:
        n = n[step]
    return n


def _round_tree(tree) -> dict:
    """Deep-copy a tree with cut positions rounded to 4 decimals — a tidy wire/return payload."""

    def rec(node):
        if _is_leaf(node):
            return {"leaf": True}
        out = {
            "axis": node["axis"],
            "at": round(float(node["at"]), 4),
            "lo": rec(node["lo"]),
            "hi": rec(node["hi"]),
        }
        if node.get("sync") is not None:
            out["sync"] = node["sync"]
        if node.get("frozen"):
            out["frozen"] = True
        return out

    return {
        "window_tau": [round(float(v), 4) for v in tree["window_tau"]],
        "window_lam": [round(float(v), 4) for v in tree["window_lam"]],
        "root": rec(tree["root"]),
    }


def _check_tree_node(node, *, min_gap_tau, min_gap_lam, tlo, thi, llo, lhi) -> None:
    """Validate one decoded tree node in place (raises ValueError on any shape violation)."""
    if not isinstance(node, dict):
        raise ValueError("tree node must be an object")
    if node.get("leaf", False) or "axis" not in node:
        if "axis" not in node and not node.get("leaf", False):
            raise ValueError("tree leaf node needs 'leaf': true")
        return
    axis = node.get("axis")
    if axis not in ("tau", "lam"):
        raise ValueError(f"tree node axis must be 'tau'|'lam', got {node.get('axis')!r}")
    if not isinstance(node.get("at"), (int, float)) or not math.isfinite(float(node["at"])):
        raise ValueError(f"tree node 'at' must be a finite number, got {node.get('at')!r}")
    at = float(node["at"])
    if axis == "tau":
        if not (tlo + min_gap_tau - 1e-12 <= at <= thi - min_gap_tau + 1e-12):
            raise ValueError(f"tau cut {at} violates min-gap {min_gap_tau} inside [{tlo}, {thi}]")
        _check_tree_node(
            node.get("lo"), min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, tlo=tlo, thi=at, llo=llo, lhi=lhi
        )
        _check_tree_node(
            node.get("hi"), min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, tlo=at, thi=thi, llo=llo, lhi=lhi
        )
    else:
        if not (llo + min_gap_lam - 1e-12 <= at <= lhi - min_gap_lam + 1e-12):
            raise ValueError(f"lambda cut {at} violates min-gap {min_gap_lam} inside [{llo}, {lhi}]")
        _check_tree_node(
            node.get("lo"), min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, tlo=tlo, thi=thi, llo=llo, lhi=at
        )
        _check_tree_node(
            node.get("hi"), min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, tlo=tlo, thi=thi, llo=at, lhi=lhi
        )


def _round_tree_save(tree) -> dict:
    """Deep-copy a tree with cut positions + windows rounded to 6 decimals (the saved payload)."""

    def rec(node):
        if _is_leaf(node):
            return {"leaf": True}
        out = {"axis": node["axis"], "at": round(float(node["at"]), 6), "lo": rec(node["lo"]), "hi": rec(node["hi"])}
        if node.get("sync") is not None:
            out["sync"] = node["sync"]
        if node.get("frozen"):
            out["frozen"] = True
        return out

    return {
        "window_tau": [round(float(v), 6) for v in tree["window_tau"]],
        "window_lam": [round(float(v), 6) for v in tree["window_lam"]],
        "root": rec(tree["root"]),
    }


def save_tree(tree, path) -> str:
    """Write a guillotine-tree dict {window_tau, window_lam, root} to `path` as JSON.

    Cut positions + windows are rounded to 6 decimals (well inside the min-gap + polish
    tolerances, so a saved tree scores identically when reloaded). Returns the path str.
    """
    payload = _round_tree_save(tree)
    p = Path(path).expanduser()
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return str(p)


def load_tree(path, *, min_gap_tau=MIN_GAP_TAU, min_gap_lam=MIN_GAP_LAM) -> dict:
    """Read a JSON guillotine-tree dict written by `save_tree` (raises ValueError when the
    file is not valid JSON, misses window_tau/window_lam/root, carries non-finite edges,
    or has an axis/cut outside tau|lam / the min-gap box)."""
    try:
        raw = Path(path).expanduser().read_text()
    except OSError as e:
        raise ValueError(f"cannot read tree file {path}: {e}")
    try:
        tree = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"tree file {path} is not valid JSON: {e}")
    if not isinstance(tree, dict):
        raise ValueError(f"tree file {path} must hold a JSON object with window_tau/window_lam/root")
    for key in ("window_tau", "window_lam", "root"):
        if key not in tree:
            raise ValueError(f"tree file {path} misses key {key!r}")
    for key in ("window_tau", "window_lam"):
        win = tree[key]
        if (
            not isinstance(win, (list, tuple))
            or len(win) != 2
            or not all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in win)
            or not float(win[0]) < float(win[1])
        ):
            raise ValueError(f"tree file {path}: {key} must be 2 increasing finite numbers, got {win!r}")
    if not isinstance(tree["root"], dict):
        raise ValueError(f"tree file {path}: root must be an object")
    tlo, thi = float(tree["window_tau"][0]), float(tree["window_tau"][1])
    llo, lhi = float(tree["window_lam"][0]), float(tree["window_lam"][1])
    _check_tree_node(tree["root"], min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, tlo=tlo, thi=thi, llo=llo, lhi=lhi)
    if not _tree_feasible(tree, min_gap_tau, min_gap_lam):
        raise ValueError(f"tree file {path}: leaf rectangles violate the min-gap box")
    return tree


def _lam_chain(lam_edges):
    """Right-leaning lambda chain over interior cuts; a bare leaf when there is no split."""
    node = {"leaf": True}
    for c in reversed([float(v) for v in lam_edges[1:-1]]):
        node = {"axis": "lam", "at": c, "lo": {"leaf": True}, "hi": node}
    return node


def _tau_chain(tau_edges):
    """Right-leaning tau chain over interior cuts; a bare leaf when there is no split."""
    node = {"leaf": True}
    for c in reversed([float(v) for v in tau_edges[1:-1]]):
        node = {"axis": "tau", "at": c, "lo": {"leaf": True}, "hi": node}
    return node


def tree_from_splits(tau_window, lam_window, splits, *, min_gap_tau=MIN_GAP_TAU, min_gap_lam=MIN_GAP_LAM) -> dict:
    """Build a guillotine tree from an ordered split list over a bounding box.

    `tau_window`/`lam_window` are the outer [lo, hi] edges (-log10 tau / log10 lambda).
    Each split is {"axis": "tau"|"lam", "tau": float, "lam": float}: the cut runs at the
    point's coordinate on `axis` and spans the single existing leaf (bin) containing the
    point — i.e. it stops at the box or at an earlier cross-axis cut. Splits apply in
    order, so deleting an early split reshapes every later one. Example: box tau [-1, 7]
    x lam [3, 5], tau splits at 0 and 2 (full-height), then {"axis": "lam", "tau": 1.0,
    "lam": 3.5} cuts only the tau-[0, 2] bin at lam 3.5.

    Raises ValueError when a window is degenerate, a point falls outside the box, or a
    cut lands within min-gap of its leaf's border on the cut axis.
    """
    try:
        tlo, thi = (float(tau_window[0]), float(tau_window[1]))
        llo, lhi = (float(lam_window[0]), float(lam_window[1]))
    except (TypeError, IndexError, ValueError):
        raise ValueError(f"tau/lambda windows must each hold 2 edges, got {tau_window!r} / {lam_window!r}")
    if not (tlo < thi):
        raise ValueError(f"tau window must be increasing, got {[tlo, thi]}")
    if not (llo < lhi):
        raise ValueError(f"lambda window must be increasing, got {[llo, lhi]}")

    root: dict = {"leaf": True}

    def _leaf_at(node, rect, tau, lam):
        # Walk to the leaf containing (tau, lam); cut points land in the hi child
        # (the digitize(right=False) convention assign_tree uses).
        while not _is_leaf(node):
            t0, t1, l0, l1 = rect
            at = float(node["at"])
            if node["axis"] == "tau":
                node, rect = (node["lo"], (t0, at, l0, l1)) if tau < at else (node["hi"], (at, t1, l0, l1))
            elif node["axis"] == "lam":
                node, rect = (node["lo"], (t0, t1, l0, at)) if lam < at else (node["hi"], (t0, t1, at, l1))
            else:
                raise ValueError(f"unknown tree axis {node['axis']!r}; expected 'tau' or 'lam'")
        return node, rect

    for i, s in enumerate(splits or []):
        try:
            axis, tau, lam = s["axis"], float(s["tau"]), float(s["lam"])
        except (TypeError, KeyError, ValueError):
            raise ValueError(f"split {i}: need {{axis, tau, lam}}, got {s!r}")
        if axis not in ("tau", "lam"):
            raise ValueError(f"split {i}: axis must be 'tau' or 'lam', got {axis!r}")
        if not (tlo <= tau <= thi and llo <= lam <= lhi):
            raise ValueError(f"split {i}: point (tau={tau}, lam={lam}) falls outside the box")
        leaf, (t0, t1, l0, l1) = _leaf_at(root, (tlo, thi, llo, lhi), tau, lam)
        if axis == "tau":
            pos, lo, hi, mg = tau, t0, t1, min_gap_tau
        else:
            pos, lo, hi, mg = lam, l0, l1, min_gap_lam
        if pos - lo < mg - 1e-12 or hi - pos < mg - 1e-12:
            raise ValueError(f"split {i}: {axis} cut at {pos} is within min_gap ({mg}) of its bin edge [{lo}, {hi}]")
        leaf.clear()
        leaf.update({"axis": axis, "at": pos, "lo": {"leaf": True}, "hi": {"leaf": True}})

    return {"window_tau": [tlo, thi], "window_lam": [llo, lhi], "root": root}


def _resnap(cols, min_gap):
    """Re-seat rounded per-column edges to min-gap in place (endpoints pinned).

    Rounding to 4 decimals can collapse two cuts closer than ~5e-5 apart into a
    non-increasing pair; push each interior cut up to prev + min_gap so the columns
    stay strictly increasing without moving the shared outer window.
    """
    for col in cols:
        for j in range(1, len(col) - 1):
            if col[j] - col[j - 1] < min_gap - 1e-12:
                col[j] = col[j - 1] + min_gap


def tree_from_columns(lambda_edges, tau_edges_per_lambda) -> dict:
    """Convert a lambda-column binning to a guillotine tree (lambda cuts at the root,
    a tau chain per column). Each column carries its OWN tau edges (independent counts
    allowed); all columns share the outer [tlo, thi] tau window. Leaves enumerate in
    lambda-major DFS order (column 0's tau stack first)."""
    lam = [float(e) for e in lambda_edges]
    if len(lam) < 2:
        raise ValueError(f"need >= 2 lambda edges, got {lam}")
    cols = [[float(v) for v in e] for e in tau_edges_per_lambda]
    if len(cols) != len(lam) - 1:
        raise ValueError(f"{len(cols)} tau columns for {len(lam) - 1} lambda cells")
    tlo = cols[0][0]
    thi = cols[0][-1]
    for k, e in enumerate(cols):
        if len(e) < 2:
            raise ValueError(f"column {k}: need >= 2 tau edges, got {e}")
        if any(e[i] >= e[i + 1] for i in range(len(e) - 1)):
            raise ValueError(f"column {k}: tau edges must be strictly increasing, got {e}")
        if e[0] != tlo or e[-1] != thi:
            raise ValueError(f"column {k}: outer tau window {e[0], e[-1]} must match column 0 {[tlo, thi]}")
    if any(lam[i] >= lam[i + 1] for i in range(len(lam) - 1)):
        raise ValueError(f"lambda edges must be strictly increasing, got {lam}")
    chains = [_tau_chain(e) for e in cols]
    node = chains[-1]
    for k in range(len(lam) - 2, 0, -1):  # interior lambda cuts only; lam[0]/lam[-1] are the window
        node = {"axis": "lam", "at": lam[k], "lo": chains[k - 1], "hi": node}
    return {"window_tau": [tlo, thi], "window_lam": [lam[0], lam[-1]], "root": node}


def tree_from_lpt(tau_edges, lambda_edges_per_tau) -> dict:
    """Convert a shared-tau + per-tau-group-lambda binning to an equivalent guillotine tree
    (tau cuts at the root, a lambda chain per band) — same rectangle set AND DFS order as
    build_group_specs_per_tau, so it scores identically."""
    tau = [float(e) for e in tau_edges]
    lmin, lmax = float(lambda_edges_per_tau[0][0]), float(lambda_edges_per_tau[0][-1])
    bands = [_lam_chain(x) for x in lambda_edges_per_tau]
    node = bands[-1]
    for k in range(len(tau) - 2, 0, -1):  # interior tau cuts only (tau[1..n-1]); tau[0]/tau[-1] are the window
        node = {"axis": "tau", "at": tau[k], "lo": bands[k - 1], "hi": node}
    return {"window_tau": [tau[0], tau[-1]], "window_lam": [lmin, lmax], "root": node}


def _synced_groups(tree) -> dict[str, list]:
    """Map sync id -> member internal nodes (nodes sharing a `sync` id move as one cut)."""
    groups: dict[str, list] = {}
    for node, _lo, _hi in _iter_internal(tree["root"], _root_rect(tree)):
        if node.get("sync"):
            groups.setdefault(str(node["sync"]), []).append(node)
    return groups


def _tree_position_search(tree, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam):
    """Gauss-Seidel coordinate descent on every internal cut's position. Each candidate must
    stay in its node's axis span (per-axis min gap) and keep the whole tree feasible; the best
    strict improvement per node is committed in place."""
    best = cost_tree(tree)
    synced = _synced_groups(tree)  # sync id -> members; joint moves keep them equal
    for _ in range(cfg.max_sweeps):
        improved = False
        spans = {id(n): (lo, hi) for n, lo, hi in _iter_internal(tree["root"], _root_rect(tree))}
        for node, _span_lo, _span_hi in list(_iter_internal(tree["root"], _root_rect(tree))):
            if node.get("frozen"):
                continue  # staged seed cut: position fixed, never wiggled again
            members = synced.get(str(node.get("sync", "")), None) if node.get("sync") else None
            if members is not None and len(members) > 1 and members[0] is not node:
                continue  # moved jointly with the group's first node
            group = members if members and len(members) > 1 else [node]
            mg = min_gap_tau if node["axis"] == "tau" else min_gap_lam
            old = float(node["at"])
            best_at, best_c = old, best
            for step in cfg.adjust_steps:
                for d in (-1.0, +1.0):
                    if budget.exhausted():
                        for m in group:
                            m["at"] = best_at
                        return tree, best_c
                    cand = old + d * step
                    if any(cand <= spans[id(m)][0] + mg - 1e-12 or cand >= spans[id(m)][1] - mg + 1e-12 for m in group):
                        continue
                    for m in group:
                        m["at"] = cand
                    if not _tree_feasible(tree, min_gap_tau, min_gap_lam):
                        continue
                    c = cost_tree(tree)
                    if c < best_c - 1e-12:
                        best_c, best_at = c, cand
            for m in group:
                m["at"] = best_at
            if best_c < best - 1e-12:
                best, improved = best_c, True
        if not improved:
            break
    return tree, best


def _block_fixed_point_tree(tree, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam, report=None):
    best = cost_tree(tree)
    for _ in range(cfg.max_block_rounds):
        start = best
        tree, best = _tree_position_search(
            tree, cost_tree, cfg=cfg, budget=budget, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam
        )
        if report:
            report("tree", best, _n_leaves(tree))
        if budget.exhausted() or (start - best) <= cfg.block_tol * max(abs(start), 1.0):
            break
    return tree, best


def _refine_node(tree, path, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam, best=None):
    """Coordinate-descend a SINGLE internal node's cut position. Cheap refine used while growing
    (the rest of the tree was already refined, so re-sweeping every node per candidate is wasteful
    and — at RTE cost — eats the whole budget before the tree can grow)."""
    node = _node_at_path(tree["root"], path)
    if node.get("frozen"):
        return tree, best if best is not None else cost_tree(tree)  # staged seed cut: fixed
    group = _synced_groups(tree).get(str(node.get("sync", "")), [node]) if node.get("sync") else [node]
    group = [m for m in group if "at" in m] or [node]
    spans = {id(n): (lo, hi) for n, lo, hi in _iter_internal(tree["root"], _root_rect(tree))}
    if any(id(m) not in spans for m in group):
        return tree, best if best is not None else cost_tree(tree)
    mg = min_gap_tau if node["axis"] == "tau" else min_gap_lam
    best_c = best if best is not None else cost_tree(tree)
    best_at = float(node["at"])
    for step in cfg.adjust_steps:
        old = best_at
        for d in (-1.0, +1.0):
            if budget.exhausted():
                for m in group:
                    m["at"] = best_at
                return tree, best_c
            cand = old + d * step
            if any(cand <= spans[id(m)][0] + mg - 1e-12 or cand >= spans[id(m)][1] - mg + 1e-12 for m in group):
                continue
            for m in group:
                m["at"] = cand
            if not _tree_feasible(tree, min_gap_tau, min_gap_lam):
                continue
            c = cost_tree(tree)
            if c < best_c - 1e-12:
                best_c, best_at = c, cand
    for m in group:
        m["at"] = best_at
    return tree, best_c


def _grow_tree(tree, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam, max_try=6, light=True):
    """Structural grow: among the widest few leaves, split each on tau AND on lambda at its
    midpoint, refine positions, and return the (tree, cost) of the best split found. None if no
    leaf can be split within the min gap. With ``light`` only the new cut is refined (cheap);
    otherwise the whole tree is re-refined (slow, RTE-bound).

    max_try must be wide enough that a lambda sub-cell (half the area of a full-lambda band, so it
    never ranks among the top few widest) is still eligible for a tau sub-split — otherwise a
    general-2D split (tau boundary that differs per lambda region) is never even evaluated."""
    leaves = sorted(
        _iter_leaves_with_path(tree["root"], _root_rect(tree)),
        key=lambda pr: (pr[1][1] - pr[1][0]) * (pr[1][3] - pr[1][2]),
        reverse=True,
    )
    best_tree, best_c = None, float("inf")
    tried = 0
    for path, (tlo, thi, llo, lhi) in leaves:
        if tried >= max_try or budget.exhausted():
            break
        cands = []
        if (thi - tlo) >= 2 * min_gap_tau:
            cands.append(("tau", 0.5 * (tlo + thi)))
        if (lhi - llo) >= 2 * min_gap_lam:
            cands.append(("lam", 0.5 * (llo + lhi)))
        if not cands:
            continue
        tried += 1
        for axis, mid in cands:
            if budget.exhausted():
                break
            cand = copy.deepcopy(tree)
            leaf = _node_at_path(cand["root"], path)
            leaf.clear()
            leaf.update({"axis": axis, "at": mid, "lo": {"leaf": True}, "hi": {"leaf": True}})
            if not _tree_feasible(cand, min_gap_tau, min_gap_lam):
                continue
            if light:
                cand, c = _refine_node(
                    cand, path, cost_tree, cfg=cfg, budget=budget, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam
                )
            else:
                cand, c = _block_fixed_point_tree(
                    cand, cost_tree, cfg=cfg, budget=budget, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam
                )
            if c < best_c:
                best_tree, best_c = cand, c
    return (best_tree, best_c) if best_tree is not None else None


def _tree_signature(tree):
    """Canonical signature of a tree's leaf-rectangle set (rounded, sorted). Two tilings
    with the same signature score identically, so beam search can drop duplicates."""
    rects = list(_leaf_rects(tree["root"], _root_rect(tree)))
    return tuple(sorted((round(tlo, 3), round(thi, 3), round(llo, 3), round(lhi, 3)) for tlo, thi, llo, lhi in rects))


def _beam_grow_tree(
    tree,
    cost_tree,
    *,
    cfg,
    budget,
    min_gap_tau,
    min_gap_lam,
    max_groups,
    beam_width=3,
    beam_positions=(0.35, 0.5, 0.65),
    beam_leaves=4,
    grow_tol=0.0,
    explore=0.0,  # early exploration: first `explore_rounds` grow rounds also continue when
    explore_rounds=0,  # the shortfall vs grow_tol stays within this fraction (jump around early)
    report=None,
):
    """Non-greedy beam search over guillotine-tree topologies (``beam_width >= 2``).

    Maintains up to ``beam_width`` candidate trees in parallel. Each round expands every
    survivor by splitting its widest few leaves on tau and/or lambda at several positions
    (not only the midpoint), scores each child at its split position, and keeps the best
    ``beam_width`` *distinct* tilings (deduped by leaf-rectangle signature). Stops when a
    round's best child fails to beat the running best by more than ``grow_tol``, or when the
    leaf cap / budget is hit.

    Unlike greedy ``_grow_tree`` — which tries only midpoint splits and commits the single
    best immediately, pruning every alternative — beam keeps rival topologies alive so an
    unfavourable early split can't lock out a better one. It is a genuine (beam-bounded)
    exploration of the tiling space rather than one greedy trajectory.

    Child positions are scored raw during the search (no per-child refine); the caller's
    final ``_block_fixed_point_tree`` polish optimizes the winner's cut positions. Returns
    the single best ``(tree, cost)`` seen across all rounds.
    """
    best_tree, best_c = copy.deepcopy(tree), cost_tree(tree)
    beam = [(copy.deepcopy(tree), best_c)]
    explore_used = 0
    while _n_leaves(best_tree) < max_groups and not budget.exhausted():
        children = []  # (signature, tree, cost)
        for btree, _bc in beam:
            cands = sorted(
                _iter_leaves_with_path(btree["root"], _root_rect(btree)),
                key=lambda pr: (pr[1][1] - pr[1][0]) * (pr[1][3] - pr[1][2]),
                reverse=True,
            )[:beam_leaves]
            for path, (tlo, thi, llo, lhi) in cands:
                for axis, lo, hi, mg in (("tau", tlo, thi, min_gap_tau), ("lam", llo, lhi, min_gap_lam)):
                    if (hi - lo) < 2 * mg:
                        continue
                    for f in beam_positions:
                        if budget.exhausted():
                            break
                        pos = lo + f * (hi - lo)
                        if pos - lo < mg - 1e-12 or hi - pos < mg - 1e-12:
                            continue
                        cand = copy.deepcopy(btree)
                        leaf = _node_at_path(cand["root"], path)
                        leaf.clear()
                        leaf.update({"axis": axis, "at": pos, "lo": {"leaf": True}, "hi": {"leaf": True}})
                        if not _tree_feasible(cand, min_gap_tau, min_gap_lam):
                            continue
                        children.append((_tree_signature(cand), cand, cost_tree(cand)))
        if not children:
            break
        children.sort(key=lambda t: t[2])
        new_beam, seen = [], set()
        for sig, ct, cc in children:
            if sig in seen:
                continue
            seen.add(sig)
            new_beam.append((ct, cc))
            if len(new_beam) >= beam_width:
                break
        round_best_tree, round_best_c = new_beam[0]
        if report:
            report("beam", round_best_c, _n_leaves(round_best_tree))
        shortfall = best_c - round_best_c
        if shortfall <= 0 and explore_used < explore_rounds:
            # Early exploration (checked BEFORE the grow bar): adopt a non-improving round
            # winner to jump out of the local basin, provided it is within `explore` fraction
            # of the running cost. The leaf cap + budget still bound the walk.
            if (round_best_c - best_c) / max(abs(best_c), 1.0) <= explore:
                explore_used += 1
                best_tree, best_c = copy.deepcopy(round_best_tree), round_best_c
                beam = new_beam
                continue
        if shortfall <= grow_tol:
            break  # this round didn't beat the running best past the threshold -> converged
        best_tree, best_c = copy.deepcopy(round_best_tree), round_best_c
        beam = new_beam
    return best_tree, best_c


def _remove_node_at_path(root, path):
    """Collapse the internal node at `path` into a leaf (its two child leaves merge into one group)."""
    n = _node_at_path(root, path)
    n.clear()
    n["leaf"] = True


def _iter_removable_with_path(node, rect, path=()):
    """Internal nodes whose BOTH children are leaves -- removing one merges two leaves into one
    (net -1 leaf). Yields (path, rect)."""
    if _is_leaf(node):
        return
    tlo, thi, llo, lhi = rect
    at = float(node["at"])
    if node["axis"] == "tau":
        rlo, rhi = (tlo, at, llo, lhi), (at, thi, llo, lhi)
    else:
        rlo, rhi = (tlo, thi, llo, at), (tlo, thi, at, lhi)
    if _is_leaf(node["lo"]) and _is_leaf(node["hi"]):
        yield (path, rect)
    yield from _iter_removable_with_path(node["lo"], rlo, (*path, "lo"))
    yield from _iter_removable_with_path(node["hi"], rhi, (*path, "hi"))


def _topology_search(tree, cost_tree, *, cfg, budget, max_groups, min_gap_tau, min_gap_lam, report=None):
    """Greedy topology local search with per-candidate position polish.

    The beam/greedy grow scores split candidates at RAW positions, so a lambda cut -- which only
    pays off once the tau structure is in place and the edges are tuned -- is pruned early and
    the search can converge to a tau-only (or lambda-only) basin. This pass escapes it: from the
    converged tree it tries STRUCTURAL moves, each polished to its position optimum before
    scoring:
      * SPLIT: cut any leaf on tau OR lambda (when under the leaf cap);
      * REALLOC: drop a redundant cut (a node above two leaves) and re-cut a leaf, typically on
        the other axis -- same leaf budget, different topology (e.g. trade a redundant tau-group
        for a lambda split of the photospheric group).
    Each candidate is cheaply pre-filtered by refining only its new cut (`_refine_node`); only
    promising ones get a full `_block_fixed_point_tree` polish. First-improvement greedy, restart
    after each adoption; bounded by `budget`; never increases the cost.
    """
    positions = (0.35, 0.5, 0.65)
    state = {"tree": copy.deepcopy(tree), "c": cost_tree(tree)}

    def adopt_if_better(cand, new_path):
        # Cheap pre-filter: refine ONLY the newly added cut against the running best.
        cand, c = _refine_node(
            cand,
            new_path,
            cost_tree,
            cfg=cfg,
            budget=budget,
            min_gap_tau=min_gap_tau,
            min_gap_lam=min_gap_lam,
            best=state["c"],
        )
        if c >= state["c"] - 1e-9 or budget.exhausted():
            return False
        # Promising -> full position polish, then adopt if it still wins.
        cand, c = _block_fixed_point_tree(
            cand, cost_tree, cfg=cfg, budget=budget, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam
        )
        if c < state["c"] - 1e-9:
            state["tree"], state["c"] = cand, c
            if report:
                report("topo", c, _n_leaves(cand))
            return True
        return False

    def try_splits(base):
        """Try splitting every leaf on both axes; adopt on the first improvement. Returns True if any."""
        for path, (tlo, thi, llo, lhi) in list(_iter_leaves_with_path(base["root"], _root_rect(base))):
            if budget.exhausted():
                return False
            for axis, lo, hi, mg in (("tau", tlo, thi, min_gap_tau), ("lam", llo, lhi, min_gap_lam)):
                if (hi - lo) < 2 * mg:
                    continue
                for f in positions:
                    if budget.exhausted():
                        return False
                    pos = lo + f * (hi - lo)
                    if pos - lo < mg - 1e-12 or hi - pos < mg - 1e-12:
                        continue
                    cand = copy.deepcopy(base)
                    leaf = _node_at_path(cand["root"], path)
                    leaf.clear()
                    leaf.update({"axis": axis, "at": pos, "lo": {"leaf": True}, "hi": {"leaf": True}})
                    if not _tree_feasible(cand, min_gap_tau, min_gap_lam):
                        continue
                    if adopt_if_better(cand, path):
                        return True
        return False

    improved = True
    while improved and not budget.exhausted():
        improved = False
        # SPLIT (grow a leaf on either axis) when there is room under the cap.
        if _n_leaves(state["tree"]) < max_groups and try_splits(state["tree"]):
            improved = True
            continue
        # REALLOC: remove each redundant cut, then re-split (often on the other axis).
        for rmpath, _rr in list(_iter_removable_with_path(state["tree"]["root"], _root_rect(state["tree"]))):
            if budget.exhausted():
                break
            if _node_at_path(state["tree"]["root"], rmpath).get("sync"):
                continue  # hard sync: never remove one synced cut without the other
            if _node_at_path(state["tree"]["root"], rmpath).get("frozen"):
                continue  # staged seed cut: never remove the fixed skeleton
            base = copy.deepcopy(state["tree"])
            _remove_node_at_path(base["root"], rmpath)  # merge two leaves -> frees one leaf slot
            if try_splits(base):
                improved = True
                break
    return state["tree"], state["c"]


def staged_seed_tree(tau_window, lam_window, n_initial_tau_bins, *, min_gap_tau=MIN_GAP_TAU) -> dict:
    """Seed the staged path: `n_initial_tau_bins` equally-spaced tau groups over the outer
    tau window (full lambda width, no lambda cuts). Raises ValueError below 1 bin or when
    the equal spacing violates `min_gap_tau`."""
    n = int(n_initial_tau_bins)
    # Best τ-only 3-bin split from exhaustive dτ=0.5 enumeration on G2_1D.dat
    # (rms=1.9933e8; runner-up also at 1.87 + second cut at 2.37): narrow photospheric
    # band [1.87, 2.87] instead of uniform spacing.
    _BEST_TAU3 = [1.87, 2.87]
    tlo, thi = float(tau_window[0]), float(tau_window[1])
    if n < 1:
        raise ValueError(f"need >= 1 initial tau bin, got {n_initial_tau_bins!r}")
    if n == 3 and tlo <= _BEST_TAU3[0] and _BEST_TAU3[-1] <= thi:
        cuts = [_BEST_TAU3[0], _BEST_TAU3[1]]  # optimal τ-only 3-bin seed, not uniform
    else:
        step = (thi - tlo) / n
        if step < min_gap_tau - 1e-12:
            raise ValueError(f"{n} initial tau bins over [{tlo}, {thi}] need spacing {min_gap_tau}, got {step}")
        cuts = [round(tlo + step * k, 4) for k in range(1, n)]
    splits = [{"axis": "tau", "tau": t, "lam": 0.5 * (float(lam_window[0]) + float(lam_window[1]))} for t in cuts]
    return tree_from_splits(list(tau_window), list(lam_window), splits, min_gap_tau=min_gap_tau)


def _tau_cuts_sorted(tree) -> list[float]:
    """Sorted interior tau-cut positions of a tau-only tree (DFS collect, then sort)."""
    cuts: list[float] = []

    def walk(node):
        if _is_leaf(node):
            return
        if node.get("axis") == "tau":
            cuts.append(float(node["at"]))
        walk(node["lo"])
        walk(node["hi"])

    walk(tree["root"])
    return sorted(cuts)


def _tau_only_tree(tau_window, lam_window, cuts) -> dict:
    """Tau-only guillotine tree (full lambda width) over `cuts` — the scan's candidate shape."""
    tlo, thi = float(tau_window[0]), float(tau_window[1])
    llo, lhi = float(lam_window[0]), float(lam_window[1])
    return {
        "window_tau": [tlo, thi],
        "window_lam": [llo, lhi],
        "root": _tau_chain([tlo, *sorted(float(c) for c in cuts), thi]),
    }


def scan_tau_seeds(
    n,
    tau_window,
    lam_window,
    n_bins,
    *,
    min_gap_tau=MIN_GAP_TAU,
    min_gap_lam=MIN_GAP_LAM,
    dtau=0.5,
    n_keep=5,
    cost_tree,
    budget=None,
    rng=None,
) -> list:
    """Latin-hypercube seed scan over tau-cut sets.

    LHS-samples `n` cut sets (`n_bins - 1` cuts each) on the `dtau` grid over `tau_window`,
    scores each raw via `cost_tree` on a tau-only tree (full lambda width), keeps the best
    `n_keep` pairwise-separated by >= `min_gap_tau` in every cut (greedy: sort by cost, take
    if separated from all taken), light-polishes each kept seed with a single-round
    `_block_fixed_point_tree` (the winner is fully polished downstream), and returns
    [(cost, tree)] best-first. Scoring is fully via the injected `cost_tree`, so this is
    data-free testable with an analytic objective.
    """
    n, n_bins, n_keep = int(n), int(n_bins), int(n_keep)
    if n <= 0 or n_keep <= 0:
        return []
    if n_bins < 1:
        raise ValueError(f"need >= 1 tau bin, got {n_bins!r}")
    if cost_tree is None:
        raise ValueError("scan_tau_seeds needs cost_tree(tree) -> cost")
    tlo, thi = float(tau_window[0]), float(tau_window[1])
    if not thi > tlo:
        raise ValueError(f"tau window must be increasing, got {[tlo, thi]}")
    k = n_bins - 1
    grid = _grid_points(tlo, thi, float(dtau))[1:-1]  # interior dtau points; endpoints never move
    if k > 0 and not grid:
        return []
    gen = np.random.default_rng(rng)
    mg = float(min_gap_tau)

    def _feasible(cuts) -> bool:
        span = [tlo, *cuts, thi]
        return all(b > a and b - a >= mg - 1e-12 for a, b in zip(span, span[1:]))

    seen: set[tuple] = set()
    scored: list[tuple[float, tuple, dict]] = []
    for _ in range(25):
        if len(seen) >= n or (budget is not None and budget.exhausted()):
            break
        strata = np.empty((n, k))
        for j in range(k):
            strata[:, j] = (gen.permutation(n) + gen.random(n)) / n
        for i in range(n):
            if len(seen) >= n:
                break
            snapped = (
                sorted(min(grid, key=lambda g, s=s: abs(g - (tlo + s * (thi - tlo)))) for s in strata[i]) if k else []
            )
            key = tuple(snapped)
            if key in seen or not _feasible(snapped):
                continue
            seen.add(key)
            tree = _tau_only_tree([tlo, thi], lam_window, snapped)
            scored.append((float(cost_tree(tree)), key, tree))
    scored.sort(key=lambda s: s[0])
    kept = []
    for cost, key, tree in scored:
        if len(kept) >= n_keep:
            break
        if k and any(min(abs(a - b) for a, b in zip(key, other)) < mg - 1e-9 for _, other, _ in kept):
            continue
        kept.append((cost, key, tree))
    tight = _Cfg(min_gap_tau=mg, min_gap_lam=min_gap_lam, max_sweeps=1, max_block_rounds=1)
    use_budget = (
        budget
        if budget is not None
        else _Budget(max_evals=10**9, max_seconds=3600.0, state={"n_evals": 0}, t0=time.perf_counter())
    )
    polished = []
    for cost, _key, tree in kept:
        t = copy.deepcopy(tree)
        t, c = _block_fixed_point_tree(
            t, cost_tree, cfg=tight, budget=use_budget, min_gap_tau=mg, min_gap_lam=min_gap_lam
        )
        polished.append((c, t))
    polished.sort(key=lambda s: s[0])
    final = []
    for c, t in polished:
        a = _tau_cuts_sorted(t)
        if k and any(min(abs(x - y) for x, y in zip(a, _tau_cuts_sorted(u))) < mg - 1e-9 for _, u in final):
            continue  # polish nudged two seeds together; keep the cheaper one
        final.append((c, t))
    return final


def _bottom_columns_tree(tau_window, lam_window, t_top, lam_at, t_lo, t_hi) -> dict:
    """Flipped bottom-region tree: tau@`t_top` root (top leaf spans the full lambda width),
    bottom region lam@`lam_at` with a per-column tau cut (`t_lo` / `t_hi`) -- 5 leaves.
    Explicit nested dict (no point location); the caller checks `_tree_feasible`."""
    tlo, thi = float(tau_window[0]), float(tau_window[1])
    llo, lhi = float(lam_window[0]), float(lam_window[1])
    return {
        "window_tau": [tlo, thi],
        "window_lam": [llo, lhi],
        "root": {
            "axis": "tau",
            "at": float(t_top),
            "lo": {
                "axis": "lam",
                "at": float(lam_at),
                "lo": {"axis": "tau", "at": float(t_lo), "lo": {"leaf": True}, "hi": {"leaf": True}},
                "hi": {"axis": "tau", "at": float(t_hi), "lo": {"leaf": True}, "hi": {"leaf": True}},
            },
            "hi": {"leaf": True},
        },
    }


def _bottom_triple(tree) -> tuple:
    """(L, t_lo, t_hi) of a `_bottom_columns_tree` tiling (polish preserves the topology)."""
    bot = tree["root"]["lo"]
    return (float(bot["at"]), float(bot["lo"]["at"]), float(bot["hi"]["at"]))


def scan_bottom_columns(
    n,
    tau_window,
    lam_window,
    t_top,
    *,
    min_gap_tau=MIN_GAP_TAU,
    min_gap_lam=MIN_GAP_LAM,
    dtau=0.5,
    dlam=0.25,
    n_keep=3,
    cost_tree,
    budget=None,
    rng=None,
) -> list:
    """Latin-hypercube scan over flipped bottom-region tilings.

    LHS-samples `n` (L, t_lo, t_hi) triples -- L on the `dlam` interior grid of
    `lam_window`, t_lo/t_hi on the `dtau` interior grid of [tlo_outer, `t_top`] -- scores
    each raw via `cost_tree` on the 5-leaf flipped tree (`_bottom_columns_tree`), keeps the
    best `n_keep` OR-separated (separated iff dL >= min_gap_lam OR d_tlo >= min_gap_tau OR
    d_thi >= min_gap_tau from every kept triple), light-polishes each kept triple with a
    single-round `_block_fixed_point_tree` (the winner is fully polished downstream) with
    the top cut frozen, and returns [(cost, tree)] best-first. Scoring is fully via the
    injected `cost_tree`, so this is data-free testable with an analytic objective.
    """
    n, n_keep = int(n), int(n_keep)
    if n <= 0 or n_keep <= 0:
        return []
    if cost_tree is None:
        raise ValueError("scan_bottom_columns needs cost_tree(tree) -> cost")
    tlo, thi = float(tau_window[0]), float(tau_window[1])
    llo, lhi = float(lam_window[0]), float(lam_window[1])
    t_top = float(t_top)
    mg_tau, mg_lam = float(min_gap_tau), float(min_gap_lam)
    lam_grid = _grid_points(llo, lhi, float(dlam))[1:-1]
    tau_grid = _grid_points(tlo, t_top, float(dtau))[1:-1] if t_top > tlo else []
    if not lam_grid or not tau_grid:
        return []
    if (t_top - tlo) < mg_tau - 1e-12 or (thi - t_top) < mg_tau - 1e-12:
        return []  # top cut itself infeasible: every triple would be
    if (lhi - llo) < 2 * mg_lam - 1e-12:
        return []

    def _feasible(triple) -> bool:
        L, a, b = triple
        return (
            L - llo >= mg_lam - 1e-12
            and lhi - L >= mg_lam - 1e-12
            and a - tlo >= mg_tau - 1e-12
            and t_top - a >= mg_tau - 1e-12
            and b - tlo >= mg_tau - 1e-12
            and t_top - b >= mg_tau - 1e-12
        )

    def _clashes(triple, other) -> bool:
        # same triple under the OR-separation rule: close on ALL three axes
        return (
            abs(triple[0] - other[0]) < mg_lam - 1e-9
            and abs(triple[1] - other[1]) < mg_tau - 1e-9
            and abs(triple[2] - other[2]) < mg_tau - 1e-9
        )

    gen = np.random.default_rng(rng)
    seen: set[tuple] = set()
    scored: list[tuple[float, tuple, dict]] = []
    for _ in range(25):
        if len(seen) >= n or (budget is not None and budget.exhausted()):
            break
        strata = np.empty((n, 3))
        for j in range(3):
            strata[:, j] = (gen.permutation(n) + gen.random(n)) / n
        for i in range(n):
            if len(seen) >= n:
                break
            sL, sa, sb = (float(v) for v in strata[i])
            key = (
                min(lam_grid, key=lambda g, s=sL: abs(g - (llo + s * (lhi - llo)))),
                min(tau_grid, key=lambda g, s=sa: abs(g - (tlo + s * (t_top - tlo)))),
                min(tau_grid, key=lambda g, s=sb: abs(g - (tlo + s * (t_top - tlo)))),
            )
            if key in seen or not _feasible(key):
                continue
            seen.add(key)
            tree = _bottom_columns_tree([tlo, thi], [llo, lhi], t_top, *key)
            if not _tree_feasible(tree, mg_tau, mg_lam):
                continue
            scored.append((float(cost_tree(tree)), key, tree))
    scored.sort(key=lambda s: s[0])
    kept = []
    for cost, key, tree in scored:
        if len(kept) >= n_keep:
            break
        if any(_clashes(key, other) for _, other, _ in kept):
            continue
        kept.append((cost, key, tree))
    tight = _Cfg(min_gap_tau=mg_tau, min_gap_lam=mg_lam, max_sweeps=1, max_block_rounds=1)
    use_budget = (
        budget
        if budget is not None
        else _Budget(max_evals=10**9, max_seconds=3600.0, state={"n_evals": 0}, t0=time.perf_counter())
    )
    polished = []
    for cost, _key, tree in kept:
        t = copy.deepcopy(tree)
        freeze_tau_cuts(t, top_only=True)  # top cut frozen: the bottom scan never moves it
        t, c = _block_fixed_point_tree(
            t, cost_tree, cfg=tight, budget=use_budget, min_gap_tau=mg_tau, min_gap_lam=mg_lam
        )
        polished.append((c, t))
    polished.sort(key=lambda s: s[0])
    final = []
    for c, t in polished:
        triple = _bottom_triple(t)
        if any(_clashes(triple, _bottom_triple(u)) for _, u in final):
            continue  # polish nudged two triples together; keep the cheaper one
        final.append((c, t))
    return final


def freeze_tau_cuts(tree, *, top_only=True) -> dict:
    """Mark staged seed tau cuts `frozen`: their positions are fixed and later grow/polish/
    topology phases must neither move nor remove them. With `top_only` (default) only the
    top seed cut (highest `at`, the top bin's lower boundary) freezes -- the bottom-two
    groups stay live for the synced lambda wiggle. New cuts from grow stay unfrozen.
    Idempotent."""
    tau_nodes = [node for node, _lo, _hi in _iter_internal(tree["root"], _root_rect(tree)) if node.get("axis") == "tau"]
    if top_only:
        tau_nodes = [max(tau_nodes, key=lambda n: float(n["at"]))] if tau_nodes else []
    for node in tau_nodes:
        node["frozen"] = True
    return tree


def split_bottom_two_tau_groups(tree, lam_at, *, min_gap_tau=MIN_GAP_TAU, min_gap_lam=MIN_GAP_LAM) -> dict:
    """Lambda-split the bottom two tau groups of a tau-only tree at `lam_at` (each bottom leaf
    cut into lo/hi lambda halves). Bottom = lowest -log10 tau (first two leaves in DFS order).
    The two lambda cuts share one `sync` id so the staged wiggle moves them as a single
    synchronized cut (plus the two tau cuts) instead of drifting apart. Raises ValueError
    when the tree has < 2 leaves or the cut violates the lambda min gap."""
    out = copy.deepcopy(tree)
    leaves = list(_iter_leaves_with_path(out["root"], _root_rect(out)))
    if len(leaves) < 2:
        raise ValueError(f"need >= 2 tau groups to split the bottom two, got {len(leaves)}")
    lam_at = float(lam_at)
    for path, (_tlo, _thi, llo, lhi) in leaves[:2]:
        if lam_at - llo < min_gap_lam - 1e-12 or lhi - lam_at < min_gap_lam - 1e-12:
            raise ValueError(f"lambda cut at {lam_at} is within min_gap ({min_gap_lam}) of [{llo}, {lhi}]")
        leaf = _node_at_path(out["root"], path)
        leaf.clear()
        leaf.update({"axis": "lam", "at": lam_at, "lo": {"leaf": True}, "hi": {"leaf": True}, "sync": "bottom-lam"})
    if not _tree_feasible(out, min_gap_tau, min_gap_lam):
        raise ValueError("bottom-two lambda split violates the tau min gap")
    return out


def _synced_wiggle(tree, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam):
    """Coordinate descent over synced cut groups: nodes sharing a `sync` id move together as
    one cut (each candidate sets all members, feasibility + cost on the joint move), then all
    tau-axis cuts get an individual pass. Returns (tree, best). Used by the staged
    bottom-two step: the shared lambda cut and/or the two tau cuts move in sync."""
    best = cost_tree(tree)
    groups: dict[str, list] = {}
    for node, _lo, _hi in _iter_internal(tree["root"], _root_rect(tree)):
        if node.get("sync"):
            groups.setdefault(str(node["sync"]), []).append(node)
    for _ in range(cfg.max_sweeps):
        improved = False
        internals = list(_iter_internal(tree["root"], _root_rect(tree)))
        spans = {(id(n)): (lo, hi) for n, lo, hi in internals}
        for _sid, members in groups.items():
            mg = min_gap_lam if members[0]["axis"] == "lam" else min_gap_tau
            old = float(members[0]["at"])
            best_at, best_c = old, best
            for step in cfg.adjust_steps:
                for d in (-1.0, +1.0):
                    if budget.exhausted():
                        for m in members:
                            m["at"] = best_at
                        return tree, best_c
                    cand = old + d * step
                    ok = True
                    for m in members:
                        lo, hi = spans[id(m)]
                        if cand <= lo + mg - 1e-12 or cand >= hi - mg + 1e-12:
                            ok = False
                            break
                    if not ok:
                        continue
                    for m in members:
                        m["at"] = cand
                    if not _tree_feasible(tree, min_gap_tau, min_gap_lam):
                        continue
                    c = cost_tree(tree)
                    if c < best_c - 1e-12:
                        best_c, best_at = c, cand
            for m in members:
                m["at"] = best_at
            if best_c < best - 1e-12:
                best, improved = best_c, True
        # individual tau-cut pass (the two tau splits around the synced lambda cut)
        for node, span_lo, span_hi in list(_iter_internal(tree["root"], _root_rect(tree))):
            if node["axis"] != "tau":
                continue
            mg = min_gap_tau
            old = float(node["at"])
            best_at, best_c = old, best
            for step in cfg.adjust_steps:
                for d in (-1.0, +1.0):
                    if budget.exhausted():
                        node["at"] = best_at
                        return tree, best_c
                    cand = old + d * step
                    if cand <= span_lo + mg - 1e-12 or cand >= span_hi - mg + 1e-12:
                        continue
                    node["at"] = cand
                    if not _tree_feasible(tree, min_gap_tau, min_gap_lam):
                        continue
                    c = cost_tree(tree)
                    if c < best_c - 1e-12:
                        best_c, best_at = c, cand
            node["at"] = best_at
            if best_c < best - 1e-12:
                best, improved = best_c, True
        if not improved:
            break
    return tree, best


# --- shared search scaffolding (both optimizers) --------------------------------------
def _new_search_context(
    *,
    model,
    metric,
    empty_penalty,
    score_fn,
    on_eval,
    window,
    min_opacity_delta,
    max_evals,
    max_seconds,
    t0,
    should_stop,
    target_rms,
    plateau_evals,
    plateau_rel,
    min_gap_tau,
    min_gap_lam,
    adjust_steps,
    on_progress,
):
    """Build the shared evaluator + budget + config + history scaffolding.

    Returns (evaluate, state, budget, cfg, history, checkpoint, on_step).
    ``checkpoint(tag, r)`` records the caller-provided raw result (no re-eval).
    """

    def _record_on_eval(n, cost, r):
        # track best rms / plateau for the stopping conditions, then the caller's hook.
        budget.record(float(r["rms"]), n)  # `budget` bound below; evaluate() only runs afterward
        if on_eval is not None:
            on_eval(n, cost, r)

    evaluate, state = make_evaluator(
        model,
        metric=metric,
        empty_penalty=empty_penalty,
        score_fn=score_fn,
        on_eval=_record_on_eval,
        window=window,
        min_opacity_delta=min_opacity_delta,
    )
    budget = _Budget(
        max_evals=max_evals,
        max_seconds=max_seconds,
        state=state,
        t0=t0,
        should_stop=should_stop,
        target_rms=target_rms,
        plateau_evals=plateau_evals,
        plateau_rel=plateau_rel,
    )
    cfg = _Cfg(min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, adjust_steps=tuple(adjust_steps))

    history: list[dict] = []

    def checkpoint(tag, r):
        """Record a checkpoint with the *raw* rms (one extra eval already done by caller)."""
        history.append(
            {
                "tag": tag,
                "n_evals": state["n_evals"],
                "rms": float(r["rms"]),
                "n_empty": int(r.get("n_empty", 0)),
                "groups": int(r.get("n_groups", 0)),
            }
        )
        if on_progress:
            on_progress(tag, float(r["rms"]), int(r.get("n_groups", 0)), state["n_evals"])

    def on_step(tag, cost, groups):
        """Lightweight live line inside a long block (penalized cost, no extra eval)."""
        if on_progress:
            on_progress(tag, float(cost), int(groups), state["n_evals"])

    return evaluate, state, budget, cfg, history, checkpoint, on_step


def _grow_threshold(grow_tol, grow_tol_rel, ref):
    """Absolute grow bar: explicit `grow_tol`, else `grow_tol_rel` fraction of `ref` rms."""
    return grow_tol if grow_tol is not None else grow_tol_rel * ref


# --- public API -----------------------------------------------------------------
def optimize_qrad(
    tau_edges,
    lambda_edges,
    *,
    flags=None,
    model=None,  # atmosphere to optimize on (validated file under models/; None -> DEFAULT_MODEL)
    opt_tau=True,
    opt_lambda=True,
    opt_flags=True,
    grow=True,
    metric="rms",
    min_gap_tau=MIN_GAP_TAU,
    min_gap_lam=MIN_GAP_LAM,
    empty_penalty=EMPTY_PENALTY,
    adjust_steps=ADJUST_STEPS,
    max_groups=8,
    max_evals=400,
    max_seconds=1800.0,
    grow_tol=None,  # absolute grow threshold; if None, grow_tol_rel * current rms is used
    grow_tol_rel=0.01,  # relative grow threshold (fraction of current rms) when grow_tol is None
    window=None,  # log10 tau_Ross (lo, hi) to score rms/max_abs over (None -> qrad_core.WINDOW)
    target_rms=None,  # early stop once the best raw rms <= this
    plateau_evals=0,  # early stop if best rms hasn't improved over this many evals (0 = off)
    plateau_rel=0.005,  # plateau "improvement" threshold (fraction of the reference rms)
    per_group_lambda=False,
    lambda_edges_per_tau=None,  # per-group-lambda warm start (one lambda-edge list per tau group)
    tau_per_lambda=None,  # columns warm start (one tau-edge list per lambda column)
    splits=None,  # split-list warm start (ordered [{axis, tau, lam}] over the 2-edge windows)
    tree=False,  # general 2D guillotine mode (both tau and lambda locally free)
    binning_tree=None,  # guillotine-tree warm start {window_tau, window_lam, root}
    beam_width=3,  # rival tree topologies kept in parallel each round
    beam_positions=(0.35, 0.5, 0.65),  # split-position fractions tried per (leaf, axis)
    beam_leaves=4,  # widest leaves considered for splitting, per beam tree
    explore=0.05,  # early exploration: adopt non-improving grow rounds within this cost fraction
    explore_rounds=2,  # how many such jumps allowed before the grow bar goes strict
    min_opacity_delta=None,  # min bottom-opacity max/min ratio to split a group (None -> score default)
    initial_tau_bins=None,  # staged seeding: N equally-spaced tau cuts, tau-only polish, bottom-two lambda split
    initial_tau_scan=0,  # staged seeding: LHS-scan this many tau-cut sets first, seed from the winner (0 = off)
    staged_lambda_scan=0,  # staged seeding: LHS-scan this many (L, t_lo, t_hi) bottom triples (0 = off, keep sync path)
    staged_lambda_n_keep=3,  # bottom-triple scan: survivors polished + returned best-first
    seed=None,  # RNG seed for the LHS scans (None = fresh entropy; same seed = reproducible run)
    score_fn=None,
    on_progress=None,
    on_eval=None,
    on_improve=None,
    should_stop=None,
) -> dict:
    """Minimize the Q_rad residual over the binning. Returns a tree result dict with the
    optimized `binning_tree`, `rms`/`rms0`, `n_evals`, `elapsed`, `n_empty`, and a `history`
    of (n_evals, rms, groups) checkpoints.

    Tree-only: every input shape (an explicit `binning_tree`, `lambda_edges_per_tau`,
    `per_group_lambda`, or shared tau + `flags`) is normalized to a guillotine tree and refined
    via a single seed/grow/polish path. `grow` enables splitting leaves (accepted only when rms
    improves by > grow_tol, default an absolute 0); `beam_width >= 2` (default 3) runs a
    non-greedy beam search over tree topologies, `beam_width == 1` the greedy grow. The opt_tau/
    opt_lambda/opt_flags toggles are retained for API compatibility but are inert under the
    tree path.

    `on_eval(n_evals, cost, raw_dict)` fires after every evaluation and `should_stop() -> bool`
    is polled at each budget check — both let a caller (e.g. the webapp) show live progress
    and abort a long run gracefully, returning the best binning found so far.
    `on_improve(tree, raw_dict, n_evals)` fires whenever a strictly better binning is found (a
    new global-best penalized cost); `tree` is a deep-copied snapshot of the leaf tiling — used
    by the CLI `--plot` to render every improved binning found during the search.
    """
    tau_edges = [float(e) for e in tau_edges]
    lambda_edges = [float(e) for e in lambda_edges]
    n_tau0 = len(tau_edges) - 1
    flags = [bool(b) for b in flags] if flags is not None else [True] * n_tau0
    if len(flags) != n_tau0:
        raise ValueError(f"flags has {len(flags)} entries, expected one per tau group ({n_tau0})")
    _check_feasible(tau_edges, min_gap_tau, "tau")
    _check_feasible(lambda_edges, min_gap_lam, "lambda")

    t0 = time.perf_counter()

    evaluate, state, budget, cfg, history, checkpoint, on_step = _new_search_context(
        model=model,
        metric=metric,
        empty_penalty=empty_penalty,
        score_fn=score_fn,
        on_eval=on_eval,
        window=window,
        min_opacity_delta=min_opacity_delta,
        max_evals=max_evals,
        max_seconds=max_seconds,
        t0=t0,
        should_stop=should_stop,
        target_rms=target_rms,
        plateau_evals=plateau_evals,
        plateau_rel=plateau_rel,
        min_gap_tau=min_gap_tau,
        min_gap_lam=min_gap_lam,
        adjust_steps=adjust_steps,
        on_progress=on_progress,
    )

    # The general-2D optimizer is tree-only: every grouping (explicit guillotine tree,
    # split list, per-tau-group lambda, or shared tau + split flags) is normalized to a
    # guillotine tree and refined via a single seed/grow/polish path. (Columns inputs take
    # the separate optimize_columns coordinate-descent path.) The shim below seeds the
    # working tree from whichever input was passed; every result is a {"tree": True,
    # "binning_tree": ...} dict.

    if binning_tree is not None:
        # Highest seed precedence: an explicit (e.g. --load-tree) tree replaces every other
        # seed shape, and the staged path is skipped for it (initial_tau_bins must be None).
        btree = copy.deepcopy(binning_tree)
    elif splits is not None:
        if len(tau_edges) != 2 or len(lambda_edges) != 2:
            raise ValueError("splits mode needs exactly 2 tau edges + 2 lambda edges (the outer windows)")
        btree = tree_from_splits(tau_edges, lambda_edges, splits)
    elif lambda_edges_per_tau is not None:
        btree = tree_from_lpt(tau_edges, [list(x) for x in lambda_edges_per_tau])
        if grow and _n_leaves(btree) >= max_groups:
            # The per-group-lambda warm start can already meet/exceed the leaf cap (the webapp's
            # default is 4 tau-groups x 2 lambda-cells = 8 leaves == MAX_GROUPS). With the cap
            # saturated the grow/beam guard `_n_leaves < max_groups` is never true, so the
            # non-greedy beam search never fires and only position-polish runs. Drop the interior
            # lambda cuts -- keep the tau skeleton + lambda window, exactly as the CLI seeds -- so
            # the beam has room to re-discover lambda (and tau) splits up to max_groups. grow=False
            # (refine-only) keeps the fine seed untouched; explicit binning_tree re-runs are above.
            lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
            btree = tree_from_lpt(tau_edges, [[lmin, lmax] for _ in range(n_tau0)])
    elif per_group_lambda:
        btree = tree_from_lpt(tau_edges, [list(lambda_edges) for _ in range(n_tau0)])
    else:
        lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
        btree = tree_from_lpt(tau_edges, [list(lambda_edges) if bool(f) else [lmin, lmax] for f in flags])

    def cost_tree(t):
        # Wrapped evaluator: fires `on_improve` on every strictly better binning (new global-best
        # penalized cost). One hook covers grow, beam, and polish since they all route here.
        nonlocal _best_cost
        c, r = evaluate(binning_tree=t)
        if on_improve is not None and c < _best_cost - 1e-12:
            _best_cost = c
            on_improve(copy.deepcopy(t), r, state["n_evals"])
        return c

    _best_cost = float("inf")
    staged = initial_tau_bins is not None
    _did_scan = False
    if staged:
        # Staged seeding overrides every other seed shape: N equally-spaced tau cuts over the
        # outer tau window, tau-only position polish, then a lambda split of the bottom two tau
        # groups (at the lambda-window midpoint), then joint position polish — then the normal
        # grow/polish/topology path below.
        if int(initial_tau_bins) > max_groups:
            raise ValueError(f"--initial-tau-bins={initial_tau_bins} needs room under --max-groups={max_groups}")
        if int(initial_tau_scan) > 0:
            # LHS tau-cut scan first: the winner's tau cuts replace the uniform/optimal-fixed seed.
            seeds = scan_tau_seeds(
                int(initial_tau_scan),
                [tau_edges[0], tau_edges[-1]],
                [lambda_edges[0], lambda_edges[-1]],
                int(initial_tau_bins),
                min_gap_tau=min_gap_tau,
                cost_tree=cost_tree,
                budget=budget,
                rng=seed,
            )
            if seeds:
                btree = seeds[0][1]
                _did_scan = True
            else:
                btree = staged_seed_tree(
                    [tau_edges[0], tau_edges[-1]],
                    [lambda_edges[0], lambda_edges[-1]],
                    int(initial_tau_bins),
                    min_gap_tau=min_gap_tau,
                )
        else:
            btree = staged_seed_tree(
                [tau_edges[0], tau_edges[-1]],
                [lambda_edges[0], lambda_edges[-1]],
                int(initial_tau_bins),
                min_gap_tau=min_gap_tau,
            )

    rms0_r = evaluate(binning_tree=btree)[1]  # rms of the user's seed binning
    rms0 = float(rms0_r["rms"])

    def _refine(seed_tree):
        """grow -> polish -> topology search from one seed tree. Returns (tree, penalized cost).
        Captures cfg/budget/cost_tree/on_step/checkpoint from the enclosing scope."""
        tree = copy.deepcopy(seed_tree)
        best = cost_tree(tree)
        if not staged:  # staged path already checkpointed "start" at the seed
            checkpoint("start", evaluate(binning_tree=tree)[1])
        # Grow FIRST so the budget builds structure (each grow cheaply refines only its new cut);
        # a heavy refine of the coarse seed up front would exhaust the budget before a leaf is split.
        if grow:
            ref = budget.best_rms if budget.best_rms != float("inf") else best
            gtol = _grow_threshold(grow_tol, grow_tol_rel, ref)
            if beam_width >= 2:
                tree, best = _beam_grow_tree(
                    tree,
                    cost_tree,
                    cfg=cfg,
                    budget=budget,
                    min_gap_tau=min_gap_tau,
                    min_gap_lam=min_gap_lam,
                    max_groups=max_groups,
                    beam_width=beam_width,
                    beam_positions=tuple(beam_positions),
                    beam_leaves=beam_leaves,
                    grow_tol=gtol,
                    explore=explore,
                    explore_rounds=explore_rounds,
                    report=on_step,
                )
            else:
                while _n_leaves(tree) < max_groups and not budget.exhausted():
                    cand = _grow_tree(
                        tree,
                        cost_tree,
                        cfg=cfg,
                        budget=budget,
                        min_gap_tau=min_gap_tau,
                        min_gap_lam=min_gap_lam,
                        light=True,
                    )
                    if cand is None:
                        break
                    ctree, cbest = cand
                    if (best - cbest) > gtol:
                        tree, best = ctree, cbest
                        budget.reset_plateau(state["n_evals"])
                        checkpoint("grow", evaluate(binning_tree=tree)[1])
                    else:
                        break
        if not budget.exhausted():
            tree, best = _block_fixed_point_tree(
                tree,
                cost_tree,
                cfg=cfg,
                budget=budget,
                min_gap_tau=min_gap_tau,
                min_gap_lam=min_gap_lam,
                report=on_step,
            )
            checkpoint("blocks", evaluate(binning_tree=tree)[1])
        # Topology local search: escape the tau-only (or lambda-only) basin the grow can settle into
        # -- a lambda split of the photospheric group only pays once tau is resolved, so the raw-scored
        # grow prunes it. Structural moves (split / reallocate a cut to the other axis), each polished
        # to its position optimum; bounded by the remaining budget; never worsens. Beam/non-greedy
        # only: greedy (beam_width == 1) is the fast fallback and stays as the grow left it.
        if grow and beam_width >= 2 and not budget.exhausted():
            tree, best = _topology_search(
                tree,
                cost_tree,
                cfg=cfg,
                budget=budget,
                max_groups=max_groups,
                min_gap_tau=min_gap_tau,
                min_gap_lam=min_gap_lam,
                report=on_step,
            )
            checkpoint("topo", evaluate(binning_tree=tree)[1])
        return tree, best

    if staged and not budget.exhausted():
        checkpoint("start", rms0_r)  # seed scored first: plan order matches fire order
        if _did_scan:
            checkpoint("tau-scan", rms0_r)  # winner already scored in the scan: reuse, no extra eval
        # (2) tau-only polish of the N-bin seed, then (3) lambda-split the bottom two tau groups
        # at the lambda-window midpoint (optionally rescanned to the best synced position) +
        # joint position polish; the normal grow/polish path
        btree, _best_cost = _block_fixed_point_tree(
            btree, cost_tree, cfg=cfg, budget=budget, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, report=on_step
        )
        checkpoint("staged-tau", evaluate(binning_tree=btree)[1])
        lam_mid = 0.5 * (float(lambda_edges[0]) + float(lambda_edges[-1]))
        _did_lambda_scan = False
        if int(staged_lambda_scan) > 0 and _n_leaves(btree) + 2 <= max_groups and not budget.exhausted():
            # Flipped bottom-region scan: freeze the top cut, LHS-scan (L, t_lo, t_hi) triples over
            # the bottom region, take the winner, checkpoint staged-lambda, and skip the mid-window
            # split_bottom_two_tau_groups + _synced_wiggle path. Falls back to the sync path when
            # the scan returns no survivor.
            _top = max(_tau_cuts_sorted(btree), default=float(tau_edges[-1]))
            freeze_tau_cuts(btree, top_only=True)
            triples = scan_bottom_columns(
                int(staged_lambda_scan),
                [tau_edges[0], tau_edges[-1]],
                [lambda_edges[0], lambda_edges[-1]],
                _top,
                min_gap_tau=min_gap_tau,
                min_gap_lam=min_gap_lam,
                n_keep=int(staged_lambda_n_keep),
                cost_tree=cost_tree,
                budget=budget,
                rng=(None if seed is None else int(seed) + 1),
            )
            if triples:
                btree, _best_cost = triples[0][1], triples[0][0]
                checkpoint("staged-lambda", evaluate(binning_tree=btree)[1])
                _did_lambda_scan = True
        if not _did_lambda_scan and _n_leaves(btree) + 2 <= max_groups and not budget.exhausted():
            try:
                btree = split_bottom_two_tau_groups(btree, lam_mid, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam)
            except ValueError:
                pass  # lambda window too narrow for the cut: keep the tau-only staging
            else:
                _best_cost = cost_tree(btree)
                # Synced wiggle: the shared bottom-two lambda cut moves as one cut, then the
                # two tau cuts get an individual pass (all accepted on joint improvement only).
                btree, _best_cost = _synced_wiggle(
                    btree,
                    cost_tree,
                    cfg=cfg,
                    budget=budget,
                    min_gap_tau=min_gap_tau,
                    min_gap_lam=min_gap_lam,
                )
                checkpoint("staged-lambda", evaluate(binning_tree=btree)[1])
        freeze_tau_cuts(btree, top_only=True)  # top seed cut fixed: grow/polish/topo never move/remove it
    btree, _best_cost = _refine(btree)
    final_r = evaluate(binning_tree=btree)[1]
    return {
        "binning_tree": _round_tree(btree),
        "tree": True,
        "rms": float(final_r["rms"]),
        "rms0": rms0,
        "n_empty": int(final_r.get("n_empty", 0)),
        "n_leaves": _n_leaves(btree),
        "n_bands_total": int(final_r.get("n_groups", 0)),
        "n_evals": state["n_evals"],
        "elapsed": round(time.perf_counter() - t0, 2),
        "stop_reason": budget.stop_reason or "converged",
        "window": list(final_r.get("window", [])),
        "history": history,
    }


# --- columns-constrained search --------------------------------------------------
def optimize_columns(
    lambda_edges,
    tau_per_lambda,
    *,
    model=None,  # atmosphere to optimize on (validated file under models/; None -> DEFAULT_MODEL)
    metric="rms",
    min_gap_tau=MIN_GAP_TAU,
    min_gap_lam=MIN_GAP_LAM,
    empty_penalty=EMPTY_PENALTY,
    adjust_steps=ADJUST_STEPS,
    max_groups=8,  # total-leaf cap: sum over columns of (len(edges) - 1)
    max_evals=400,
    max_seconds=1800.0,
    grow=True,  # per-column grow/prune (count changes); False = cut positions only
    grow_tol=None,  # absolute penalized-cost improvement to accept a split; None -> grow_tol_rel * best rms
    grow_tol_rel=0.01,
    window=None,  # log10 tau_Ross (lo, hi) to score rms/max_abs over (None -> qrad_core.WINDOW)
    target_rms=None,  # early stop once the best raw rms <= this
    plateau_evals=0,  # early stop if best rms hasn't improved over this many evals (0 = off)
    plateau_rel=0.005,
    min_opacity_delta=None,  # forwarded to the scorer (None -> score default)
    score_fn=None,
    on_progress=None,
    on_eval=None,
    on_improve=None,
    should_stop=None,
) -> dict:
    """Minimize the Q_rad residual while staying inside the column family.

    Decision variables: the interior lambda-cut positions (count fixed) plus each column's
    interior tau-cut positions AND count (grown/pruned per column under the `max_groups`
    total-leaf cap). The search is coordinate descent: alternate (a) a lambda-cut position
    sweep with (b) a per-column tau-stack refine (positions, then local grow by midpoint
    splits, then prune by removals, each accepted on strict penalized-cost improvement),
    looping to a local optimum or budget end. Every candidate is evaluated via
    `make_evaluator` on `tree_from_columns(...)` trees, so the empty-band penalty,
    MIN_GAP feasibility, and the shared outer tau window (column 0's [tlo, thi], endpoints
    never move) hold exactly as in `optimize_qrad` — whose grow/polish path this never
    touches (that path may leave the column family).

    Budgets/hooks mirror `optimize_qrad` so callers can swap them: `max_evals`,
    `max_seconds`, `window`, `target_rms`, `plateau_evals/plateau_rel`, `on_progress` /
    `on_eval` / `on_improve` / `should_stop`, and an injectable `score_fn` for data-free
    tests. Returns the same dict shape plus the column parametrization: `lambda_edges`,
    `tau_per_lambda`, and `binning_tree` (the columns tree, for downstream .dat/plot reuse).
    """
    lam = [float(e) for e in lambda_edges]
    cols = [[float(v) for v in c] for c in tau_per_lambda]
    if len(lam) < 2:
        raise ValueError(f"need >= 2 lambda edges, got {lam}")
    if not cols:
        raise ValueError("need >= 1 tau column, got none")
    _check_feasible(lam, min_gap_lam, "lambda")
    for k, col in enumerate(cols):
        _check_feasible(col, min_gap_tau, f"tau column {k}")
    seed_tree = tree_from_columns(lam, cols)  # raises on column-count / outer-window mismatch

    t0 = time.perf_counter()

    evaluate, state, budget, cfg, history, checkpoint, on_step = _new_search_context(
        model=model,
        metric=metric,
        empty_penalty=empty_penalty,
        score_fn=score_fn,
        on_eval=on_eval,
        window=window,
        min_opacity_delta=min_opacity_delta,
        max_evals=max_evals,
        max_seconds=max_seconds,
        t0=t0,
        should_stop=should_stop,
        target_rms=target_rms,
        plateau_evals=plateau_evals,
        plateau_rel=plateau_rel,
        min_gap_tau=min_gap_tau,
        min_gap_lam=min_gap_lam,
        adjust_steps=adjust_steps,
        on_progress=on_progress,
    )

    _hook_best = float("inf")
    _last_r: list = [None]  # raw dict of the latest evaluation (cost_current caches it)

    def cost_current():
        # Evaluate the CURRENT (lam, cols) — the only evaluation entry point, so every
        # candidate routes through tree_from_columns + the penalized evaluator.
        nonlocal _hook_best
        t = tree_from_columns(lam, cols)
        c, r = evaluate(binning_tree=t)
        _last_r[0] = r
        if on_improve is not None and c < _hook_best - 1e-12:
            _hook_best = c
            on_improve(copy.deepcopy(t), r, state["n_evals"])
        return c

    def n_leaves_now():
        return sum(len(c) - 1 for c in cols)

    def _polish_index(edges, j, mg, cur):
        """Try +-adjust_steps around edges[j] (visit-start-relative, like _tree_position_search);
        commit the best strict improvement in place; return its cost. Endpoints never move."""
        old = edges[j]
        best_at, best_c = old, cur
        for step in cfg.adjust_steps:
            for d in (-1.0, +1.0):
                if budget.exhausted():
                    edges[j] = best_at
                    return best_c
                edges[j] = old + d * step
                if not (_valid_monotone(edges) and _min_gap_ok(edges, mg)):
                    continue
                c = cost_current()
                if c < best_c - 1e-12:
                    best_c, best_at = c, edges[j]
        edges[j] = best_at
        return best_c

    def _sweep(edges, mg):
        """Gauss-Seidel position sweep over one edge list's interior cuts. Returns True if
        anything improved. Maintains the invariant best == cost of the current state."""
        nonlocal best
        if len(edges) <= 2:
            return False
        any_imp = False
        for _ in range(cfg.max_sweeps):
            if budget.exhausted():
                return any_imp
            improved = False
            for j in range(1, len(edges) - 1):
                c = _polish_index(edges, j, mg, best)
                if c < best - 1e-12:
                    best, improved, any_imp = c, True, True
            if not improved:
                break
        return any_imp

    def _grow_column(k):
        """Local grow: try a midpoint split of every interval of column k (light-polished),
        commit the best one past the grow threshold; repeat until none qualifies or the
        total-leaf cap / budget binds. Returns True if the column grew."""
        nonlocal best
        if not grow:
            return False
        grew = False
        while n_leaves_now() < max_groups and not budget.exhausted():
            col = cols[k]
            ref = budget.best_rms if budget.best_rms != float("inf") else best
            gtol = _grow_threshold(grow_tol, grow_tol_rel, ref)
            best_trial, best_tc = None, best
            for i in range(len(col) - 1):
                if budget.exhausted():
                    break
                if (col[i + 1] - col[i]) < 2 * min_gap_tau - 1e-12:
                    continue
                trial = col[: i + 1] + [0.5 * (col[i] + col[i + 1])] + col[i + 1 :]
                cols[k] = trial
                c = _polish_index(trial, i + 1, min_gap_tau, cost_current())
                cols[k] = col  # revert; only the round winner below is committed
                if c < best_tc - 1e-12:
                    best_tc, best_trial = c, list(trial)
            if best_trial is not None and (best - best_tc) > gtol:
                cols[k] = best_trial
                best = best_tc
                budget.reset_plateau(state["n_evals"])
                grew = True
                on_step(f"tau-split-col{k}", best, n_leaves_now())
            else:
                break
        return grew

    def _prune_column(k):
        """Local prune: try removing each interior cut of column k, commit the best strict
        improvement; repeat until none improves. Returns True if the column shrank."""
        nonlocal best
        if not grow:
            return False
        pruned = False
        while not budget.exhausted():
            col = cols[k]
            if len(col) <= 2:
                return pruned
            best_trial, best_tc = None, best
            for j in range(1, len(col) - 1):
                if budget.exhausted():
                    break
                trial = col[:j] + col[j + 1 :]
                cols[k] = trial
                c = cost_current()
                cols[k] = col  # revert; only the round winner below is committed
                if c < best_tc - 1e-12:
                    best_tc, best_trial = c, trial
            if best_trial is not None:
                cols[k] = best_trial
                best = best_tc
                budget.reset_plateau(state["n_evals"])
                pruned = True
                on_step(f"tau-merge-col{k}", best, n_leaves_now())
            else:
                break
        return pruned

    rms0 = float(evaluate(binning_tree=seed_tree)[1]["rms"])  # rms of the user's seed binning
    best = cost_current()
    checkpoint("start", _last_r[0])
    for _ in range(cfg.max_block_rounds):
        if budget.exhausted():
            break
        start = best
        _sweep(lam, min_gap_lam)  # (a) lambda-cut positions; count fixed
        on_step("lambda-wiggle", best, n_leaves_now())
        if budget.exhausted():
            break
        for k in range(len(cols)):  # (b) per-column tau stacks: positions + grow/prune
            if budget.exhausted():
                break
            _sweep(cols[k], min_gap_tau)
            _grow_column(k)
            _prune_column(k)
            _sweep(cols[k], min_gap_tau)  # re-seat positions after any structural change
            on_step(f"tau-col{k}", best, n_leaves_now())
        on_step("joint-wiggle", best, n_leaves_now())
        checkpoint("round", _last_r[0])
        if budget.exhausted() or (start - best) <= cfg.block_tol * max(abs(start), 1.0):
            break

    lam_out = [round(float(v), 4) for v in lam]
    cols_out = [[round(float(v), 4) for v in c] for c in cols]
    # Rounding can collapse two cuts closer than ~5e-5 apart; re-seat to min-gap
    # (endpoints pinned) so tree_from_columns never raises after the budget is spent.
    _resnap(cols_out, min_gap_tau)
    final_tree = tree_from_columns(lam_out, cols_out)
    final_r = evaluate(binning_tree=final_tree)[1]
    n_leaves = sum(len(c) - 1 for c in cols_out)
    return {
        "binning_tree": _round_tree(final_tree),
        "tree": True,
        "columns": True,
        "rms": float(final_r["rms"]),
        "rms0": rms0,
        "n_empty": int(final_r.get("n_empty", 0)),
        "n_leaves": n_leaves,
        "n_groups": n_leaves,
        "n_bands_total": int(final_r.get("n_bands", n_leaves * 3)),
        "lambda_edges": lam_out,
        "tau_per_lambda": cols_out,
        "n_evals": state["n_evals"],
        "elapsed": round(time.perf_counter() - t0, 2),
        "stop_reason": budget.stop_reason or "converged",
        "window": list(final_r.get("window", [])),
        "history": history,
    }


# --- exhaustive grid search ----------------------------------------------------


def _grid_points(lo, hi, d):
    """Grid lo..hi inclusive at step d, with hi forced as the last point."""
    n = int(round((hi - lo) / d)) + 1
    pts = [lo + i * d for i in range(n)]
    pts[-1] = hi
    return pts


def _grid_cache_files(model, tau_pts, lam_pts, min_opacity_delta):
    """Disk-cache paths for the per-rectangle q_per_band table (keyed by model + grid + gating)."""
    key = "|".join(
        [
            qrad_core._model_name(model),
            ",".join(f"{p:.4g}" for p in tau_pts),
            ",".join(f"{p:.4g}" for p in lam_pts),
            f"mod{float(min_opacity_delta):.4g}",
        ]
    )
    h = hashlib.blake2b(key.encode(), digest_size=8).hexdigest()
    base = _REPO / f"grid_qcache_{h}"
    return base.with_suffix(".Q.npy"), base.with_suffix(".rects.npy")


def _precompute_grid_q(tau_pts, lam_pts, model, min_opacity_delta, ckpt_seconds=300.0):
    """Precompute the 3-segment q(rho) profile for EVERY grid-aligned rectangle in the window.

    Because Q_rad = sum of independent per-band Q (qrad_core._qrad_from_table), a partition's Q is
    just the sum of its rectangles' precomputed profiles -- so this one-time RTE sweep (the only
    transfer cost) makes scoring any tiling a microsecond array sum. Cached to disk (.npy).
    Returns (Q[n_rects,3,nz], rect_index {(i,j,k,l): row}).

    Resumable: Q is a memory-mapped .npy written row-by-row, and a progress-index sidecar
    (`<qf>.idx.npy` = # rows completed) is flushed every `ckpt_seconds` (default 5 min). An
    interrupted precompute therefore resumes from the last checkpoint instead of restarting at
    rectangle 0; the sidecar is deleted once the table is complete.
    """
    ntau, nlam = len(tau_pts), len(lam_pts)
    rects = [
        (i, j, k, l) for i in range(ntau) for j in range(i + 1, ntau) for k in range(nlam) for l in range(k + 1, nlam)
    ]
    qf, rf = _grid_cache_files(model, tau_pts, lam_pts, min_opacity_delta)
    idxf = qf.with_suffix(".idx.npy")  # resume marker: # rows completed

    # Completed-cache fast path: Q + rects present AND no in-progress sidecar.
    if qf.exists() and rf.exists() and not idxf.exists():
        Q = np.load(qf, mmap_mode="r")
        if Q.shape[0] == len(rects):
            rects_disk = np.load(rf)
            print(f"[grid] loaded rectangle-q cache: {qf.name} ({len(rects)} rects)")
            return np.asarray(Q), {tuple(int(x) for x in r): m for m, r in enumerate(rects_disk)}

    ref = qrad_core.reference(model)
    nz = len(ref["ltau"])
    shape = (len(rects), 3, nz)

    # Resume from the last checkpoint if the sidecar + a shape-matching Q exist; else start fresh.
    resume_at = 0
    if qf.exists() and idxf.exists():
        try:
            if np.load(qf, mmap_mode="r").shape == shape:
                resume_at = int(np.load(idxf))
        except Exception:
            resume_at = 0
    if resume_at > 0:
        print(f"[grid] resuming rectangle-q precompute at {resume_at}/{len(rects)}")
        Q = np.lib.format.open_memmap(qf, mode="r+")
    else:
        qf.unlink(missing_ok=True)
        idxf.unlink(missing_ok=True)
        Q = np.lib.format.open_memmap(qf, mode="w+", dtype=np.float64, shape=shape)

    from tqdm import tqdm

    next_ckpt = time.perf_counter() + ckpt_seconds
    for m in tqdm(
        range(resume_at, len(rects)), initial=resume_at, total=len(rects), desc="grid precompute", unit="rect"
    ):
        i, j, k, l = rects[m]
        tree = {
            "window_tau": [tau_pts[i], tau_pts[j]],
            "window_lam": [lam_pts[k], lam_pts[l]],
            "root": {"leaf": True},
        }
        try:
            r = qrad_core.score_binning(None, None, None, model, binning_tree=tree, min_opacity_delta=min_opacity_delta)
            Q[m] = np.asarray(r["q_per_band"], dtype=np.float64)
        except ValueError:
            pass  # empty rectangle (no sub-bins inside) -> Q[m] stays 0 (it adds no heating)
        now = time.perf_counter()
        if now >= next_ckpt or m == len(rects) - 1:
            Q.flush()
            np.save(idxf, np.int64(m + 1))
            next_ckpt = now + ckpt_seconds
    Q.flush()
    np.save(rf, np.asarray(rects, dtype=np.int64))
    idxf.unlink(missing_ok=True)  # complete -> drop the resume marker
    print(f"[grid] cached rectangle-q table: {qf.name} ({len(rects)} rects)")
    return np.asarray(Q), {r: m for m, r in enumerate(rects)}


def _reconstruct_guillotine(leaves, tau_pts, lam_pts):
    """Build a guillotine tree from a set of leaf rects given as (i0,i1,k0,k1) grid tuples.

    A guillotine partition always has a full axis-aligned cut that no leaf crosses; find it
    (trying tau then lambda), split the leaf set, and recurse. k is small (<= ~8) so this is
    trivial next to the enumeration that produced the leaf set. Every tree sharing a leaf-rect
    set scores identically (Q_rad is additive per leaf), so any valid representative suffices.
    """
    if len(leaves) == 1:
        return {"leaf": True}
    i0_min = min(r[0] for r in leaves)
    for c in sorted({r[1] for r in leaves}):  # tau cut: left group's right edge at c
        if c <= i0_min:
            continue
        lo = {r for r in leaves if r[1] <= c}
        hi = {r for r in leaves if r[0] >= c}
        if lo and hi and len(lo) + len(hi) == len(leaves):
            return {
                "axis": "tau",
                "at": float(tau_pts[c]),
                "lo": _reconstruct_guillotine(lo, tau_pts, lam_pts),
                "hi": _reconstruct_guillotine(hi, tau_pts, lam_pts),
            }
    k0_min = min(r[2] for r in leaves)
    for c in sorted({r[3] for r in leaves}):  # lambda cut: low group's top edge at c
        if c <= k0_min:
            continue
        lo = {r for r in leaves if r[3] <= c}
        hi = {r for r in leaves if r[2] >= c}
        if lo and hi and len(lo) + len(hi) == len(leaves):
            return {
                "axis": "lam",
                "at": float(lam_pts[c]),
                "lo": _reconstruct_guillotine(lo, tau_pts, lam_pts),
                "hi": _reconstruct_guillotine(hi, tau_pts, lam_pts),
            }
    raise ValueError(f"leaf set is not a guillotine partition: {sorted(leaves)}")


def grid_search(
    tau_window,
    lam_window,
    *,
    dtau=0.5,
    dlam=0.25,
    model=None,
    max_groups=8,
    min_opacity_delta=1.0,
    window=None,
    refine_topk: int = 50,
    on_progress=None,
    on_improve=None,
):
    """Exhaustive grid search over ALL general guillotine partitions.

    Cuts are restricted to a regular grid (step `dtau` in -log10 tau, `dlam` in log10 lambda) but
    may recurse on either axis, so a lambda-split region can carry its own tau cuts (and vice
    versa) -- a strict superset of the shared-tau / tau-then-lambda form. Every rectangle's
    3-segment q(rho) is precomputed once (`_precompute_grid_q`); then ALL guillotine tilings up to
    `max_groups` leaves are enumerated (deduped by leaf-rectangle set) and scored by array
    summation, so the result is the EXACT grid optimum -- no heuristic, no basin trapping. A final
    `score_binning` call gives the authoritative rms on the winning tree. The guillotine count
    grows ~exponentially with the grid, so use a coarse dtau/dlam. Returns the same dict shape as
    `optimize_qrad`.
    """
    t0 = time.perf_counter()
    tlo0, thi0 = float(tau_window[0]), float(tau_window[-1])
    llo0, lhi0 = float(lam_window[0]), float(lam_window[-1])
    tau_pts = _grid_points(tlo0, thi0, dtau)
    lam_pts = _grid_points(llo0, lhi0, dlam)
    ntau, nlam = len(tau_pts), len(lam_pts)
    print(f"[grid] tau grid ({dtau}): {len(tau_pts)} pts  lambda grid ({dlam}): {len(lam_pts)} pts")

    Q, rect_index = _precompute_grid_q(tau_pts, lam_pts, model, min_opacity_delta)
    print(
        f"[mem] after precompute: Q.shape={Q.shape} dtype={str(Q.dtype)} "
        f"nbytes={Q.nbytes / 1e6:.1f}MB mmap={isinstance(Q, np.memmap)} "
        f"rect_index={len(rect_index)} rect_index_deep={_deep_size_mb(rect_index):.1f}MB "
        f"RSS={_rss_mb():.0f}MB",
        flush=True,
    )
    ref = qrad_core.reference(model)
    q_full = np.asarray(ref["q_full"])
    rho = np.asarray(ref["rho"])
    ltau = np.asarray(ref["ltau"])
    win = qrad_core.WINDOW if window is None else (float(window[0]), float(window[1]))
    in_win = (ltau >= min(win)) & (ltau <= max(win))
    print(
        f"[mem] ref arrays: q_full={q_full.nbytes / 1e6:.2f}MB rho={rho.nbytes / 1e6:.2f}MB "
        f"ltau={ltau.nbytes / 1e6:.2f}MB nz={len(ltau)} in_win={int(in_win.sum())} "
        f"RSS={_rss_mb():.0f}MB",
        flush=True,
    )

    def rms_of(idxs):
        q = Q[np.asarray(idxs)].sum(axis=(0, 1))
        resid = (q - q_full) / rho
        return float(np.sqrt(np.mean(resid[in_win] ** 2)))

    whole = rect_index[(0, ntau - 1, 0, nlam - 1)]
    rms0 = rms_of([whole])
    # Enumerate ALL general guillotine partitions (cuts recurse on either axis, so a lambda-split
    # region can carry its OWN tau cuts -- the shared-tau / tau-then-lambda form is the special case
    # where every tau cut sits above every lambda cut). pexact(rect, k) yields every partition of
    # `rect` into exactly k grid-aligned leaves, deduped by sorted leaf-rect-index signature: the
    # same tiling arises from several cut orders (a vertical-then-horizontal split equals the
    # horizontal-then-vertical one), and the per-cell signature set collapses them, so each distinct
    # leaf-rectangle set is scored exactly once. Leaf-rect Q is precomputed, so scoring is a
    # microsecond array sum; the top-K are then re-ranked with the authoritative (non-additive)
    # score_binning. COST: the guillotine count grows ~exponentially with the grid -- use a coarse
    # dtau/dlam so the enumeration terminates; a fine grid will not finish.
    rect_grid = {m: rk for rk, m in rect_index.items()}  # rect index -> (i0,i1,k0,k1)

    def tree_from_sig(sig):
        """Reconstruct a guillotine tree from a leaf-rect-index signature -- done lazily, only
        for top-K survivors and on_improve, since every tree sharing a leaf-rect set scores
        identically (Q_rad is additive per leaf)."""
        return _reconstruct_guillotine({rect_grid[m] for m in sig}, tau_pts, lam_pts)

    memo: dict = {}

    def pexact(i0: int, i1: int, k0: int, k1: int, k: int) -> set:
        """Distinct guillotine partitions of tau[i0:i1] x lam[k0:k1] into exactly k leaves.

        Returns a SET of signatures (sorted tuples of leaf-rect indices), memoized on (rect, k).
        Only the leaf-rect set is stored -- the guillotine tree is reconstructed lazily from the
        signature for the top-K survivors (tree_from_sig), since every tree sharing a leaf-rect
        set scores identically (Q_rad is additive per leaf). This keeps the memo at one compact
        tuple per distinct tiling instead of a (rows_list, node_dict) per tiling -- the difference
        between fitting and OOM on a 62x17 / max_groups=5 grid.
        """
        key = (i0, i1, k0, k1, k)
        cached = memo.get(key)
        if cached is not None:
            return cached
        sigs: set = set()
        if k == 1:
            sigs.add((rect_index[(i0, i1, k0, k1)],))
        else:
            for c in range(i0 + 1, i1):  # tau cut at tau index c -> left/right halves on tau
                for cl in range(1, k):
                    left = pexact(i0, c, k0, k1, cl)
                    if not left:
                        continue
                    right = pexact(c, i1, k0, k1, k - cl)
                    if not right:
                        continue
                    for lsig in left:
                        for rsig in right:
                            sigs.add(tuple(sorted(lsig + rsig)))
            for r in range(k0 + 1, k1):  # lambda cut at lambda index r -> top/bottom on lambda
                for cl in range(1, k):
                    top = pexact(i0, i1, k0, r, cl)
                    if not top:
                        continue
                    bot = pexact(i0, i1, r, k1, k - cl)
                    if not bot:
                        continue
                    for tsig in top:
                        for bsig in bot:
                            sigs.add(tuple(sorted(tsig + bsig)))
        memo[key] = sigs
        return sigs

    from tqdm import tqdm

    K = max(int(refine_topk), 1)
    topk: list = []  # max-heap of (-memoized_rms, cnt, sig)
    worst_kept = np.inf
    best_memo = np.inf
    cnt = 0
    n_tilings = 0
    win_tau = [float(tau_pts[0]), float(tau_pts[-1])]
    win_lam = [float(lam_pts[0]), float(lam_pts[-1])]
    # pexact is memoized on (rect, k); the pre-pass builds the memo once (same node-building the
    # loop below would do lazily). INSTRUMENTED: print memo growth + RSS per k so the OOM point
    # is visible -- the memo holds EVERY distinct tiling of EVERY sub-rectangle, so this is the
    # suspected memory blowup.
    print(f"[mem] === pre-pass: building memo for k=1..{max_groups} (ntau={ntau} nlam={nlam}) ===", flush=True)
    total_tilings = 0
    for _k in range(1, max_groups + 1):
        _d = pexact(0, ntau - 1, 0, nlam - 1, _k)
        _n_full = len(_d)
        total_tilings += _n_full
        _n_states = len(memo)
        _n_sigs = sum(len(v) for v in memo.values())
        print(
            f"[mem] k={_k}: full_rect_tilings={_n_full} cumulative={total_tilings} "
            f"memo_states={_n_states} memo_sigs={_n_sigs} RSS={_rss_mb():.0f}MB",
            flush=True,
        )
    print(f"[grid] scoring {total_tilings} distinct tilings (k=1..{max_groups})", flush=True)
    pbar = tqdm(total=total_tilings, desc="grid tilings", unit="tiling")
    for k in range(1, max_groups + 1):
        for sig in pexact(0, ntau - 1, 0, nlam - 1, k):
            rms = rms_of(sig)
            n_tilings += 1
            pbar.update(1)
            if n_tilings % 200000 == 0:
                print(
                    f"[mem] loop: n_tilings={n_tilings} memo_states={len(memo)} topk={len(topk)} RSS={_rss_mb():.0f}MB",
                    flush=True,
                )
            if rms < worst_kept or len(topk) < K:
                cnt += 1
                heapq.heappush(topk, (-rms, cnt, sig))
                if len(topk) > K:
                    heapq.heappop(topk)
                worst_kept = -topk[0][0]
                if rms < best_memo - 1e-12:
                    best_memo = rms
                    if on_progress is not None:
                        on_progress("best", n_tilings, rms, k)
                    if on_improve is not None:
                        on_improve(
                            {
                                "window_tau": list(win_tau),
                                "window_lam": list(win_lam),
                                "root": copy.deepcopy(tree_from_sig(sig)),
                            },
                            {"rms": rms, "n_groups": k},
                            n_tilings,
                        )
    pbar.close()
    # Memo no longer needed: the top-K signatures are captured. Free it before the (heavy,
    # full-RTE) re-rank phase.
    memo.clear()
    # Re-rank the top-K with authoritative score_binning; keep the true best. Each eval is a
    # full RTE solve (~3 s), so this is the slow phase -- a bar over the known K is worthwhile.
    best = None
    ranked = sorted(topk, key=lambda x: -x[0])  # ascending memoized rms
    for _neg, _c, sig in tqdm(ranked, total=len(ranked), desc="refine top-K", unit="eval"):
        tree_i = {"window_tau": list(win_tau), "window_lam": list(win_lam), "root": copy.deepcopy(tree_from_sig(sig))}
        r_i = qrad_core.score_binning(
            None, None, None, model, binning_tree=tree_i, min_opacity_delta=min_opacity_delta, window=window
        )
        if best is None or r_i["rms"] < best[1]["rms"] - 1e-12:
            best = (tree_i, r_i)
    tree, final = best
    if on_progress is not None:
        on_progress("done", n_tilings, float(final["rms"]), _n_leaves(tree))
    return {
        "binning_tree": _round_tree(tree),
        "tree": True,
        "rms": float(final["rms"]),
        "rms0": rms0,
        "n_empty": int(final.get("n_empty", 0)),
        "n_leaves": _n_leaves(tree),
        "n_bands_total": int(final.get("n_groups", 0)),
        "n_evals": n_tilings,
        "elapsed": round(time.perf_counter() - t0, 2),
        "stop_reason": "grid_search",
        "window": list(final.get("window", [])),
        "history": [{"tag": "grid", "n_evals": n_tilings, "rms": float(final["rms"]), "groups": _n_leaves(tree)}],
        "grid": {"dtau": dtau, "dlam": dlam, "tau_pts": tau_pts, "lam_pts": lam_pts},
    }


# --- CLI ------------------------------------------------------------------------
app = typer.Typer(add_completion=False, help="Optimize the opacity binning to minimize the Q_rad residual.")


def _flags_str(flags) -> str:
    return "".join("1" if b else "0" for b in flags)


@app.command()
def main(
    tau_bin_edges: list[float] = typer.Option(
        [-0.63, 0.3488, 1.2275, 2.885, 7.0], "--tau-bin-edges", help="Interior+outer tau edges (use = for negatives)."
    ),
    lambda_bin_edges: list[float] = typer.Option([3.0, 3.8, 5.0], "--lambda-bin-edges", help="Lambda-cell edges."),
    model: str = typer.Option(
        "", "--model", help="Atmosphere under models/ to optimize on (default: G2_1D.dat). Must pass validation."
    ),
    split_lambda: str = typer.Option("", "--split-lambda", help="Per-tau-group 0/1 split flags (default all-on)."),
    opt_tau: bool = typer.Option(True, "--opt-tau/--no-opt-tau"),
    opt_lambda: bool = typer.Option(True, "--opt-lambda/--no-opt-lambda"),
    opt_flags: bool = typer.Option(True, "--opt-flags/--no-opt-flags"),
    grow: bool = typer.Option(True, "--grow/--no-grow", help="Allow growing the tau-group count."),
    per_group_lambda: bool = typer.Option(
        False, "--per-group-lambda/--shared-lambda", help="Give each tau group its own lambda split."
    ),
    metric: str = typer.Option("rms", "--metric", help="Objective: rms | maxabs | int_q."),
    min_gap_tau: float = typer.Option(MIN_GAP_TAU, "--min-gap-tau"),
    min_gap_lam: float = typer.Option(MIN_GAP_LAM, "--min-gap-lam"),
    min_opacity_delta: float = typer.Option(
        1.0,
        "--min-opacity-delta",
        help="Only split a group into low/mid/high when its bottom opacity max/min >= this (1 = always split).",
    ),
    max_groups: int = typer.Option(8, "--max-groups"),
    initial_tau_bins: int = typer.Option(
        3,
        "--initial-tau-bins",
        help="Staged seeding: start from N equally-spaced tau cuts (0 = off), polish tau positions, "
        "lambda-split the bottom two tau groups, then run the normal grow/polish path.",
    ),
    initial_tau_scan: int = typer.Option(
        64,
        "--initial-tau-scan",
        help="Staged seeding: LHS-scan NUM tau-cut sets on the dtau grid and seed from the winner (0 = off).",
    ),
    staged_lambda_scan: int = typer.Option(
        0,
        "--staged-lambda-scan",
        help="Staged seeding: LHS-scan NUM (L, t_lo, t_hi) bottom-region triples after tau polish (0 = off).",
    ),
    seed: int = typer.Option(
        None,
        "--seed",
        help="RNG seed for the staged LHS scans (same seed = reproducible run; omit = fresh entropy).",
    ),
    max_evals: int = typer.Option(400, "--max-evals"),
    max_seconds: float = typer.Option(1800.0, "--max-seconds"),
    window_lo: float = typer.Option(-5.0, "--window-lo", help="Score rms over log10(tau_Ros) >= this."),
    window_hi: float = typer.Option(3.0, "--window-hi", help="Score rms over log10(tau_Ros) <= this."),
    target_rms: float = typer.Option(0.0, "--target-rms", help="Stop once rms <= this (0 = off)."),
    plateau_evals: int = typer.Option(0, "--plateau-evals", help="Stop if rms stalls this many evals (0 = off)."),
    grow_tol_rel: float = typer.Option(
        0.01, "--grow-tol-rel", help="Grow a tau group only if rms improves > this frac."
    ),
    save_plot: str = typer.Option("", "--save-plot", help="Write a before/after Q/rho plot to this path."),
    save_dat: bool = typer.Option(
        False, "--save-dat/--no-save-dat", help="After optimizing, write the optimized binning's kappa .dat table."
    ),
    plot: str = typer.Option(
        "",
        "--plot",
        help="Directory: write a tau-lambda binning plot for every improved binning found during the search.",
    ),
    tree: bool = typer.Option(False, "--tree/--no-tree", help="General 2D guillotine mode."),
    beam_width: int = typer.Option(
        3, "--beam-width", help="Rival tree topologies kept in parallel each grow round (1 = greedy grow)."
    ),
    beam_leaves: int = typer.Option(
        4, "--beam-leaves", help="Widest leaves considered for splitting, per beam tree (beam_width >= 2)."
    ),
    beam_positions: list[float] = typer.Option(
        [0.35, 0.5, 0.65], "--beam-positions", help="Split-position fractions tried per (leaf, axis) (beam_width >= 2)."
    ),
    columns: bool = typer.Option(
        False,
        "--columns/--no-columns",
        help="Columns-constrained mode: lambda-cut count fixed, each lambda column keeps its own tau "
        "stack (positions + grow/prune under --max-groups). Replaces the general-guillotine grow/polish path.",
    ),
    tau_per_lambda: list[str] = typer.Option(
        [],
        "--tau-per-lambda",
        help="Columns-mode warm start: repeat once per lambda column (in order), each a comma-separated "
        "increasing tau-edge list sharing the outer window, e.g. --tau-per-lambda=-0.63,0.1,7 "
        "--tau-per-lambda=-0.63,2.0,7. Defaults to the shared --tau-bin-edges in every column.",
    ),
    use_grid_search: bool = typer.Option(
        False,
        "--grid-search/--no-grid-search",
        help="Exhaustive grid search over tau-then-lambda tilings (cuts on a dtau/dlam grid). "
        "Precomputes every rectangle's Q once, then enumerates ALL tilings up to --max-groups -> "
        "exact grid optimum. Replaces the beam/multi-start heuristic.",
    ),
    dtau: float = typer.Option(0.5, "--dtau", help="Grid step in -log10(tau) for --grid-search."),
    dlam: float = typer.Option(0.25, "--dlam", help="Grid step in log10(lambda) for --grid-search."),
    plot_every: int = typer.Option(
        0,
        "--plot-every",
        help="With --plot: also save a tiling frame every N evals (not just improvements), "
        "so the animation covers every step. 0 = improvements only.",
    ),
    save_tree_path: str = typer.Option(
        "",
        "--save-tree",
        help="Write the result binning_tree to this JSON path after optimizing (round-trip exact via load-tree).",
    ),
    load_tree_path: str = typer.Option(
        "",
        "--load-tree",
        help="Load a saved binning_tree JSON as the warm start (highest seed precedence; "
        "staged seeding is skipped when loaded). Mutually exclusive with --columns and --grid-search.",
    ),
):
    model_name = qrad_core._model_name(model or None)
    report = qrad_core.validate_model_file(qrad_core.MODELS_DIR / model_name)
    if not report["ok"]:
        raise typer.BadParameter(f"model models/{model_name} is not valid: {report['error']}")

    print("[startup] precomputing invariants (reads the ODF; ~10-30s)...")
    try:
        qrad_core.require_data_files()
    except qrad_core.MissingDataError as e:
        raise typer.BadParameter(str(e))
    qrad_core.precompute(model_name)

    n_tau = len(tau_bin_edges) - 1
    flags = parse_split_lambda(split_lambda) if split_lambda.strip() else [True] * n_tau
    if len(flags) != n_tau:
        raise typer.BadParameter(f"--split-lambda has {len(flags)} entries, expected {n_tau}")

    print(f"[qrad-opt] atmosphere=models/{model_name} metric={metric} beam_width={beam_width} grow={grow}")
    print(f"[qrad-opt] start: tau={_fmt(tau_bin_edges)} lam={_fmt(lambda_bin_edges)} flags={_flags_str(flags)}")

    def _progress(tag, value, groups, n):
        print(f"  [{n:4d} evals] {tag:8s} rms={value:.4e} groups={groups}")

    plot_dir = Path(plot).expanduser() if plot.strip() else None
    if plot_dir:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"[qrad-opt] binning plots -> {plot_dir}")
    _plot_seen: set[tuple] = set()
    _plot_n = [0]
    _rms_hist: list[tuple[int, float]] = []  # best-so-far (n_evals, rms) for the top strip

    def _on_improve(tree, r, n_evals):
        if plot_dir is None:
            return
        sig = _tree_signature(tree)  # dedupe identical tilings (e.g. sub-1e-3 position wiggles)
        if sig in _plot_seen:
            return
        _plot_seen.add(sig)
        _plot_n[0] += 1
        _rms_hist.append((int(n_evals), float(r["rms"])))
        out = plot_dir / f"frame_{n_evals:05d}_rms_{float(r['rms']):.3e}.png"
        _plot_tree_binning(
            tree,
            out,
            rms=float(r["rms"]),
            n_empty=int(r.get("n_empty", 0)),
            groups=int(r.get("n_groups", 0)),
            n_evals=n_evals,
            seq=_plot_n[0],
            model=model_name,
            rms_history=list(_rms_hist),
        )

    _plot_every = [0]  # evals since last --plot-every frame

    def _on_eval_frame(n_evals, cost, r):
        # --plot-every N: save a frame every N evals (any tiling, not just improvements).
        # Same frame_NNNNN_rms_*.png naming as improvements, so one glob covers every step.
        # The scorer raw dict carries group_tau/lam_edges; pass them as display rects.
        if plot_dir is None or not plot_every or plot_every < 1:
            return
        if not isinstance(r, dict) or r.get("group_tau_edges") is None:
            return
        _plot_every[0] += 1
        if _plot_every[0] < plot_every:
            return
        _plot_every[0] = 0
        _plot_n[0] += 1
        _rms_hist.append((int(n_evals), float(r.get("rms", cost))))
        gte = np.asarray(r["group_tau_edges"], dtype=float)
        gle = np.asarray(r["group_lam_edges"], dtype=float)
        rects = [(float(a), float(b), float(c), float(d)) for (a, b), (c, d) in zip(gte, gle)]
        tw = (float(gte[:, 0].min()), float(gte[:, 1].max()))
        lw = (float(gle[:, 0].min()), float(gle[:, 1].max()))
        out = plot_dir / f"frame_{n_evals:05d}_rms_{float(r.get('rms', cost)):.3e}.png"
        _plot_tree_binning(
            {"rects": rects, "window_tau": list(tw), "window_lam": list(lw)},
            out,
            rms=float(r.get("rms", cost)),
            n_empty=int(r.get("n_empty", 0)),
            groups=int(r.get("n_groups", 0)),
            n_evals=n_evals,
            seq=_plot_n[0],
            model=model_name,
            rms_history=list(_rms_hist),
        )

    loaded_tree = None
    if load_tree_path.strip():
        if columns:
            raise typer.BadParameter("--load-tree is mutually exclusive with --columns (trees are general-2D).")
        if use_grid_search:
            raise typer.BadParameter("--load-tree is mutually exclusive with --grid-search.")
        try:
            loaded_tree = load_tree(load_tree_path.strip(), min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam)
        except ValueError as e:
            raise typer.BadParameter(str(e))
    if use_grid_search and columns:
        raise typer.BadParameter("--columns is mutually exclusive with --grid-search.")
    if columns and tree:
        raise typer.BadParameter("--columns is mutually exclusive with --tree.")
    if tau_per_lambda and not columns:
        raise typer.BadParameter("--tau-per-lambda requires --columns.")
    if use_grid_search:

        def _grid_progress(tag, a, b, c):
            from tqdm import tqdm

            tqdm.write(f"  [grid] {tag}: rms={b:.4e} leaves={c} tilings={a}")

        result = grid_search(
            tau_bin_edges,
            lambda_bin_edges,
            dtau=dtau,
            dlam=dlam,
            model=model_name,
            max_groups=max_groups,
            min_opacity_delta=min_opacity_delta,
            window=(window_lo, window_hi),
            on_progress=_grid_progress,
            on_improve=_on_improve if plot_dir else None,
        )
    elif columns:
        from tausort import parse_tau_per_lambda

        cols_seed = (
            parse_tau_per_lambda(tau_per_lambda)
            if tau_per_lambda
            else [list(tau_bin_edges) for _ in range(len(lambda_bin_edges) - 1)]
        )
        if len(cols_seed) != len(lambda_bin_edges) - 1:
            raise typer.BadParameter(
                f"--tau-per-lambda has {len(cols_seed)} entries, expected one per lambda column ({len(lambda_bin_edges) - 1})"
            )
        result = optimize_columns(
            lambda_bin_edges,
            cols_seed,
            model=model_name,
            metric=metric,
            min_gap_tau=min_gap_tau,
            min_gap_lam=min_gap_lam,
            max_groups=max_groups,
            max_evals=max_evals,
            max_seconds=max_seconds,
            grow=grow,
            window=(window_lo, window_hi),
            target_rms=(target_rms if target_rms > 0 else None),
            plateau_evals=plateau_evals,
            grow_tol_rel=grow_tol_rel,
            min_opacity_delta=min_opacity_delta,
            on_progress=_progress,
            on_improve=_on_improve if plot_dir else None,
            on_eval=_on_eval_frame if plot_dir and plot_every else None,
        )
    else:
        result = optimize_qrad(
            tau_bin_edges,
            lambda_bin_edges,
            flags=flags,
            model=model_name,
            opt_tau=opt_tau,
            opt_lambda=opt_lambda,
            opt_flags=opt_flags,
            grow=grow,
            metric=metric,
            min_gap_tau=min_gap_tau,
            min_gap_lam=min_gap_lam,
            max_groups=max_groups,
            max_evals=max_evals,
            max_seconds=max_seconds,
            window=(window_lo, window_hi),
            target_rms=(target_rms if target_rms > 0 else None),
            plateau_evals=plateau_evals,
            grow_tol_rel=grow_tol_rel,
            per_group_lambda=per_group_lambda,
            tree=tree,
            beam_width=beam_width,
            beam_positions=tuple(beam_positions),
            beam_leaves=beam_leaves,
            min_opacity_delta=min_opacity_delta,
            binning_tree=loaded_tree,
            initial_tau_bins=None if loaded_tree is not None else (initial_tau_bins if initial_tau_bins else None),
            initial_tau_scan=(
                0 if loaded_tree is not None else (initial_tau_scan if initial_tau_scan and initial_tau_bins else 0)
            ),
            staged_lambda_scan=(
                0 if loaded_tree is not None else (staged_lambda_scan if staged_lambda_scan and initial_tau_bins else 0)
            ),
            seed=seed,
            on_progress=_progress,
            on_eval=_on_eval_frame if plot_dir and plot_every else None,
            on_improve=_on_improve if plot_dir else None,
        )

    imp = (result["rms0"] - result["rms"]) / result["rms0"] * 100.0
    print("\n[qrad-opt] DONE")
    print(f"  rms: {result['rms0']:.4e} -> {result['rms']:.4e}  ({imp:+.1f}%)")
    _mode = "columns" if columns else "general-2D tree"
    print(f"  {_mode}: {result['n_leaves']} leaf bands, n_empty={result['n_empty']}")
    print(f"  {result['n_evals']} evals in {result['elapsed']}s")
    if plot_dir:
        print(f"  binning plots -> {plot_dir} ({_plot_n[0]} improved binnings)")
        # Always emit the final optimized tiling: the beam path plots each improvement via
        # on_improve, but grid_search never fires it, so render the winner explicitly here.
        final_plot = plot_dir / "final.png"
        _plot_tree_binning(
            result["binning_tree"],
            final_plot,
            rms=result["rms"],
            n_empty=result["n_empty"],
            groups=result["n_leaves"],
            n_evals=result["n_evals"],
            model=model_name,
            rms_history=list(_rms_hist),
        )
        print(f"  final binning -> {final_plot}")

    if save_plot:
        if columns:
            from tausort import parse_tau_per_lambda as _parse_tpl

            _seed_cols = (
                _parse_tpl(tau_per_lambda)
                if tau_per_lambda
                else [list(tau_bin_edges) for _ in range(len(lambda_bin_edges) - 1)]
            )
            seed_tree = tree_from_columns(lambda_bin_edges, _seed_cols)
        else:
            seed_tree = tree_from_lpt(
                tau_bin_edges,
                [[lambda_bin_edges[0], lambda_bin_edges[-1]] for _ in range(len(tau_bin_edges) - 1)],
            )
        _plot_before_after(
            seed_tree,
            result["binning_tree"],
            save_plot,
            model_name,
            min_opacity_delta=min_opacity_delta,
        )
        print(f"  before/after plot -> {save_plot}")

    if save_dat:
        written, _name = qrad_core.save_kappa_dat(
            None,
            None,
            None,
            model_name,
            binning_tree=result["binning_tree"],
            min_opacity_delta=min_opacity_delta,
        )
        print(f"  kappa table -> {written}")

    if save_tree_path.strip():
        written_tree = save_tree(result["binning_tree"], save_tree_path.strip())
        print(f"  binning tree -> {written_tree}")


def _fmt(edges) -> str:
    return "[" + ", ".join(f"{float(e):.4g}" for e in edges) + "]"


def _tree_bands_str(tree) -> str:
    """One line per leaf rectangle (tau window, lambda window) — the general-2D bands."""
    rects = list(_leaf_rects(tree["root"], _root_rect(tree)))
    return "\n    ".join(f"tau[{tlo:.3f},{thi:.3f}] lam[{llo:.3f},{lhi:.3f}]" for tlo, thi, llo, lhi in rects)


def _plot_before_after(before_tree, after_tree, path, model=None, min_opacity_delta=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _kw = {} if min_opacity_delta is None else {"min_opacity_delta": min_opacity_delta}
    a = qrad_core.score_binning(None, None, None, model, binning_tree=before_tree, **_kw)
    b = qrad_core.score_binning(None, None, None, model, binning_tree=after_tree, **_kw)
    ltau, rho = a["ltau"], a["rho"]
    order = np.argsort(ltau)
    win = (ltau[order] >= qrad_core.WINDOW[0] - 1.0) & (ltau[order] <= qrad_core.WINDOW[1] + 1.0)
    idx = order[win]

    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax[0].plot(ltau[idx], a["q_full"][idx] / rho[idx], color="#3ecf8e", lw=2, label="full ODF (reference)")
    ax[0].plot(
        ltau[idx], a["q"][idx] / rho[idx], color="#5b9bd5", lw=1.6, ls="--", label=f"before (rms={a['rms']:.2e})"
    )
    ax[0].plot(ltau[idx], b["q"][idx] / rho[idx], color="#ff9d3c", lw=2.2, label=f"after (rms={b['rms']:.2e})")
    ax[0].set_ylabel("Q / rho [erg/g/s]")
    ax[0].legend(fontsize=9)
    ax[0].set_title(f"Q_rad binning optimization (models/{qrad_core._model_name(model)})")
    ax[1].axhline(0, color="#888", lw=0.8)
    ax[1].plot(ltau[idx], a["resid"][idx], color="#5b9bd5", lw=1.6, ls="--", label="before - full")
    ax[1].plot(ltau[idx], b["resid"][idx], color="#ff9d3c", lw=2.2, label="after - full")
    ax[1].set_ylabel("(Q - Q_full) / rho")
    ax[1].set_xlabel("log10 tau_Ros")
    ax[1].set_xlim(qrad_core.WINDOW[1], qrad_core.WINDOW[0])
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_tree_binning(
    tree,
    path,
    *,
    rms=None,
    n_empty=None,
    groups=None,
    n_evals=None,
    seq=None,
    model=None,
    rms_history=None,
):
    """Render a binning tree's tau-lambda leaf rectangles to `path` (one patch per group).

    With ``rms_history`` (list of (n_evals, rms) best-so-far points), a small rms-vs-evals
    strip spans the top showing the full trajectory with the current point marked.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rects = list(_leaf_rects(tree["root"], _root_rect(tree))) if "root" in tree else list(tree["rects"])
    wt, wl = tree["window_tau"], tree["window_lam"]
    cmap = plt.get_cmap("tab20")

    if rms_history:
        fig, (ax_top, ax) = plt.subplots(
            2, 1, figsize=(7.5, 6.8), gridspec_kw={"height_ratios": [1, 4], "hspace": 0.35}
        )
        xs = [p[0] for p in rms_history]
        ys = [p[1] for p in rms_history]
        ax_top.plot(xs, ys, color="#1f5fa8", lw=1.6, marker="o", ms=3)
        ax_top.set_yscale("log")
        ax_top.set_ylabel("rms", fontsize=9)
        ax_top.tick_params(labelsize=8)
        ax_top.grid(True, color="#ddd", lw=0.5, which="both")
        if n_evals is not None and rms is not None:
            ax_top.plot([n_evals], [rms], marker="o", ms=7, color="red", zorder=5)
        if len(xs) > 1:
            ax_top.set_xlim(0, max(xs) * 1.05)
    else:
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax_top = None
    for i, (tlo, thi, llo, lhi) in enumerate(rects):
        ax.add_patch(
            Rectangle((llo, tlo), lhi - llo, thi - tlo, facecolor=cmap(i % 20), edgecolor="black", lw=1.4, alpha=0.5)
        )
        ax.text(
            0.5 * (llo + lhi), 0.5 * (tlo + thi), str(i + 1), ha="center", va="center", fontsize=9, fontweight="bold"
        )
    # Overlay the actual ODF sub-bins: each plotted at (log10 lambda, -log10 tau_Ros(tau_lambda=1)),
    # coloured by the leaf rectangle it falls into so the data is visible against the bins.
    if model is not None:
        inv = qrad_core.precompute(model)
        xs = np.asarray(inv["bin_x_all"], dtype=float)
        ys = np.asarray(inv["bin_y_all"], dtype=float)
        skip = int(getattr(qrad_core, "SKIP", 0) or 0)
        if skip:
            xs, ys = xs[skip:], ys[skip:]
        leaf = np.full(xs.shape, -1, dtype=int)  # point-in-rectangle -> matches the patch colours exactly
        for i, (tlo, thi, llo, lhi) in enumerate(rects):
            m = (leaf == -1) & (xs >= llo) & (xs <= lhi) & (ys >= tlo) & (ys <= thi)
            leaf[m] = i
        inside = leaf >= 0
        if inside.any():
            ax.scatter(xs[inside], ys[inside], s=5, c=cmap(leaf[inside] % 20), edgecolor="none", alpha=0.8, zorder=3)
        if (~inside).any():
            ax.scatter(xs[~inside], ys[~inside], s=4, c="#666", edgecolor="none", alpha=0.4, zorder=3)
    ax.set_xlim(float(wl[0]), float(wl[1]))
    ax.set_ylim(float(wt[0]), float(wt[1]))
    ax.set_xlabel(r"$\log_{10}(\lambda/\mathrm{\AA})$")
    ax.set_ylabel(r"$-\log_{10}\,\tau$")
    bits = []
    if seq is not None:
        bits.append(f"#{seq}")
    if n_evals is not None:
        bits.append(f"{n_evals} evals")
    if groups is not None:
        bits.append(f"{groups} bands")
    if n_empty:
        bits.append(f"{n_empty} empty")
    subtitle = "   ".join(bits)
    title = "tau-lambda binning"
    if rms is not None:
        title += f"   rms={rms:.3e}"
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""))
    ax.grid(True, color="#ddd", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    app()
