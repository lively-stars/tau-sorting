"""Q_rad-driven binning optimizer.

Searches over the opacity binning to directly MINIMIZE the Q_rad rms residual against
the full-ODF reference — unlike tausort's `optimize_tau_bin_edges`, which maximizes a
proxy (per-group high-segment overlap). As we saw in the webapp, the proxy and the real
target disagree, so this optimizes the metric that actually matters.

Decision variables (chosen scope): the interior tau edges (count may grow), the interior
lambda-cell edge positions, and the per-tau-group split flags. The outer tau/lambda
window is held fixed, as is the number of lambda cells. The atmosphere is selectable via
`model` (a validated 1D model under models/; None -> DEFAULT_MODEL).

Search is block-coordinate: alternate (A) coordinate descent on tau edges, (B) greedy
split-flag flips, (C) coordinate descent on lambda edges, to a fixed point; then
optionally grow the tau-group count by inserting an edge, accepting only real rms
improvements. Each evaluation is a full RTE solve (~3 s via `qrad_core.score_binning`),
so this is a run-and-wait / batch tool, bounded by an eval + wall-clock budget.

CLI:  uv run python qrad_optimize.py --help
"""

from __future__ import annotations

import copy
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
    """Return (evaluate, state). `evaluate(tau, lam, flags) -> (cost, raw_dict)`.

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

    def evaluate(tau_edges, lambda_edges, flags, *, lambda_edges_per_tau=None, binning_tree=None):
        if binning_tree is not None:
            r = score(None, None, None, model, binning_tree=binning_tree, **_kw)
        elif lambda_edges_per_tau is not None:
            r = score(
                list(tau_edges), None, None, model, lambda_edges_per_tau=[list(x) for x in lambda_edges_per_tau], **_kw
            )
        else:
            r = score(list(tau_edges), list(lambda_edges), list(flags), model, **_kw)
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
    method: str = "cd"
    min_gap_tau: float = MIN_GAP_TAU
    min_gap_lam: float = MIN_GAP_LAM
    adjust_steps: tuple = ADJUST_STEPS
    # Few sweeps per tau visit + many rounds => the expensive tau block yields, so the
    # cheap flags/lambda blocks aren't starved of budget (they interleave each round).
    max_sweeps: int = 2
    max_block_rounds: int = 12
    block_tol: float = 1e-4  # relative improvement to keep iterating blocks


# --- continuous-position search (tau edges or lambda edges) ---------------------
def _coord_descent(edges, cost, *, adjust_steps, min_gap, max_sweeps, budget):
    """Gauss-Seidel coordinate descent on interior edges (outer pair fixed).

    For each interior edge, try +/- each adjust step, apply the best strict improvement,
    then move on; repeat sweeps until a full sweep yields no improvement. Rejects any
    candidate that breaks monotonicity or the min-gap. Mirrors tausort's nudge pattern.
    """
    edges = list(edges)
    best = cost(edges)
    for _ in range(max_sweeps):
        improved = False
        for i in range(1, len(edges) - 1):
            best_cand, best_cost = None, best
            for step in adjust_steps:
                for d in (-1.0, +1.0):
                    if budget.exhausted():
                        if best_cand is not None:  # don't discard a found improvement on cutoff
                            edges, best = best_cand, best_cost
                        return edges, best
                    cand = list(edges)
                    cand[i] += d * step
                    if not _valid_monotone(cand) or not _min_gap_ok(cand, min_gap):
                        continue
                    c = cost(cand)
                    if c < best_cost - 1e-12:
                        best_cost, best_cand = c, cand
            if best_cand is not None:
                edges, best, improved = best_cand, best_cost, True
        if not improved:
            break
    return edges, best


def _softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def _edges_from_u(u, a, b, min_gap):
    """Map unconstrained u in R^N to N interior edges via a softmax simplex over the
    N+1 gaps, so the result is always strictly increasing, in (a,b), and min-gap feasible."""
    logits = np.concatenate([[0.0], np.asarray(u, float)])  # fix one d.o.f.
    w = _softmax(logits)
    n = len(u)
    free = (b - a) - (n + 1) * min_gap
    gaps = min_gap + free * w
    return (a + np.cumsum(gaps)[:-1]).tolist()


def _u_from_edges(interior, a, b, min_gap):
    edges = np.concatenate([[a], np.asarray(interior, float), [b]])
    gaps = np.diff(edges)
    free = (b - a) - (len(interior) + 1) * min_gap
    w = np.clip((gaps - min_gap) / free, 1e-9, None)
    w = w / w.sum()
    return (np.log(w[1:]) - np.log(w[0])).tolist()


def _nelder_mead_edges(edges, cost, *, min_gap, budget):
    """Nelder-Mead over the softmax-simplex reparameterization (bounds + monotonicity
    automatic). Warm-started from `edges`; returns the better of NM and the start."""
    from scipy.optimize import minimize

    a, b = edges[0], edges[-1]
    interior = list(edges[1:-1])
    c0 = cost(list(edges))
    if not interior or (b - a) - (len(interior) + 1) * min_gap <= 0:
        return list(edges), c0
    x0 = _u_from_edges(interior, a, b, min_gap)

    def obj(u):
        if budget.exhausted():
            return 1e30
        return cost([a, *_edges_from_u(u, a, b, min_gap), b])

    remaining = max(int(budget.max_evals - budget.state["n_evals"]), len(x0) + 2)
    res = minimize(
        obj,
        np.asarray(x0, float),
        method="Nelder-Mead",
        options={"xatol": 1e-2, "fatol": max(c0 * 1e-4, 1e-9), "maxfev": remaining, "disp": False},
    )
    cand = [a, *_edges_from_u(res.x, a, b, min_gap), b]
    c = cost(cand)
    return (cand, c) if c < c0 - 1e-12 else (list(edges), c0)


def _optimize_positions(edges, cost, *, cfg, min_gap, budget):
    if cfg.method == "nm":
        return _nelder_mead_edges(edges, cost, min_gap=min_gap, budget=budget)
    return _coord_descent(
        edges, cost, adjust_steps=cfg.adjust_steps, min_gap=min_gap, max_sweeps=cfg.max_sweeps, budget=budget
    )


# --- discrete split-flag search -------------------------------------------------
def _flag_search(tau, lam, flags, cost3, *, budget):
    """Greedy single-flip: repeatedly flip the one tau-group flag that most lowers the
    cost, until no flip helps. Only meaningful with >1 lambda cell."""
    flags = list(flags)
    best = cost3(tau, lam, flags)
    while True:
        best_cand, best_cost = None, best
        for i in range(len(flags)):
            if budget.exhausted():
                if best_cand is not None:  # commit a found flip on cutoff
                    flags, best = best_cand, best_cost
                return flags, best
            cand = list(flags)
            cand[i] = not cand[i]
            c = cost3(tau, lam, cand)
            if c < best_cost - 1e-12:
                best_cost, best_cand = c, cand
        if best_cand is None:
            return flags, best
        flags, best = best_cand, best_cost


# --- grow: insert a tau edge at the widest group --------------------------------
def _insert_edge(tau, flags, min_gap):
    """Insert a midpoint edge in the widest tau group that can still respect min_gap;
    the new group inherits the parent's split flag. Returns (None, None) if none fits."""
    for k in sorted(range(len(tau) - 1), key=lambda j: tau[j + 1] - tau[j], reverse=True):
        mid = 0.5 * (tau[k] + tau[k + 1])
        if (mid - tau[k]) >= min_gap and (tau[k + 1] - mid) >= min_gap:
            return tau[: k + 1] + [mid] + tau[k + 1 :], flags[: k + 1] + [flags[k]] + flags[k + 1 :]
    return None, None


# --- block-coordinate fixed point ----------------------------------------------
def _block_fixed_point(tau, lam, flags, evaluate, *, opt_tau, opt_lambda, opt_flags, cfg, budget, report=None):
    tau, lam, flags = list(tau), list(lam), list(flags)
    n_lambda = len(lam) - 1
    best = evaluate(tau, lam, flags)[0]
    for _ in range(cfg.max_block_rounds):
        start = best
        # Cheap blocks first (few evals each) so they always run before the expensive
        # tau block; tau is capped to `max_sweeps` per visit and revisited over rounds.
        if opt_lambda and len(lam) > 2:
            lam, best = _optimize_positions(
                lam, lambda e: evaluate(tau, e, flags)[0], cfg=cfg, min_gap=cfg.min_gap_lam, budget=budget
            )
            if report:
                report("lambda", best, len(tau) - 1)
        if opt_flags and n_lambda > 1:
            flags, best = _flag_search(tau, lam, flags, lambda t, ll, f: evaluate(t, ll, f)[0], budget=budget)
            if report:
                report("flags", best, len(tau) - 1)
        if opt_tau:
            tau, best = _optimize_positions(
                tau, lambda e: evaluate(e, lam, flags)[0], cfg=cfg, min_gap=cfg.min_gap_tau, budget=budget
            )
            if report:
                report("tau", best, len(tau) - 1)
        if budget.exhausted() or (start - best) <= cfg.block_tol * max(abs(start), 1.0):
            break
    return tau, lam, flags, best


# --- per-tau-group lambda variant -----------------------------------------------
def _per_group_lambda_search(lam_per_tau, cost_pg, lmin, lmax, *, cfg, budget):
    """For each tau group independently, keep the better of *no split* vs a *single
    lambda cut* (its position optimized by coordinate descent). This lets each tau
    group choose its own wavelength split, generalizing the shared cut + on/off flag.
    `cost_pg(lambda_edges_per_tau) -> cost` scores the whole binning (tau fixed)."""
    lam_per_tau = [list(x) for x in lam_per_tau]
    best = cost_pg(lam_per_tau)
    can_split = (lmax - lmin) - 2.0 * cfg.min_gap_lam > 0
    for k in range(len(lam_per_tau)):
        if budget.exhausted():
            return lam_per_tau, best

        def cost_k(edges, _k=k):
            cand = [list(x) for x in lam_per_tau]
            cand[_k] = list(edges)
            return cost_pg(cand)

        c_nosplit = cost_k([lmin, lmax])
        split_edges, c_split = [lmin, lmax], float("inf")
        if can_split:
            cur = lam_per_tau[k]
            p0 = cur[1] if len(cur) >= 3 else 0.5 * (lmin + lmax)
            p0 = min(max(p0, lmin + cfg.min_gap_lam), lmax - cfg.min_gap_lam)
            split_edges, c_split = _optimize_positions(
                [lmin, p0, lmax], cost_k, cfg=cfg, min_gap=cfg.min_gap_lam, budget=budget
            )
        # tie or no gain -> prefer no split (parsimony)
        if c_split < c_nosplit - 1e-12:
            lam_per_tau[k], best = list(split_edges), c_split
        else:
            lam_per_tau[k], best = [lmin, lmax], c_nosplit
    return lam_per_tau, best


def _insert_edge_pg(tau, lam_per_tau, min_gap):
    """Grow: split the widest tau group; the new group inherits the parent's lambda edges."""
    for k in sorted(range(len(tau) - 1), key=lambda j: tau[j + 1] - tau[j], reverse=True):
        mid = 0.5 * (tau[k] + tau[k + 1])
        if (mid - tau[k]) >= min_gap and (tau[k + 1] - mid) >= min_gap:
            new_tau = tau[: k + 1] + [mid] + tau[k + 1 :]
            new_lpt = (
                [list(x) for x in lam_per_tau[: k + 1]]
                + [list(lam_per_tau[k])]
                + [list(x) for x in lam_per_tau[k + 1 :]]
            )
            return new_tau, new_lpt
    return None, None


def _block_fixed_point_pg(tau, lpt, evaluate, *, opt_tau, opt_lambda, lmin, lmax, cfg, budget, report=None):
    """Block-coordinate fixed point for per-tau-group lambda: alternate per-group lambda
    choice and shared-tau coordinate descent, to a fixed point."""
    tau, lpt = list(tau), [list(x) for x in lpt]
    best = evaluate(tau, None, None, lambda_edges_per_tau=lpt)[0]
    for _ in range(cfg.max_block_rounds):
        start = best
        if opt_lambda:
            lpt, best = _per_group_lambda_search(
                lpt,
                lambda cand: evaluate(tau, None, None, lambda_edges_per_tau=cand)[0],
                lmin,
                lmax,
                cfg=cfg,
                budget=budget,
            )
            if report:
                report("lambda", best, len(tau) - 1)
        if opt_tau:
            tau, best = _optimize_positions(
                tau,
                lambda e: evaluate(e, None, None, lambda_edges_per_tau=lpt)[0],
                cfg=cfg,
                min_gap=cfg.min_gap_tau,
                budget=budget,
            )
            if report:
                report("tau", best, len(tau) - 1)
        if budget.exhausted() or (start - best) <= cfg.block_tol * max(abs(start), 1.0):
            break
    return tau, lpt, best


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
        return {"axis": node["axis"], "at": round(float(node["at"]), 4), "lo": rec(node["lo"]), "hi": rec(node["hi"])}

    return {
        "window_tau": [round(float(v), 4) for v in tree["window_tau"]],
        "window_lam": [round(float(v), 4) for v in tree["window_lam"]],
        "root": rec(tree["root"]),
    }


def _lam_chain(lam_edges):
    """Right-leaning lambda chain over interior cuts; a bare leaf when there is no split."""
    node = {"leaf": True}
    for c in reversed([float(v) for v in lam_edges[1:-1]]):
        node = {"axis": "lam", "at": c, "lo": {"leaf": True}, "hi": node}
    return node


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


def tree_from_flags(tau_edges, lambda_edges, flags) -> dict:
    lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
    lpt = [list(lambda_edges) if flags[k] else [lmin, lmax] for k in range(len(tau_edges) - 1)]
    return tree_from_lpt(tau_edges, lpt)


def _tree_position_search(tree, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam):
    """Gauss-Seidel coordinate descent on every internal cut's position, generalizing
    _coord_descent. Each candidate must stay in its node's axis span (per-axis min gap) and
    keep the whole tree feasible; the best strict improvement per node is committed in place."""
    best = cost_tree(tree)
    for _ in range(cfg.max_sweeps):
        improved = False
        for node, span_lo, span_hi in list(_iter_internal(tree["root"], _root_rect(tree))):
            mg = min_gap_tau if node["axis"] == "tau" else min_gap_lam
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
    span = next(((lo, hi) for n, lo, hi in _iter_internal(tree["root"], _root_rect(tree)) if n is node), None)
    best_c = best if best is not None else cost_tree(tree)
    if span is None or "at" not in node:
        return tree, best_c
    span_lo, span_hi = span
    mg = min_gap_tau if node["axis"] == "tau" else min_gap_lam
    best_at = float(node["at"])
    for step in cfg.adjust_steps:
        old = best_at
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
    method="cd",
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
    tree=False,  # general 2D guillotine mode (both tau and lambda locally free)
    binning_tree=None,  # guillotine-tree warm start {window_tau, window_lam, root}
    min_opacity_delta=None,  # min bottom-opacity max/min ratio to split a group (None -> score default)
    score_fn=None,
    on_progress=None,
    on_eval=None,
    should_stop=None,
) -> dict:
    """Minimize the Q_rad residual over the binning. Returns a result dict with the
    optimized `tau_edges`/`lambda_edges`/`flags`, `rms`/`rms0`, `n_evals`, `elapsed`,
    `n_empty`, and a `history` of (n_evals, rms, groups) checkpoints.

    Warm-starts from the passed binning. Blocks are individually toggleable via
    opt_tau/opt_lambda/opt_flags, and `grow` enables inserting tau groups (accepted
    only when rms improves by > grow_tol, default 1% of the current rms).

    `on_eval(n_evals, cost, raw_dict)` fires after every evaluation and `should_stop() -> bool`
    is polled at each budget check — both let a caller (e.g. the webapp) show live progress
    and abort a long run gracefully, returning the best binning found so far.
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
    cfg = _Cfg(method=method, min_gap_tau=min_gap_tau, min_gap_lam=min_gap_lam, adjust_steps=tuple(adjust_steps))

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

    if tree or binning_tree is not None:
        # General 2D guillotine. Re-run (binning_tree given) continues refining that tree; a first
        # run seeds a COARSE tree (the tau bands, no lambda splits) and lets grow BUILD the tiling.
        # Seeding from the full per-group grid at max_leaves would leave grow no room, so it could
        # only nudge a fixed grid; from the coarse seed, grow splits leaves on either axis — including
        # a tau sub-split inside a single lambda region, which is what makes the tiling non-grid.
        if binning_tree is not None:
            btree = copy.deepcopy(binning_tree)
        else:
            lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
            btree = tree_from_lpt(tau_edges, [[lmin, lmax] for _ in range(n_tau0)])

        def cost_tree(t):
            return evaluate(None, None, None, binning_tree=t)[0]

        _, r0 = evaluate(None, None, None, binning_tree=btree)
        rms0 = float(r0["rms"])
        best = cost_tree(btree)
        checkpoint("start", r0)
        # Grow FIRST so the budget builds structure (each grow cheaply refines only its new cut).
        # A heavy refine of the coarse seed up front would, at RTE cost, exhaust the budget before
        # a single leaf is split — leaving a plain grid. Grow lets tau/lambda cuts nest to any
        # depth (a tau sub-split inside one lambda region == the non-grid tiling we want).
        if grow:
            while _n_leaves(btree) < max_groups and not budget.exhausted():
                cand = _grow_tree(
                    btree,
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
                # Tree policy: grow to the leaf cap, accepting ANY real rms improvement (gtol 0 by
                # default, overridable via grow_tol). A guillotine split adds DOF so it almost always
                # helps a little; the empty-band penalty + min-gap still reject degenerate/empty
                # splits (those raise the cost), so "grow to more" can't collapse the binning.
                gtol = grow_tol if grow_tol is not None else 0.0
                if (best - cbest) > gtol:
                    btree, best = ctree, cbest
                    budget.reset_plateau(state["n_evals"])
                    checkpoint("grow", evaluate(None, None, None, binning_tree=btree)[1])
                else:
                    break
        # Final polish: sweep every cut position with whatever budget is left.
        if not budget.exhausted():
            btree, best = _block_fixed_point_tree(
                btree,
                cost_tree,
                cfg=cfg,
                budget=budget,
                min_gap_tau=min_gap_tau,
                min_gap_lam=min_gap_lam,
                report=on_step,
            )
            checkpoint("blocks", evaluate(None, None, None, binning_tree=btree)[1])
        final_r = evaluate(None, None, None, binning_tree=btree)[1]
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

    if per_group_lambda:
        lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
        if lambda_edges_per_tau is not None and len(lambda_edges_per_tau) == n_tau0:
            # continue from a prior per-group binning (so re-running keeps refining the cuts),
            # clamping each interior cut into the current [lmin, lmax] window.
            lpt = [
                [lmin, *[min(max(float(e), lmin), lmax) for e in x[1:-1]], lmax] if len(x) >= 3 else [lmin, lmax]
                for x in lambda_edges_per_tau
            ]
        else:
            # warm start: each split tau group gets the shared lambda edges; unsplit -> [lmin, lmax]
            lpt = [list(lambda_edges) if flags[k] else [lmin, lmax] for k in range(n_tau0)]
        _, r0 = evaluate(tau_edges, None, None, lambda_edges_per_tau=lpt)
        rms0 = float(r0["rms"])
        checkpoint("start", r0)
        tau, lpt, best = _block_fixed_point_pg(
            tau_edges,
            lpt,
            evaluate,
            opt_tau=opt_tau,
            opt_lambda=opt_lambda,
            lmin=lmin,
            lmax=lmax,
            cfg=cfg,
            budget=budget,
            report=on_step,
        )
        checkpoint("blocks", evaluate(tau, None, None, lambda_edges_per_tau=lpt)[1])
        if grow:
            while (len(tau) - 1) < max_groups and not budget.exhausted():
                cand_tau, cand_lpt = _insert_edge_pg(tau, lpt, min_gap_tau)
                if cand_tau is None:
                    break
                gtol = grow_tol if grow_tol is not None else grow_tol_rel * best
                ct, clpt, cbest = _block_fixed_point_pg(
                    cand_tau,
                    cand_lpt,
                    evaluate,
                    opt_tau=opt_tau,
                    opt_lambda=opt_lambda,
                    lmin=lmin,
                    lmax=lmax,
                    cfg=cfg,
                    budget=budget,
                    report=on_step,
                )
                if (best - cbest) > gtol:
                    tau, lpt, best = ct, clpt, cbest
                    budget.reset_plateau(state["n_evals"])  # a real grow shouldn't be cut short by plateau
                    checkpoint("grow", evaluate(tau, None, None, lambda_edges_per_tau=lpt)[1])
                else:
                    break
        final_r = evaluate(tau, None, None, lambda_edges_per_tau=lpt)[1]
        return {
            "tau_edges": [round(float(e), 4) for e in tau],
            "lambda_edges_per_tau": [[round(float(e), 4) for e in x] for x in lpt],
            "per_group_lambda": True,
            "rms": float(final_r["rms"]),
            "rms0": rms0,
            "n_empty": int(final_r.get("n_empty", 0)),
            "n_groups": len(tau) - 1,
            "n_bands_total": int(final_r.get("n_groups", 0)),
            "n_evals": state["n_evals"],
            "elapsed": round(time.perf_counter() - t0, 2),
            "stop_reason": budget.stop_reason or "converged",
            "window": list(final_r.get("window", [])),
            "history": history,
        }

    _, r0 = evaluate(tau_edges, lambda_edges, flags)
    rms0 = float(r0["rms"])
    checkpoint("start", r0)

    tau, lam, flg, best = _block_fixed_point(
        tau_edges,
        lambda_edges,
        flags,
        evaluate,
        opt_tau=opt_tau,
        opt_lambda=opt_lambda,
        opt_flags=opt_flags,
        cfg=cfg,
        budget=budget,
        report=on_step,
    )
    checkpoint("blocks", evaluate(tau, lam, flg)[1])

    if grow:
        while (len(tau) - 1) < max_groups and not budget.exhausted():
            cand_tau, cand_flags = _insert_edge(tau, flg, min_gap_tau)
            if cand_tau is None:
                break
            gtol = grow_tol if grow_tol is not None else 0.01 * best
            ct, cl, cf, cbest = _block_fixed_point(
                cand_tau,
                lam,
                cand_flags,
                evaluate,
                opt_tau=opt_tau,
                opt_lambda=opt_lambda,
                opt_flags=opt_flags,
                cfg=cfg,
                budget=budget,
                report=on_step,
            )
            if (best - cbest) > gtol:
                tau, lam, flg, best = ct, cl, cf, cbest
                budget.reset_plateau(state["n_evals"])  # a real grow shouldn't be cut short by plateau
                checkpoint("grow", evaluate(tau, lam, flg)[1])
            else:
                break

    final_r = evaluate(tau, lam, flg)[1]
    return {
        "tau_edges": [round(float(e), 4) for e in tau],
        "lambda_edges": [round(float(e), 4) for e in lam],
        "flags": [bool(b) for b in flg],
        "rms": float(final_r["rms"]),
        "rms0": rms0,
        "n_empty": int(final_r.get("n_empty", 0)),
        "n_groups": len(tau) - 1,
        "n_evals": state["n_evals"],
        "elapsed": round(time.perf_counter() - t0, 2),
        "stop_reason": budget.stop_reason or "converged",
        "window": list(final_r.get("window", [])),
        "history": history,
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
    method: str = typer.Option("cd", "--method", help="Position search: cd (coordinate descent) | nm (Nelder-Mead)."),
    min_gap_tau: float = typer.Option(MIN_GAP_TAU, "--min-gap-tau"),
    min_gap_lam: float = typer.Option(MIN_GAP_LAM, "--min-gap-lam"),
    min_opacity_delta: float = typer.Option(
        1.0,
        "--min-opacity-delta",
        help="Only split a group into low/mid/high when its bottom opacity max/min >= this (1 = always split).",
    ),
    max_groups: int = typer.Option(8, "--max-groups"),
    max_evals: int = typer.Option(400, "--max-evals"),
    max_seconds: float = typer.Option(1800.0, "--max-seconds"),
    window_lo: float = typer.Option(-5.0, "--window-lo", help="Score rms over log10(tau_Ros) >= this."),
    window_hi: float = typer.Option(4.0, "--window-hi", help="Score rms over log10(tau_Ros) <= this."),
    target_rms: float = typer.Option(0.0, "--target-rms", help="Stop once rms <= this (0 = off)."),
    plateau_evals: int = typer.Option(0, "--plateau-evals", help="Stop if rms stalls this many evals (0 = off)."),
    grow_tol_rel: float = typer.Option(
        0.01, "--grow-tol-rel", help="Grow a tau group only if rms improves > this frac."
    ),
    save_plot: str = typer.Option("", "--save-plot", help="Write a before/after Q/rho plot to this path."),
    save_dat: bool = typer.Option(
        False, "--save-dat/--no-save-dat", help="After optimizing, write the optimized binning's kappa .dat table."
    ),
):
    model_name = qrad_core._model_name(model or None)
    report = qrad_core.validate_model_file(qrad_core.MODELS_DIR / model_name)
    if not report["ok"]:
        raise typer.BadParameter(f"model models/{model_name} is not valid: {report['error']}")

    print("[startup] precomputing invariants (reads the ODF; ~10-30s)...")
    qrad_core.precompute(model_name)

    n_tau = len(tau_bin_edges) - 1
    flags = parse_split_lambda(split_lambda) if split_lambda.strip() else [True] * n_tau
    if len(flags) != n_tau:
        raise typer.BadParameter(f"--split-lambda has {len(flags)} entries, expected {n_tau}")

    print(f"[qrad-opt] atmosphere=models/{model_name} metric={metric} method={method} grow={grow}")
    print(f"[qrad-opt] start: tau={_fmt(tau_bin_edges)} lam={_fmt(lambda_bin_edges)} flags={_flags_str(flags)}")

    def _progress(tag, value, groups, n):
        print(f"  [{n:4d} evals] {tag:8s} rms={value:.4e} groups={groups}")

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
        method=method,
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
        min_opacity_delta=min_opacity_delta,
        on_progress=_progress,
    )

    imp = (result["rms0"] - result["rms"]) / result["rms0"] * 100.0
    print("\n[qrad-opt] DONE")
    print(f"  rms: {result['rms0']:.4e} -> {result['rms']:.4e}  ({imp:+.1f}%)")
    print(f"  tau   = {_fmt(result['tau_edges'])}   ({result['n_groups']} tau groups)")
    if result.get("per_group_lambda"):
        print(f"  per-group lambda ({result['n_bands_total']} (tau,lambda) groups), n_empty={result['n_empty']}:")
        for k, lp in enumerate(result["lambda_edges_per_tau"]):
            cut = "no split" if len(lp) <= 2 else f"cuts at {_fmt(lp[1:-1])}"
            print(f"    tau{k}: {cut}")
    else:
        print(f"  lambda= {_fmt(result['lambda_edges'])}")
        print(f"  flags = {_flags_str(result['flags'])}   n_empty={result['n_empty']}")
    print(f"  {result['n_evals']} evals in {result['elapsed']}s")

    if save_plot:
        after = (
            {"lpt": result["lambda_edges_per_tau"]}
            if result.get("per_group_lambda")
            else {"lam": result["lambda_edges"], "flags": result["flags"]}
        )
        _plot_before_after(
            tau_bin_edges,
            lambda_bin_edges,
            flags,
            result["tau_edges"],
            after,
            save_plot,
            model_name,
            min_opacity_delta=min_opacity_delta,
        )
        print(f"  before/after plot -> {save_plot}")

    if save_dat:
        if result.get("per_group_lambda"):
            written, _name = qrad_core.save_kappa_dat(
                result["tau_edges"],
                None,
                None,
                model_name,
                lambda_edges_per_tau=result["lambda_edges_per_tau"],
                min_opacity_delta=min_opacity_delta,
            )
        else:
            written, _name = qrad_core.save_kappa_dat(
                result["tau_edges"],
                result["lambda_edges"],
                result["flags"],
                model_name,
                min_opacity_delta=min_opacity_delta,
            )
        print(f"  kappa table -> {written}")


def _fmt(edges) -> str:
    return "[" + ", ".join(f"{float(e):.4g}" for e in edges) + "]"


def _plot_before_after(tau0, lam0, flags0, tau1, after, path, model=None, min_opacity_delta=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _kw = {} if min_opacity_delta is None else {"min_opacity_delta": min_opacity_delta}
    a = qrad_core.score_binning(tau0, lam0, qrad_core.resolve_flags(flags0, len(tau0) - 1), model, **_kw)
    if "lpt" in after:
        b = qrad_core.score_binning(tau1, None, None, model, lambda_edges_per_tau=after["lpt"], **_kw)
    else:
        b = qrad_core.score_binning(
            tau1, after["lam"], qrad_core.resolve_flags(after["flags"], len(tau1) - 1), model, **_kw
        )
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


if __name__ == "__main__":
    app()
