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


def _tree_position_search(tree, cost_tree, *, cfg, budget, min_gap_tau, min_gap_lam):
    """Gauss-Seidel coordinate descent on every internal cut's position. Each candidate must
    stay in its node's axis span (per-axis min gap) and keep the whole tree feasible; the best
    strict improvement per node is committed in place."""
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
        if (best_c - round_best_c) <= grow_tol:
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
            base = copy.deepcopy(state["tree"])
            _remove_node_at_path(base["root"], rmpath)  # merge two leaves -> frees one leaf slot
            if try_splits(base):
                improved = True
                break
    return state["tree"], state["c"]


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
    tree=False,  # general 2D guillotine mode (both tau and lambda locally free)
    binning_tree=None,  # guillotine-tree warm start {window_tau, window_lam, root}
    # beam-search knobs (non-greedy tree-topology search; used when beam_width >= 2):
    beam_width=3,  # rival tree topologies kept in parallel each round
    beam_positions=(0.35, 0.5, 0.65),  # split-position fractions tried per (leaf, axis)
    beam_leaves=4,  # widest leaves considered for splitting, per beam tree
    min_opacity_delta=None,  # min bottom-opacity max/min ratio to split a group (None -> score default)
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

    # The optimizer is tree-only: every grouping (explicit guillotine tree, per-tau-group
    # lambda, or shared tau + split flags) is normalized to a guillotine tree and refined via a
    # single seed/grow/polish path. The shim below seeds the working tree from whichever input
    # was passed; every result is a {"tree": True, "binning_tree": ...} dict.
    if binning_tree is not None:
        btree = copy.deepcopy(binning_tree)
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

    _best_cost = float("inf")

    def cost_tree(t):
        # Wrapped evaluator: fires `on_improve` on every strictly better binning (new global-best
        # penalized cost). One hook covers grow, beam, and polish since they all route here.
        nonlocal _best_cost
        c, r = evaluate(binning_tree=t)
        if on_improve is not None and c < _best_cost - 1e-12:
            _best_cost = c
            on_improve(copy.deepcopy(t), r, state["n_evals"])
        return c

    rms0 = float(evaluate(binning_tree=btree)[1]["rms"])  # rms of the user's seed binning

    def _refine(seed_tree):
        """grow -> polish -> topology search from one seed tree. Returns (tree, penalized cost).
        Captures cfg/budget/cost_tree/on_step/checkpoint from the enclosing scope."""
        tree = copy.deepcopy(seed_tree)
        best = cost_tree(tree)
        checkpoint("start", evaluate(binning_tree=tree)[1])
        # Grow FIRST so the budget builds structure (each grow cheaply refines only its new cut);
        # a heavy refine of the coarse seed up front would exhaust the budget before a leaf is split.
        if grow:
            gtol = grow_tol if grow_tol is not None else 0.0
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
                    report=on_step,
                )
                budget.reset_plateau(state["n_evals"])
                checkpoint("grow", evaluate(binning_tree=tree)[1])
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
            return {"axis": "tau", "at": float(tau_pts[c]),
                    "lo": _reconstruct_guillotine(lo, tau_pts, lam_pts),
                    "hi": _reconstruct_guillotine(hi, tau_pts, lam_pts)}
    k0_min = min(r[2] for r in leaves)
    for c in sorted({r[3] for r in leaves}):  # lambda cut: low group's top edge at c
        if c <= k0_min:
            continue
        lo = {r for r in leaves if r[3] <= c}
        hi = {r for r in leaves if r[2] >= c}
        if lo and hi and len(lo) + len(hi) == len(leaves):
            return {"axis": "lam", "at": float(lam_pts[c]),
                    "lo": _reconstruct_guillotine(lo, tau_pts, lam_pts),
                    "hi": _reconstruct_guillotine(hi, tau_pts, lam_pts)}
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
    print(f"[mem] after precompute: Q.shape={Q.shape} dtype={str(Q.dtype)} "
          f"nbytes={Q.nbytes/1e6:.1f}MB mmap={isinstance(Q, np.memmap)} "
          f"rect_index={len(rect_index)} rect_index_deep={_deep_size_mb(rect_index):.1f}MB "
          f"RSS={_rss_mb():.0f}MB", flush=True)
    ref = qrad_core.reference(model)
    q_full = np.asarray(ref["q_full"])
    rho = np.asarray(ref["rho"])
    ltau = np.asarray(ref["ltau"])
    win = qrad_core.WINDOW if window is None else (float(window[0]), float(window[1]))
    in_win = (ltau >= min(win)) & (ltau <= max(win))
    print(f"[mem] ref arrays: q_full={q_full.nbytes/1e6:.2f}MB rho={rho.nbytes/1e6:.2f}MB "
          f"ltau={ltau.nbytes/1e6:.2f}MB nz={len(ltau)} in_win={int(in_win.sum())} "
          f"RSS={_rss_mb():.0f}MB", flush=True)

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
    print(f"[mem] === pre-pass: building memo for k=1..{max_groups} (ntau={ntau} nlam={nlam}) ===",
          flush=True)
    total_tilings = 0
    for _k in range(1, max_groups + 1):
        _d = pexact(0, ntau - 1, 0, nlam - 1, _k)
        _n_full = len(_d)
        total_tilings += _n_full
        _n_states = len(memo)
        _n_sigs = sum(len(v) for v in memo.values())
        print(f"[mem] k={_k}: full_rect_tilings={_n_full} cumulative={total_tilings} "
              f"memo_states={_n_states} memo_sigs={_n_sigs} RSS={_rss_mb():.0f}MB", flush=True)
    print(f"[grid] scoring {total_tilings} distinct tilings (k=1..{max_groups})", flush=True)
    pbar = tqdm(total=total_tilings, desc="grid tilings", unit="tiling")
    for k in range(1, max_groups + 1):
        for sig in pexact(0, ntau - 1, 0, nlam - 1, k):
            rms = rms_of(sig)
            n_tilings += 1
            pbar.update(1)
            if n_tilings % 200000 == 0:
                print(f"[mem] loop: n_tilings={n_tilings} memo_states={len(memo)} "
                      f"topk={len(topk)} RSS={_rss_mb():.0f}MB", flush=True)
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
                            {"window_tau": list(win_tau), "window_lam": list(win_lam),
                             "root": copy.deepcopy(tree_from_sig(sig))},
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
        tree_i = {"window_tau": list(win_tau), "window_lam": list(win_lam),
                  "root": copy.deepcopy(tree_from_sig(sig))}
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
    max_evals: int = typer.Option(400, "--max-evals"),
    max_seconds: float = typer.Option(1800.0, "--max-seconds"),
    window_lo: float = typer.Option(-1.0, "--window-lo", help="Score rms over log10(tau_Ros) >= this."),
    window_hi: float = typer.Option(2.0, "--window-hi", help="Score rms over log10(tau_Ros) <= this."),
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
    use_grid_search: bool = typer.Option(
        False,
        "--grid-search/--no-grid-search",
        help="Exhaustive grid search over tau-then-lambda tilings (cuts on a dtau/dlam grid). "
        "Precomputes every rectangle's Q once, then enumerates ALL tilings up to --max-groups -> "
        "exact grid optimum. Replaces the beam/multi-start heuristic.",
    ),
    dtau: float = typer.Option(0.5, "--dtau", help="Grid step in -log10(tau) for --grid-search."),
    dlam: float = typer.Option(0.25, "--dlam", help="Grid step in log10(lambda) for --grid-search."),
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

    def _on_improve(tree, r, n_evals):
        if plot_dir is None:
            return
        sig = _tree_signature(tree)  # dedupe identical tilings (e.g. sub-1e-3 position wiggles)
        if sig in _plot_seen:
            return
        _plot_seen.add(sig)
        _plot_n[0] += 1
        out = plot_dir / f"step_{n_evals:04d}_rms_{float(r['rms']):.3e}.png"
        _plot_tree_binning(
            tree,
            out,
            rms=float(r["rms"]),
            n_empty=int(r.get("n_empty", 0)),
            groups=int(r.get("n_groups", 0)),
            n_evals=n_evals,
            seq=_plot_n[0],
            model=model_name,
        )

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
            on_progress=_progress,
            on_improve=_on_improve if plot_dir else None,
        )

    imp = (result["rms0"] - result["rms"]) / result["rms0"] * 100.0
    print("\n[qrad-opt] DONE")
    print(f"  rms: {result['rms0']:.4e} -> {result['rms']:.4e}  ({imp:+.1f}%)")
    print(f"  general-2D tree: {result['n_leaves']} leaf bands, n_empty={result['n_empty']}")
    print("    " + _tree_bands_str(result["binning_tree"]))
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
        )
        print(f"  final binning -> {final_plot}")

    if save_plot:
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


def _plot_tree_binning(tree, path, *, rms=None, n_empty=None, groups=None, n_evals=None, seq=None, model=None):
    """Render a binning tree's tau-lambda leaf rectangles to `path` (one patch per group)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rects = list(_leaf_rects(tree["root"], _root_rect(tree)))
    wt, wl = tree["window_tau"], tree["window_lam"]
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(7.5, 6))
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
