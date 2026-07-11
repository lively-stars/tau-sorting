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
a single seed/grow/polish path. `method` selects the *grow* strategy — `"beam"` (non-greedy
beam search over tree topologies) vs the default greedy grow (one midpoint split per round,
committed immediately); `cd`/`nm` behave as greedy. Each evaluation is a full RTE solve (~3 s
via `qrad_core.score_binning`), so this is a run-and-wait / batch tool, bounded by an eval +
wall-clock budget.

In tree mode, `method="beam"` swaps the greedy grow (one midpoint split, committed
immediately, no backtracking) for a beam search that keeps rival topologies alive and tries
several split positions per leaf — a bounded exploration of the tiling space rather than a
single greedy trajectory.

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
    method: str = "cd"
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


def tree_from_flags(tau_edges, lambda_edges, flags) -> dict:
    lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
    lpt = [list(lambda_edges) if flags[k] else [lmin, lmax] for k in range(len(tau_edges) - 1)]
    return tree_from_lpt(tau_edges, lpt)


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
    """Non-greedy beam search over guillotine-tree topologies (``method="beam"``).

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
    # method="beam" knobs (non-greedy tree-topology search; inert unless tree mode + grow):
    beam_width=3,  # rival tree topologies kept in parallel each round
    beam_positions=(0.35, 0.5, 0.65),  # split-position fractions tried per (leaf, axis)
    beam_leaves=4,  # widest leaves considered for splitting, per beam tree
    min_opacity_delta=None,  # min bottom-opacity max/min ratio to split a group (None -> score default)
    score_fn=None,
    on_progress=None,
    on_eval=None,
    should_stop=None,
) -> dict:
    """Minimize the Q_rad residual over the binning. Returns a tree result dict with the
    optimized `binning_tree`, `rms`/`rms0`, `n_evals`, `elapsed`, `n_empty`, and a `history`
    of (n_evals, rms, groups) checkpoints.

    Tree-only: every input shape (an explicit `binning_tree`, `lambda_edges_per_tau`,
    `per_group_lambda`, or shared tau + `flags`) is normalized to a guillotine tree and refined
    via a single seed/grow/polish path. `grow` enables splitting leaves (accepted only when rms
    improves by > grow_tol, default an absolute 0); `method="beam"` swaps the greedy grow for a
    non-greedy beam search over tree topologies (cd/nm behave as greedy). The opt_tau/
    opt_lambda/opt_flags toggles are retained for API compatibility but are inert under the
    tree path.

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

    # The optimizer is tree-only: every grouping (explicit guillotine tree, per-tau-group
    # lambda, or shared tau + split flags) is normalized to a guillotine tree and refined via a
    # single seed/grow/polish path. The shim below seeds the working tree from whichever input
    # was passed; every result is a {"tree": True, "binning_tree": ...} dict.
    if binning_tree is not None:
        btree = copy.deepcopy(binning_tree)
    elif lambda_edges_per_tau is not None:
        btree = tree_from_lpt(tau_edges, [list(x) for x in lambda_edges_per_tau])
    elif per_group_lambda:
        btree = tree_from_lpt(tau_edges, [list(lambda_edges) for _ in range(n_tau0)])
    else:
        lmin, lmax = float(lambda_edges[0]), float(lambda_edges[-1])
        btree = tree_from_lpt(tau_edges, [list(lambda_edges) if bool(f) else [lmin, lmax] for f in flags])

    def cost_tree(t):
        return evaluate(binning_tree=t)[0]

    _, r0 = evaluate(binning_tree=btree)
    rms0 = float(r0["rms"])
    best = cost_tree(btree)
    checkpoint("start", r0)
    # Grow FIRST so the budget builds structure (each grow cheaply refines only its new cut).
    # A heavy refine of the coarse seed up front would, at RTE cost, exhaust the budget before
    # a single leaf is split — leaving a plain grid. Grow lets tau/lambda cuts nest to any
    # depth (a tau sub-split inside one lambda region == the non-grid tiling we want).
    if grow:
        gtol = grow_tol if grow_tol is not None else 0.0
        if cfg.method == "beam":
            # Non-greedy: keep rival topologies alive (beam search) instead of committing the
            # single best midpoint split. Explores a beam-bounded slice of the tiling space;
            # the final _block_fixed_point_tree polish below optimizes the winner's positions.
            btree, best = _beam_grow_tree(
                btree,
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
            checkpoint("grow", evaluate(binning_tree=btree)[1])
        else:
            # Greedy: one midpoint split per round, committed immediately (no backtracking).
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
                # A guillotine split adds DOF so it almost always helps a little; the empty-band
                # penalty + min-gap still reject degenerate/empty splits (those raise the cost).
                if (best - cbest) > gtol:
                    btree, best = ctree, cbest
                    budget.reset_plateau(state["n_evals"])
                    checkpoint("grow", evaluate(binning_tree=btree)[1])
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
        checkpoint("blocks", evaluate(binning_tree=btree)[1])
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
    method: str = typer.Option(
        "cd",
        "--method",
        help="Grow strategy: beam (non-greedy beam search over guillotine-tree topologies) "
        "vs the default greedy grow (one midpoint split per round); cd/nm behave as greedy "
        "(the optimizer is tree-only).",
    ),
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
    tree: bool = typer.Option(False, "--tree/--no-tree", help="General 2D guillotine mode (required for method=beam)."),
    beam_width: int = typer.Option(
        3, "--beam-width", help="method=beam: rival tree topologies kept in parallel each round."
    ),
    beam_leaves: int = typer.Option(
        4, "--beam-leaves", help="method=beam: widest leaves considered for splitting, per beam tree."
    ),
    beam_positions: list[float] = typer.Option(
        [0.35, 0.5, 0.65], "--beam-positions", help="method=beam: split-position fractions tried per (leaf, axis)."
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
        tree=tree,
        beam_width=beam_width,
        beam_positions=tuple(beam_positions),
        beam_leaves=beam_leaves,
        min_opacity_delta=min_opacity_delta,
        on_progress=_progress,
    )

    imp = (result["rms0"] - result["rms"]) / result["rms0"] * 100.0
    print("\n[qrad-opt] DONE")
    print(f"  rms: {result['rms0']:.4e} -> {result['rms']:.4e}  ({imp:+.1f}%)")
    if result.get("tree"):
        print(f"  general-2D tree: {result['n_leaves']} leaf bands, n_empty={result['n_empty']}")
        print("    " + _tree_bands_str(result["binning_tree"]))
    else:
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
        if result.get("tree"):
            after = {"tree": result["binning_tree"]}
            seed_tree = tree_from_lpt(
                tau_bin_edges,
                [[lambda_bin_edges[0], lambda_bin_edges[-1]] for _ in range(len(tau_bin_edges) - 1)],
            )
            _plot_before_after(
                tau_bin_edges,
                lambda_bin_edges,
                flags,
                tau_bin_edges,
                after,
                save_plot,
                model_name,
                min_opacity_delta=min_opacity_delta,
                before_tree=seed_tree,
            )
        else:
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
        if result.get("tree"):
            written, _name = qrad_core.save_kappa_dat(
                None,
                None,
                None,
                model_name,
                binning_tree=result["binning_tree"],
                min_opacity_delta=min_opacity_delta,
            )
        elif result.get("per_group_lambda"):
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


def _tree_bands_str(tree) -> str:
    """One line per leaf rectangle (tau window, lambda window) — the general-2D bands."""
    rects = list(_leaf_rects(tree["root"], _root_rect(tree)))
    return "\n    ".join(f"tau[{tlo:.3f},{thi:.3f}] lam[{llo:.3f},{lhi:.3f}]" for tlo, thi, llo, lhi in rects)


def _plot_before_after(tau0, lam0, flags0, tau1, after, path, model=None, min_opacity_delta=None, before_tree=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _kw = {} if min_opacity_delta is None else {"min_opacity_delta": min_opacity_delta}
    if before_tree is not None:
        a = qrad_core.score_binning(None, None, None, model, binning_tree=before_tree, **_kw)
    else:
        a = qrad_core.score_binning(tau0, lam0, qrad_core.resolve_flags(flags0, len(tau0) - 1), model, **_kw)
    if "tree" in after:
        b = qrad_core.score_binning(None, None, None, model, binning_tree=after["tree"], **_kw)
    elif "lpt" in after:
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
