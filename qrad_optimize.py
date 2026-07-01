"""Q_rad-driven binning optimizer.

Searches over the opacity binning to directly MINIMIZE the Q_rad rms residual against
the full-ODF reference — unlike tausort's `optimize_tau_bin_edges`, which maximizes a
proxy (per-group high-segment overlap). As we saw in the webapp, the proxy and the real
target disagree, so this optimizes the metric that actually matters.

Decision variables (chosen scope): the interior tau edges (count may grow), the interior
lambda-cell edge positions, and the per-tau-group split flags. The outer tau/lambda
window is held fixed, as is the number of lambda cells. It optimizes for a single,
selectable star.

Search is block-coordinate: alternate (A) coordinate descent on tau edges, (B) greedy
split-flag flips, (C) coordinate descent on lambda edges, to a fixed point; then
optionally grow the tau-group count by inserting an edge, accepting only real rms
improvements. Each evaluation is a full RTE solve (~3 s via `qrad_core.score_binning`),
so this is a run-and-wait / batch tool, bounded by an eval + wall-clock budget.

CLI:  uv run python qrad_optimize.py --help
"""

from __future__ import annotations

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
def make_evaluator(star, *, metric="rms", empty_penalty=EMPTY_PENALTY, score_fn=None, on_eval=None):
    """Return (evaluate, state). `evaluate(tau, lam, flags) -> (cost, raw_dict)`.

    cost = base_metric * (1 + empty_penalty * n_empty). The multiplicative empty-band
    penalty counteracts the fact that an empty band is silently dropped from the Q sum
    (which would otherwise *lower* rms and let the search collapse groups). `score_fn`
    is dependency-injected (defaults to qrad_core.score_binning) so the search logic is
    unit-testable against an analytic objective with no ODF. `state["n_evals"]` counts calls.
    `on_eval(n_evals, cost, raw_dict)` (optional) fires after every evaluation — used by the
    webapp for a live progress ticker.
    """
    score = score_fn if score_fn is not None else qrad_core.score_binning
    state = {"n_evals": 0}
    _key = {"rms": "rms", "maxabs": "max_abs", "int_q": "int_q_pct"}[metric]

    def evaluate(tau_edges, lambda_edges, flags):
        r = score(list(tau_edges), list(lambda_edges), list(flags), star)
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

    def exhausted(self) -> bool:
        if self.should_stop is not None and self.should_stop():
            return True
        return self.state["n_evals"] >= self.max_evals or (time.perf_counter() - self.t0) >= self.max_seconds


@dataclass
class _Cfg:
    method: str = "cd"
    min_gap_tau: float = MIN_GAP_TAU
    min_gap_lam: float = MIN_GAP_LAM
    adjust_steps: tuple = ADJUST_STEPS
    max_sweeps: int = 6
    max_block_rounds: int = 4
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
        if opt_tau:
            tau, best = _optimize_positions(
                tau, lambda e: evaluate(e, lam, flags)[0], cfg=cfg, min_gap=cfg.min_gap_tau, budget=budget
            )
            if report:
                report("tau", best, len(tau) - 1)
        if opt_flags and n_lambda > 1:
            flags, best = _flag_search(tau, lam, flags, lambda t, ll, f: evaluate(t, ll, f)[0], budget=budget)
            if report:
                report("flags", best, len(tau) - 1)
        if opt_lambda and len(lam) > 2:
            lam, best = _optimize_positions(
                lam, lambda e: evaluate(tau, e, flags)[0], cfg=cfg, min_gap=cfg.min_gap_lam, budget=budget
            )
            if report:
                report("lambda", best, len(tau) - 1)
        if budget.exhausted() or (start - best) <= cfg.block_tol * max(abs(start), 1.0):
            break
    return tau, lam, flags, best


# --- public API -----------------------------------------------------------------
def optimize_qrad(
    tau_edges,
    lambda_edges,
    *,
    flags=None,
    star="G_SSD",
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
    grow_tol=None,
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
    evaluate, state = make_evaluator(
        star, metric=metric, empty_penalty=empty_penalty, score_fn=score_fn, on_eval=on_eval
    )
    budget = _Budget(max_evals=max_evals, max_seconds=max_seconds, state=state, t0=t0, should_stop=should_stop)
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
    split_lambda: str = typer.Option("", "--split-lambda", help="Per-tau-group 0/1 split flags (default all-on)."),
    star: str = typer.Option("G_SSD", "--star", help="STAGGER model atmosphere to optimize for."),
    opt_tau: bool = typer.Option(True, "--opt-tau/--no-opt-tau"),
    opt_lambda: bool = typer.Option(True, "--opt-lambda/--no-opt-lambda"),
    opt_flags: bool = typer.Option(True, "--opt-flags/--no-opt-flags"),
    grow: bool = typer.Option(True, "--grow/--no-grow", help="Allow growing the tau-group count."),
    metric: str = typer.Option("rms", "--metric", help="Objective: rms | maxabs | int_q."),
    method: str = typer.Option("cd", "--method", help="Position search: cd (coordinate descent) | nm (Nelder-Mead)."),
    min_gap_tau: float = typer.Option(MIN_GAP_TAU, "--min-gap-tau"),
    min_gap_lam: float = typer.Option(MIN_GAP_LAM, "--min-gap-lam"),
    max_groups: int = typer.Option(8, "--max-groups"),
    max_evals: int = typer.Option(400, "--max-evals"),
    max_seconds: float = typer.Option(1800.0, "--max-seconds"),
    save_plot: str = typer.Option("", "--save-plot", help="Write a before/after Q/rho plot to this path."),
):
    print("[startup] precomputing invariants (reads the ODF; ~10-30s)...")
    qrad_core.precompute()

    n_tau = len(tau_bin_edges) - 1
    flags = parse_split_lambda(split_lambda) if split_lambda.strip() else [True] * n_tau
    if len(flags) != n_tau:
        raise typer.BadParameter(f"--split-lambda has {len(flags)} entries, expected {n_tau}")

    print(f"[qrad-opt] star={star} metric={metric} method={method} grow={grow}")
    print(f"[qrad-opt] start: tau={_fmt(tau_bin_edges)} lam={_fmt(lambda_bin_edges)} flags={_flags_str(flags)}")

    def _progress(tag, value, groups, n):
        print(f"  [{n:4d} evals] {tag:8s} rms={value:.4e} groups={groups}")

    result = optimize_qrad(
        tau_bin_edges,
        lambda_bin_edges,
        flags=flags,
        star=star,
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
        on_progress=_progress,
    )

    imp = (result["rms0"] - result["rms"]) / result["rms0"] * 100.0
    print("\n[qrad-opt] DONE")
    print(f"  rms: {result['rms0']:.4e} -> {result['rms']:.4e}  ({imp:+.1f}%)")
    print(f"  tau   = {_fmt(result['tau_edges'])}   ({result['n_groups']} groups)")
    print(f"  lambda= {_fmt(result['lambda_edges'])}")
    print(f"  flags = {_flags_str(result['flags'])}   n_empty={result['n_empty']}")
    print(f"  {result['n_evals']} evals in {result['elapsed']}s")

    if save_plot:
        _plot_before_after(
            tau_bin_edges,
            lambda_bin_edges,
            flags,
            result["tau_edges"],
            result["lambda_edges"],
            result["flags"],
            star,
            save_plot,
        )
        print(f"  before/after plot -> {save_plot}")


def _fmt(edges) -> str:
    return "[" + ", ".join(f"{float(e):.4g}" for e in edges) + "]"


def _plot_before_after(tau0, lam0, flags0, tau1, lam1, flags1, star, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = qrad_core.score_binning(tau0, lam0, qrad_core.resolve_flags(flags0, len(tau0) - 1), star)
    b = qrad_core.score_binning(tau1, lam1, qrad_core.resolve_flags(flags1, len(tau1) - 1), star)
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
    ax[0].set_title(f"Q_rad binning optimization ({star})")
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
