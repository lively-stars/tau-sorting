"""Draw the staged-seeding tiling sequence: (1) N equally-spaced tau cuts,
(2) tau positions optimized, (3) lambda split of the bottom two tau groups
+ joint position wiggle. Data-free: builds guillotine trees via
qrad_optimize.tree_from_splits and renders leaf rectangles."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import qrad_optimize as qo

TAU_WINDOW = (-0.63, 7.0)
LAM_WINDOW = (3.0, 5.0)
N_INITIAL = 3
LAM_CUT = 3.8

tlo, thi = TAU_WINDOW
llo, lhi = LAM_WINDOW


def equal_tau(n):
    step = (thi - tlo) / n
    return [tlo + step * k for k in range(1, n)]


def tree_phase1():
    # N equally spaced tau cuts, full lambda width
    splits = [{"axis": "tau", "tau": t, "lam": 0.5 * (llo + lhi)} for t in equal_tau(N_INITIAL)]
    return qo.tree_from_splits(list(TAU_WINDOW), list(LAM_WINDOW), splits)


def tree_phase2():
    # tau positions optimally distributed (denser at deep/bottom tau); mock optimum
    splits = [
        {"axis": "tau", "tau": t, "lam": 0.5 * (llo + lhi)}
        for t in [0.35, 2.90]  # wiggled from (1.91, 4.46)
    ]
    return qo.tree_from_splits(list(TAU_WINDOW), list(LAM_WINDOW), splits)


def tree_phase3():
    # lambda cut at LAM_CUT spanning the bottom two tau groups + tau wiggle
    t0, t1 = 0.50, 3.10
    splits = [
        {"axis": "tau", "tau": t0, "lam": 0.5 * (llo + lhi)},
        {"axis": "tau", "tau": t1, "lam": 0.5 * (llo + lhi)},
        {"axis": "lam", "tau": 0.5 * (tlo + t0), "lam": LAM_CUT},  # bottom group
        {"axis": "lam", "tau": 0.5 * (t0 + t1), "lam": LAM_CUT},  # second-bottom group
    ]
    return qo.tree_from_splits(list(TAU_WINDOW), list(LAM_WINDOW), splits)


def draw(ax, tree, title):
    root_rect = (tlo, thi, llo, lhi)
    rects = list(qo._leaf_rects(tree["root"], root_rect))
    colors = plt.colormaps["tab10"].colors
    for i, (a, b, c, d) in enumerate(rects):
        ax.add_patch(Rectangle((c, a), d - c, b - a, facecolor=colors[i % 10], edgecolor="black", lw=1.5, alpha=0.55))
        ax.text(0.5 * (c + d), 0.5 * (a + b), f"{i}", ha="center", va="center", fontsize=13)
    ax.set_xlim(llo, lhi)
    ax.set_ylim(tlo, thi)
    ax.set_xlabel(r"log10 $\lambda$ [A]")
    ax.set_ylabel(r"-log10 $\tau$")
    ax.set_title(title, fontsize=9)


fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
draw(axes[0], tree_phase1(), f"(1) equal tau cuts [{3} groups]")
draw(axes[1], tree_phase2(), "(2) tau optimized [3 groups]")
draw(axes[2], tree_phase3(), "(3) bottom-two lam-split [5 groups]")
fig.suptitle("Staged seeding: tau cuts -> tau wiggle -> bottom-two lambda split")
fig.tight_layout()
fig.savefig("plots/staged_seeding.png", dpi=120, bbox_inches="tight")
print("wrote plots/staged_seeding.png")
