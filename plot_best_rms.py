#!/usr/bin/env python3
"""
Log-linear plot of best achievable RMS residual as a function of the number
of tau-lambda opacity bins (data from best.txt).
"""

import matplotlib.pyplot as plt
import numpy as np

n_bins, rms = np.loadtxt("best.txt", delimiter=",", skiprows=1).T

fig, ax = plt.subplots(figsize=(7.5, 5.2))

ax.semilogy(n_bins, rms, "o-", color="#1f5fa8", linewidth=1.8,
            markersize=7.5, markerfacecolor="#3d8bea",
            markeredgecolor="#1f5fa8", markeredgewidth=1.4)

ax.set_xlabel(r"Number of $\tau\!-\!\lambda$ bins", fontsize=13)
ax.set_ylabel("Best RMS residual", fontsize=13)

ax.set_xticks(n_bins)
ax.set_xlim(min(n_bins) - 0.5, max(n_bins) + 0.5)
ax.grid(True, which="major", linestyle="-", alpha=0.35)
ax.grid(True, which="minor", linestyle=":", alpha=0.2)
ax.tick_params(labelsize=11)

# Small table of exact values in the bottom-left of the axes
cell_text = [[f"{y:.4e}"] for y in rms]
tbl = ax.table(cellText=cell_text, rowLabels=[str(int(n)) for n in n_bins],
               colLabels=["best RMS"], loc="lower left", cellLoc="center",
               colWidths=[0.2])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.25)

# Set table background white and fully opaque
for key, cell in tbl.get_celld().items():
    cell.set_facecolor('white')
    cell.set_alpha(1)

fig.tight_layout()
out = "best_rms_vs_nbins.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"✓ saved {out}")
