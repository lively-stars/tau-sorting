from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib
import numpy as np

from kappa_band_reader import read_kappa_4_band_comparison

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _spaced_indices(size: int, count: int) -> np.ndarray:
    if size < count:
        raise ValueError(f"Cannot pick {count} indices from axis of length {size}")
    return np.linspace(0, size - 1, count, dtype=int)


def _load_mixed_comparison(
    comparison_path: str | Path, expected_nbands: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    comp = np.load(comparison_path, allow_pickle=True)
    rec = comp[()]
    if rec.dtype.names is None:
        raise ValueError("Comparison file must be a structured array with named fields")
    for field in ("T", "p", "mixed"):
        if field not in rec.dtype.names:
            raise ValueError(f"Comparison file missing '{field}' field")

    t_axis = np.asarray(rec["T"], dtype=np.float64)
    p_axis = np.asarray(rec["p"], dtype=np.float64)
    mixed_raw = np.asarray(rec["mixed"], dtype=np.float64)

    if mixed_raw.ndim != 3:
        raise ValueError(
            f"Expected mixed to be 3D; got shape {mixed_raw.shape}"
        )

    # Accept either [nt, np, nbands] (tausort.py output) or [nbands, nt, np].
    if mixed_raw.shape[-1] == expected_nbands:
        mixed_nt_np_nb = mixed_raw
    elif mixed_raw.shape[0] == expected_nbands:
        mixed_nt_np_nb = np.transpose(mixed_raw, (1, 2, 0))
    else:
        raise ValueError(
            f"Could not infer mixed band axis for shape {mixed_raw.shape} and nbands={expected_nbands}"
        )

    if mixed_nt_np_nb.shape[0] != t_axis.shape[0]:
        raise ValueError(
            "Comparison mixed T axis does not match comparison T coordinate length"
        )
    if mixed_nt_np_nb.shape[1] != p_axis.shape[0]:
        raise ValueError(
            "Comparison mixed p axis does not match comparison p coordinate length"
        )
    if mixed_nt_np_nb.shape[2] != expected_nbands:
        raise ValueError("Comparison mixed band count does not match kap_mean bands")

    return t_axis, p_axis, mixed_nt_np_nb


def plot_kap_mean_grid(
    input_path: str | Path,
    output_path: str | Path = "kap_mean_grid_4x3.png",
    *,
    endian: str = "<",
    comparison_path: Optional[str | Path] = "tau_bin_opacities.npy",
) -> Path:
    """
    Create a 4x3 grid of kap_mean plots at 12 different (T, p) points.
    Each panel plots kap_mean over band index and optionally overlays
    tau_bin_opacities.npy mixed data at the closest matching T/p point.
    """
    data = read_kappa_4_band_comparison(input_path, endian=endian)
    t_indices = _spaced_indices(data.NT, 4)
    p_indices = _spaced_indices(data.Np, 3)
    band_idx = np.arange(data.Nbands_out)
    source_label = "kap_mean"

    compare_t = compare_p = compare_mixed = None
    if comparison_path is not None and Path(comparison_path).exists():
        compare_t, compare_p, compare_mixed = _load_mixed_comparison(
            comparison_path, data.Nbands_out
        )
        source_label = "kap_mean vs ln(mixed)"

    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True, sharey=True)

    for row, t_i in enumerate(t_indices):
        for col, p_i in enumerate(p_indices):
            ax = axes[row, col]
            y_kap = data.kap_mean[:, t_i, p_i]
            ax.plot(
                band_idx,
                y_kap,
                marker="o",
                linewidth=1.6,
                color="tab:blue",
                label="kap_mean",
            )

            if compare_mixed is not None and compare_t is not None and compare_p is not None:
                t_lin = float(10.0 ** data.tab_T[t_i])
                p_lin = float(10.0 ** data.tab_p[p_i])
                t_cmp_i = int(np.argmin(np.abs(compare_t - t_lin)))
                p_cmp_i = int(np.argmin(np.abs(compare_p - p_lin)))
                y_cmp = np.log(np.clip(compare_mixed[t_cmp_i, p_cmp_i, :], 1.0e-300, None))
                ax.plot(
                    band_idx,
                    y_cmp,
                    marker="s",
                    linestyle="--",
                    linewidth=1.4,
                    color="tab:orange",
                    label="ln(mixed)",
                )

            ax.grid(True, alpha=0.3)
            ax.set_title(
                f"T[{t_i}]={data.tab_T[t_i]:.3f}, p[{p_i}]={data.tab_p[p_i]:.3f}",
                fontsize=9,
            )
            if row == 3:
                ax.set_xlabel("Band index")
            if col == 0:
                ax.set_ylabel("kap_mean")
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"{source_label} at 12 sampled (T, p) grid points", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(output_path)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a 4x3 grid of kap_mean vectors at sampled (T,p) points."
    )
    parser.add_argument(
        "--input",
        default="kappa_4_band_comparison.dat",
        help="Path to kappa_4_band_comparison.dat",
    )
    parser.add_argument(
        "--output",
        default="kap_mean_grid_4x3.png",
        help="Output image path",
    )
    parser.add_argument(
        "--endian",
        default="<",
        choices=["<", ">"],
        help="Byte order for the binary file",
    )
    parser.add_argument(
        "--comparison",
        default="tau_bin_opacities.npy",
        help="Optional tau_bin_opacities .npy file with fields T,p,mixed",
    )
    parser.add_argument(
        "--no-comparison",
        action="store_true",
        help="Disable overlay of comparison mixed data",
    )
    args = parser.parse_args()

    comparison_path = None if args.no_comparison else args.comparison
    out = plot_kap_mean_grid(
        args.input,
        args.output,
        endian=args.endian,
        comparison_path=comparison_path,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    _main()
