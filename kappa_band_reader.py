from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class KappaBandComparison:
    tau5000bin: int
    NT: int
    Np: int
    Nbands_out: int
    pp_axis: int
    full_odf: int
    scatter_on: int
    back_heating: int
    tab_T: np.ndarray
    tab_p: np.ndarray
    kap_5000: np.ndarray | None
    B_5000: np.ndarray | None
    kap_mean: np.ndarray
    B_band: np.ndarray
    nuout: np.ndarray | None


def _read_exact(fp, dtype: np.dtype, count: int, label: str) -> np.ndarray:
    arr = np.fromfile(fp, dtype=dtype, count=count)
    if arr.size != count:
        raise ValueError(f"Unexpected EOF while reading {label}: expected {count}, got {arr.size}")
    return arr


def read_kappa_4_band_comparison(path: str | Path, endian: str = "<", strict: bool = True) -> KappaBandComparison:
    """
    Read a kappa_4_band_comparison.dat file written by C fwrite calls.

    Parameters
    ----------
    path:
        Binary file path.
    endian:
        Byte order, '<' for little-endian and '>' for big-endian.
    strict:
        If True, fail on trailing bytes that do not match the expected layout.
    """
    file_path = Path(path)
    i4 = np.dtype(f"{endian}i4")
    f8 = np.dtype(f"{endian}f8")
    f4 = np.dtype(f"{endian}f4")

    with file_path.open("rb") as fp:
        header = _read_exact(fp, i4, 8, "header")
        tau5000bin, NT, Np, Nbands_out, pp_axis, full_odf, scatter_on, back_heating = (int(v) for v in header)

        tab_T = _read_exact(fp, f8, NT, "tab_T")
        tab_p = _read_exact(fp, f8, Np, "tab_p")

        kap_5000 = None
        B_5000 = None
        if tau5000bin == 1:
            kap_5000 = _read_exact(fp, f4, NT * Np, "kap_5000").reshape(NT, Np)
            B_5000 = _read_exact(fp, f4, NT, "B_5000")

        kap_mean = _read_exact(fp, f4, Nbands_out * NT * Np, "kap_mean").reshape(Nbands_out, NT, Np)
        B_band = _read_exact(fp, f4, Nbands_out * NT, "B_band").reshape(Nbands_out, NT)

        nuout = None
        if full_odf == 1:
            nuout = np.fromfile(fp, dtype=f4)
            if strict:
                trailing = fp.read()
                if trailing:
                    raise ValueError("Trailing bytes after nuout that are not full float32 values")
        elif strict:
            trailing = fp.read()
            if trailing:
                raise ValueError(f"Unexpected trailing bytes at end of file: {len(trailing)} bytes")

    return KappaBandComparison(
        tau5000bin=tau5000bin,
        NT=NT,
        Np=Np,
        Nbands_out=Nbands_out,
        pp_axis=pp_axis,
        full_odf=full_odf,
        scatter_on=scatter_on,
        back_heating=back_heating,
        tab_T=tab_T,
        tab_p=tab_p,
        kap_5000=kap_5000,
        B_5000=B_5000,
        kap_mean=kap_mean,
        B_band=B_band,
        nuout=nuout,
    )


def write_kappa_4_band_comparison(path: str | Path, data: KappaBandComparison, endian: str = "<") -> None:
    """
    Write a kappa_4_band_comparison.dat file, the exact inverse of
    :func:`read_kappa_4_band_comparison`.

    The byte layout mirrors the C ``tausort`` writer: an 8-int header, the
    temperature/pressure axes, the optional tau=5000 reference section, the
    band-mean opacities, the band-integrated Planck terms, and the optional
    full-ODF frequency grid. Which optional sections are emitted is governed by
    the header flags on ``data`` (``tau5000bin`` and ``full_odf``), exactly as
    the reader keys off them, so any value written here round-trips back through
    the reader unchanged.

    Gray output is not a special mode: it is simply the ``Nbands_out == 1`` case,
    which produces a file byte-identical to the C ``kappa_grey.dat``.

    Parameters
    ----------
    path:
        Destination binary file path.
    data:
        Fully-populated :class:`KappaBandComparison`. Array shapes must match the
        header dimensions; mismatches raise ``ValueError`` via reshape.
    endian:
        Byte order, '<' for little-endian and '>' for big-endian.
    """
    file_path = Path(path)
    i4 = np.dtype(f"{endian}i4")
    f8 = np.dtype(f"{endian}f8")
    f4 = np.dtype(f"{endian}f4")

    NT, Np, Nbands_out = data.NT, data.Np, data.Nbands_out

    header = np.array(
        [
            data.tau5000bin,
            NT,
            Np,
            Nbands_out,
            data.pp_axis,
            data.full_odf,
            data.scatter_on,
            data.back_heating,
        ],
        dtype=i4,
    )

    with file_path.open("wb") as fp:
        header.tofile(fp)
        np.asarray(data.tab_T, dtype=f8).reshape(NT).tofile(fp)
        np.asarray(data.tab_p, dtype=f8).reshape(Np).tofile(fp)

        if data.tau5000bin == 1:
            if data.kap_5000 is None or data.B_5000 is None:
                raise ValueError("tau5000bin=1 requires kap_5000 and B_5000")
            np.asarray(data.kap_5000, dtype=f4).reshape(NT * Np).tofile(fp)
            np.asarray(data.B_5000, dtype=f4).reshape(NT).tofile(fp)

        np.asarray(data.kap_mean, dtype=f4).reshape(Nbands_out * NT * Np).tofile(fp)
        np.asarray(data.B_band, dtype=f4).reshape(Nbands_out * NT).tofile(fp)

        if data.full_odf == 1:
            if data.nuout is None:
                raise ValueError("full_odf=1 requires nuout")
            np.asarray(data.nuout, dtype=f4).ravel().tofile(fp)
