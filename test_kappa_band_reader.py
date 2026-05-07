from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from kappa_band_reader import read_kappa_4_band_comparison


def _write_layout_file(
    path: Path,
    *,
    tau5000bin: int,
    NT: int,
    Np: int,
    Nbands_out: int,
    pp_axis: int,
    full_odf: int,
    scatter_on: int,
    back_heating: int,
    tab_T: np.ndarray,
    tab_p: np.ndarray,
    kap_5000: np.ndarray | None,
    B_5000: np.ndarray | None,
    kap_mean: np.ndarray,
    B_band: np.ndarray,
    nuout: np.ndarray | None,
) -> None:
    i4 = np.dtype("<i4")
    f8 = np.dtype("<f8")
    f4 = np.dtype("<f4")

    header = np.array(
        [tau5000bin, NT, Np, Nbands_out, pp_axis, full_odf, scatter_on, back_heating],
        dtype=i4,
    )

    with path.open("wb") as fp:
        header.tofile(fp)
        np.asarray(tab_T, dtype=f8).tofile(fp)
        np.asarray(tab_p, dtype=f8).tofile(fp)
        if tau5000bin == 1:
            if kap_5000 is None or B_5000 is None:
                raise ValueError("tau5000bin=1 requires kap_5000 and B_5000")
            np.asarray(kap_5000, dtype=f4).reshape(NT * Np).tofile(fp)
            np.asarray(B_5000, dtype=f4).reshape(NT).tofile(fp)
        np.asarray(kap_mean, dtype=f4).reshape(Nbands_out * NT * Np).tofile(fp)
        np.asarray(B_band, dtype=f4).reshape(Nbands_out * NT).tofile(fp)
        if full_odf == 1:
            if nuout is None:
                raise ValueError("full_odf=1 requires nuout")
            np.asarray(nuout, dtype=f4).tofile(fp)


class TestKappaBandReader(unittest.TestCase):
    def test_round_trip_with_all_optional_sections(self) -> None:
        NT, Np, Nbands_out = 3, 2, 4

        tab_T = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        tab_p = np.array([10.0, 20.0], dtype=np.float64)
        kap_5000 = np.arange(NT * Np, dtype=np.float32).reshape(NT, Np) + 0.5
        B_5000 = np.array([5.0, 6.0, 7.0], dtype=np.float32)
        kap_mean = np.arange(Nbands_out * NT * Np, dtype=np.float32).reshape(
            Nbands_out, NT, Np
        )
        B_band = np.arange(Nbands_out * NT, dtype=np.float32).reshape(Nbands_out, NT)
        nuout = np.array([100.0, 200.0, 300.0, 400.0], dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "synthetic.dat"
            _write_layout_file(
                out,
                tau5000bin=1,
                NT=NT,
                Np=Np,
                Nbands_out=Nbands_out,
                pp_axis=0,
                full_odf=1,
                scatter_on=1,
                back_heating=0,
                tab_T=tab_T,
                tab_p=tab_p,
                kap_5000=kap_5000,
                B_5000=B_5000,
                kap_mean=kap_mean,
                B_band=B_band,
                nuout=nuout,
            )

            data = read_kappa_4_band_comparison(out)

        self.assertEqual(data.tau5000bin, 1)
        self.assertEqual(data.NT, NT)
        self.assertEqual(data.Np, Np)
        self.assertEqual(data.Nbands_out, Nbands_out)
        self.assertEqual(data.pp_axis, 0)
        self.assertEqual(data.full_odf, 1)
        self.assertEqual(data.scatter_on, 1)
        self.assertEqual(data.back_heating, 0)
        np.testing.assert_allclose(data.tab_T, tab_T)
        np.testing.assert_allclose(data.tab_p, tab_p)
        np.testing.assert_allclose(data.kap_5000, kap_5000)
        np.testing.assert_allclose(data.B_5000, B_5000)
        np.testing.assert_allclose(data.kap_mean, kap_mean)
        np.testing.assert_allclose(data.B_band, B_band)
        np.testing.assert_allclose(data.nuout, nuout)

    @unittest.skipUnless(
        Path("kappa_4_band_comparison.dat").exists(),
        "kappa_4_band_comparison.dat not available",
    )
    def test_repository_file_smoke(self) -> None:
        data = read_kappa_4_band_comparison("kappa_4_band_comparison.dat")

        self.assertGreater(data.NT, 0)
        self.assertGreater(data.Np, 0)
        self.assertGreater(data.Nbands_out, 0)
        self.assertEqual(data.tab_T.shape, (data.NT,))
        self.assertEqual(data.tab_p.shape, (data.Np,))
        self.assertEqual(data.kap_mean.shape, (data.Nbands_out, data.NT, data.Np))
        self.assertEqual(data.B_band.shape, (data.Nbands_out, data.NT))
        self.assertTrue(np.isfinite(data.tab_T).all())
        self.assertTrue(np.isfinite(data.tab_p).all())
        self.assertTrue(np.isfinite(data.kap_mean).all())
        self.assertTrue(np.isfinite(data.B_band).all())
        if data.tau5000bin == 1:
            self.assertIsNotNone(data.kap_5000)
            self.assertIsNotNone(data.B_5000)
            self.assertEqual(data.kap_5000.shape, (data.NT, data.Np))
            self.assertEqual(data.B_5000.shape, (data.NT,))
        else:
            self.assertIsNone(data.kap_5000)
            self.assertIsNone(data.B_5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
