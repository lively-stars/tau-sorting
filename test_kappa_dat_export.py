"""Unit tests for the parametrized kappa .dat export from tausort.main.

Covers build_kappa_dat_filename (parameter-encoding filename) and
build_kappa_band_comparison (packing into the C tausort convention:
kap_mean = ln(mixed), leading band axis, log10 T/p axes), plus a
write -> read round-trip through kappa_band_reader.
"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kappa_band_reader import read_kappa_4_band_comparison, write_kappa_4_band_comparison
from tausort import build_kappa_band_comparison, build_kappa_dat_filename


def _synthetic_results(nt, npr, nb, nan_band=None):
    mixed = (np.arange(nt * npr * nb, dtype=np.float64).reshape(nt, npr, nb) + 1.0) * 1e-3
    b_band = np.arange(nt * nb, dtype=np.float64).reshape(nt, nb) + 1.0
    if nan_band is not None:
        mixed[:, :, nan_band] = np.nan  # empty band, as main NaN-masks it
    odf = SimpleNamespace(T=np.linspace(3.5, 4.0, nt), P=np.linspace(-1.0, 5.0, npr))
    return {"kappa_mixed": mixed, "B_band": b_band}, odf


class TestKappaDatExport(unittest.TestCase):
    def test_filename_encodes_params(self):
        # Single lambda cell: full tau edges are spelled out (backward compatible).
        edges = [-0.6347, -0.4, -0.2375, -0.075, 0.15, 0.7, 1.5, 3.8, 7.0]  # 8 tau groups
        fn = build_kappa_dat_filename(24, 3, [3.0, 5.0], tau_edges_per_lambda=[edges])
        self.assertTrue(fn.startswith("kappa_24band_tg8_sp3_tau_"))
        self.assertIn("-0.6347", fn)  # negative + decimals survive
        self.assertIn("_lam_3_5.dat", fn)  # trailing zeros dropped (3.0 -> 3)

    def test_filename_multi_lambda(self):
        # Multiple lambda cells: ragged tau edges -> encode counts, not full edges.
        per_cell = [[-0.63, 0.5, 1.2, 7.0], [-0.63, 2.0, 7.0]]  # 3 and 2 tau groups
        fn = build_kappa_dat_filename(15, 3, [3.0, 4.0, 5.0], tau_edges_per_lambda=per_cell)
        self.assertTrue(fn.startswith("kappa_15band_lm2_tg3-2_sp3_"))
        self.assertIn("_lam_3_4_5.dat", fn)

    def test_filename_split_lambda(self):
        # Split-flag mode: shared tau edges + per-group flags encoded as 1/0.
        tau = [-0.63, 0.15, 1.5, 7.0]  # 3 tau groups
        flags = [True, False, True]  # nBands = 3*(2 + 1 + 2) = 15
        fn = build_kappa_dat_filename(15, 3, [3.0, 3.8, 5.0], tau_bin_edges=tau, split_along_lambda=flags)
        self.assertTrue(fn.startswith("kappa_15band_lm2_sl101_sp3_tau_"))
        self.assertIn("_lam_3_3.8_5.dat", fn)

    def test_pack_shapes_logs_axes(self):
        nt, npr, nb = 5, 4, 6
        results, odf = _synthetic_results(nt, npr, nb)
        kbc = build_kappa_band_comparison(results, odf)  # type: ignore  # lightweight test stubs

        # Leading band axis, C convention.
        self.assertEqual(kbc.kap_mean.shape, (nb, nt, npr))
        self.assertEqual(kbc.B_band.shape, (nb, nt))
        # kap_mean = ln(mixed), reordered [NT, Np, Nbands] -> [Nbands, NT, Np].
        np.testing.assert_allclose(kbc.kap_mean, np.log(results["kappa_mixed"]).transpose(2, 0, 1))
        np.testing.assert_allclose(kbc.B_band, np.log(results["B_band"]).T)
        # Axes are log10(T)/log10(p) == odf.T/odf.P as stored.
        np.testing.assert_allclose(kbc.tab_T, odf.T)
        np.testing.assert_allclose(kbc.tab_p, odf.P)
        # Header flags for our (band, no tau5000, no full-odf) case.
        self.assertEqual(
            (kbc.tau5000bin, kbc.full_odf, kbc.scatter_on, kbc.back_heating, kbc.Nbands_out),
            (0, 0, 0, 0, nb),
        )
        self.assertIsNone(kbc.kap_5000)
        self.assertIsNone(kbc.nuout)

    def test_empty_band_becomes_nan(self):
        results, odf = _synthetic_results(4, 3, 5, nan_band=2)
        kbc = build_kappa_band_comparison(results, odf)  # type: ignore  # lightweight test stubs
        self.assertTrue(np.isnan(kbc.kap_mean[2]).all())
        # Other bands stay finite.
        self.assertTrue(np.isfinite(kbc.kap_mean[0]).all())

    def test_write_read_roundtrip(self):
        results, odf = _synthetic_results(6, 5, 4, nan_band=1)
        kbc = build_kappa_band_comparison(results, odf)  # type: ignore  # lightweight test stubs
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "kappa_test.dat"
            write_kappa_4_band_comparison(path, kbc)
            back = read_kappa_4_band_comparison(path)  # strict=True: must have no trailing bytes

        self.assertEqual((back.NT, back.Np, back.Nbands_out), (6, 5, 4))
        # kap_mean / B_band are float32 on disk; compare with NaN-aware tolerance.
        np.testing.assert_allclose(back.kap_mean, kbc.kap_mean.astype(np.float32), rtol=1e-5, equal_nan=True)
        np.testing.assert_allclose(back.B_band, kbc.B_band.astype(np.float32), rtol=1e-5, equal_nan=True)
        # f8 axes round-trip exactly.
        np.testing.assert_array_equal(back.tab_T, kbc.tab_T)
        np.testing.assert_array_equal(back.tab_p, kbc.tab_p)


if __name__ == "__main__":
    unittest.main()
