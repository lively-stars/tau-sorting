from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from plot_kap_mean_grid import plot_kap_mean_grid


class TestPlotKapMeanGrid(unittest.TestCase):
    @unittest.skipUnless(
        Path("kappa_4_band_comparison.dat").exists(),
        "kappa_4_band_comparison.dat not available",
    )
    def test_plot_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "kap_mean_grid_4x3.png"
            written = plot_kap_mean_grid(
                "kappa_4_band_comparison.dat",
                output,
                comparison_path=None,
            )

            self.assertEqual(written, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    @unittest.skipUnless(
        Path("kappa_4_band_comparison.dat").exists()
        and Path("tau_bin_opacities.npy").exists(),
        "comparison inputs not available",
    )
    def test_plot_with_comparison_overlay_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "kap_mean_grid_4x3_compare.png"
            written = plot_kap_mean_grid(
                "kappa_4_band_comparison.dat",
                output,
                comparison_path="tau_bin_opacities.npy",
            )

            self.assertEqual(written, output)
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
