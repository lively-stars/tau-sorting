"""Data-gated regression tests: the deterministic fine-grid scan must reach the
Q_rad winner basin (~4.87e7) on seeds that historically missed it.

Seeds 0 and 7 are documented losers under the old optimizer (scattered 7.9e7-1.2e8;
only seed 3 of 25 ever hit 4.87e7), plus seed 5 as a third non-3 scatter seed. Each
test runs `optimize_qrad` with `scan_allocation=True, deterministic_fine=True` on a
reduced budget (`initial_tau_scan=0`, `grow=False`, `max_evals` just above the ~540
raw grid evals + polish) and asserts the winner basin (rms <= 6.5e7, cleanly below
the old-loser floor of >= 7.9e7).

Slow (~10 min/seed at ~1.3 s/eval): kept in this separate file so the data-free
`test_qrad_optimize` suite stays fast. Gated on `qrad_core.check_data_files()` (which
honors TAUSORT_DATA_DIR), so they run in a checkout with the gitignored ODF data and
skip cleanly elsewhere.
"""

from __future__ import annotations

import unittest

import qrad_core as qc
import qrad_optimize as qo


def _data_missing() -> list[str]:
    return qc.check_data_files()


class TestDeterministicFineSeeds(unittest.TestCase):
    """Deterministic fine-grid scan reaches the winner basin on historic miss seeds."""

    _SEEDS = (0, 7, 5)
    _RMS_BASIN = 6.5e7  # winner ~4.87e7; old losers scattered >= 7.9e7

    @staticmethod
    def _run(seed: int) -> dict:
        return qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            grow=False,
            initial_tau_bins=3,
            initial_tau_scan=0,
            staged_lambda_scan=8,  # > 0 arms the scan_allocation + deterministic-fine path
            scan_allocation=True,
            deterministic_fine=True,
            max_groups=5,
            seed=seed,
            max_evals=900,  # ~540 grid + staged-tau/polish headroom; no beam/topo with grow=False
        )

    def _check_seed(self, seed: int) -> None:
        missing = _data_missing()
        if missing:
            self.skipTest(f"ODF/continuum data absent ({', '.join(missing)})")
        res = self._run(seed)
        self.assertLessEqual(res["rms"], self._RMS_BASIN, f"seed {seed}: rms {res['rms']:.4e} missed the basin")
        self.assertEqual(res["n_leaves"], 5)
        self.assertEqual(res["n_empty"], 0)
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))

    def test_seed_0_reaches_winner_basin(self):
        self._check_seed(0)

    def test_seed_7_reaches_winner_basin(self):
        self._check_seed(7)

    def test_seed_5_reaches_winner_basin(self):
        self._check_seed(5)


if __name__ == "__main__":
    unittest.main()
