"""Data-free unit tests for the Q_rad optimizer's search logic.

These never touch the ODF: they inject an analytic `score_fn` with a known optimum
(mirroring `test_build_split_band_index.py`'s data-free style) so the coordinate
descent / Nelder-Mead / greedy-flag / grow logic and the guardrails can be checked fast.
"""

from __future__ import annotations

import unittest

import numpy as np

import qrad_optimize as qo
import tausort as ts


def bowl(target_interior, *, reward_groups=False, flag_target=None):
    """Analytic score_fn: a quadratic bowl in the interior tau edges (min at
    `target_interior`), optionally rewarding more groups and/or a target flag pattern."""
    target = np.asarray(target_interior, float)

    def score(tau, lam, flags, star):
        interior = np.asarray(tau[1:-1], float)
        n = min(len(interior), len(target))
        rms = 1e8 + 1e7 * float(np.sum((interior[:n] - target[:n]) ** 2))
        if reward_groups:
            rms -= 1.5e6 * (len(tau) - 1)
        if flag_target is not None:
            m = min(len(flags), len(flag_target))
            rms += 5e6 * int(np.sum(np.asarray(flags[:m], int) != np.asarray(flag_target[:m], int)))
        return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.1, "n_empty": 0, "n_groups": len(tau) - 1}

    return score


class TestCoordinateDescent(unittest.TestCase):
    def test_cd_converges_to_optimum(self):
        score = bowl([0.0, 2.0])
        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 7.0],
            [3.0, 5.0],
            flags=[True] * 3,
            grow=False,
            opt_lambda=False,
            opt_flags=False,
            method="cd",
            score_fn=score,
        )
        self.assertLess(res["rms"], res["rms0"])
        self.assertAlmostEqual(res["rms"], 1e8, delta=1e5)  # bowl floor
        self.assertEqual(res["tau_edges"], sorted(res["tau_edges"]))

    def test_nm_converges_to_optimum(self):
        score = bowl([0.0, 2.0])
        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 7.0],
            [3.0, 5.0],
            flags=[True] * 3,
            grow=False,
            opt_lambda=False,
            opt_flags=False,
            method="nm",
            score_fn=score,
        )
        self.assertLess(res["rms"], res["rms0"])
        self.assertAlmostEqual(res["rms"], 1e8, delta=5e5)


class TestGuardrails(unittest.TestCase):
    def test_result_edges_monotone_and_min_gap(self):
        score = bowl([0.0, 2.0])
        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 7.0],
            [3.0, 5.0],
            flags=[True] * 3,
            grow=True,
            opt_lambda=False,
            opt_flags=False,
            method="cd",
            score_fn=score,
        )
        te = res["tau_edges"]
        self.assertEqual(te, sorted(te))
        self.assertTrue(qo._min_gap_ok(te, qo.MIN_GAP_TAU))

    def test_infeasible_raises(self):
        # 1 group needs span >= min_gap; 0.1 < 0.5 -> ValueError
        with self.assertRaises(ValueError):
            qo.optimize_qrad([0.0, 0.1], [3.0, 5.0], flags=[True], min_gap_tau=0.5, score_fn=bowl([]))

    def test_wrong_flag_length_raises(self):
        with self.assertRaises(ValueError):
            qo.optimize_qrad([-0.63, 1.0, 7.0], [3.0, 5.0], flags=[True], score_fn=bowl([]))


class TestFlagSearch(unittest.TestCase):
    def test_greedy_flip_finds_target_pattern(self):
        target = [True, False, True, False]
        score = bowl([0.0, 1.0, 2.0, 3.0], flag_target=target)
        # start from the opposite pattern; only the flag block runs (2 lambda cells)
        res = qo.optimize_qrad(
            [-0.63, -0.1, 0.5, 1.5, 7.0],
            [3.0, 3.8, 5.0],
            flags=[False, True, False, True],
            grow=False,
            opt_tau=False,
            opt_lambda=False,
            opt_flags=True,
            method="cd",
            score_fn=score,
        )
        self.assertEqual(res["flags"], target)


class TestGrow(unittest.TestCase):
    def test_grow_adds_groups_when_rewarded(self):
        score = bowl([0.0, 2.0, 4.0, 5.0, 6.0], reward_groups=True)
        res = qo.optimize_qrad(
            [-0.63, 3.5, 7.0],
            [3.0, 5.0],
            flags=[True] * 2,
            grow=True,
            opt_lambda=False,
            opt_flags=False,
            method="cd",
            max_groups=6,
            score_fn=score,
        )
        self.assertGreater(res["n_groups"], 2)
        self.assertIn("grow", [h["tag"] for h in res["history"]])

    def test_grow_declines_when_no_improvement(self):
        # bowl minimized at the current single interior edge; extra groups only add error
        score = bowl([2.0])  # not reward_groups
        res = qo.optimize_qrad(
            [-0.63, 2.0, 7.0],
            [3.0, 5.0],
            flags=[True] * 2,
            grow=True,
            opt_lambda=False,
            opt_flags=False,
            method="cd",
            max_groups=6,
            score_fn=score,
        )
        self.assertEqual(res["n_groups"], 2)
        self.assertNotIn("grow", [h["tag"] for h in res["history"]])


class TestReparameterization(unittest.TestCase):
    def test_roundtrip(self):
        a, b, mg = -0.63, 7.0, 0.15
        interior = [0.35, 1.23, 2.885]
        back = qo._edges_from_u(qo._u_from_edges(interior, a, b, mg), a, b, mg)
        self.assertTrue(np.allclose(interior, back, atol=1e-9))

    def test_random_u_always_feasible(self):
        a, b, mg = -0.63, 7.0, 0.15
        rng = np.random.default_rng(0)
        for _ in range(1000):
            full = [a, *qo._edges_from_u(rng.normal(size=4) * 3, a, b, mg), b]
            self.assertEqual(full, sorted(full))
            self.assertTrue(qo._min_gap_ok(full, mg))


class TestEmptyPenalty(unittest.TestCase):
    def test_empty_band_scores_worse(self):
        ev0, _ = qo.make_evaluator(
            "X", score_fn=lambda t, l, f, s: {"rms": 1e8, "max_abs": 0, "int_q_pct": 0, "n_empty": 0}
        )
        ev1, _ = qo.make_evaluator(
            "X", score_fn=lambda t, l, f, s: {"rms": 1e8, "max_abs": 0, "int_q_pct": 0, "n_empty": 1}
        )
        c0 = ev0([-0.63, 7.0], [3.0, 5.0], [True])[0]
        c1 = ev1([-0.63, 7.0], [3.0, 5.0], [True])[0]
        self.assertGreater(c1, c0)


class TestPerTauGroupLambdaCore(unittest.TestCase):
    def test_equivalence_with_split_flags(self):
        # a flags-equivalent per-tau lambda must reproduce the split-flag grouping exactly
        tau = [-0.63, 0.35, 1.23, 2.89, 7.0]
        lam = [3.0, 3.8, 5.0]
        flags = [True, False, True, True]
        lpt = [lam if f else [lam[0], lam[-1]] for f in flags]
        gt_s, gl_s, s2c, s2s = ts.build_group_specs_split_lambda(tau, lam, flags)
        gt_p, gl_p, offs = ts.build_group_specs_per_tau(tau, lpt)
        self.assertTrue(np.allclose(gt_s, gt_p))
        self.assertTrue(np.allclose(gl_s, gl_p))
        rng = np.random.default_rng(1)
        n = 3000
        tv = 10.0 ** (-rng.uniform(-1, 7, n))
        wl = 10.0 ** (rng.uniform(3, 5, n)) / 1e8
        bi_s = ts.assign_split_lambda(tv, wl, tau, lam, s2c, s2s)
        bi_p = ts.assign_per_tau_lambda(tv, wl, tau, lpt, offs)
        self.assertTrue(np.array_equal(bi_s, bi_p))

    def test_per_group_specs_and_offsets(self):
        tau = [-0.63, 1.0, 7.0]
        lpt = [[3.0, 3.5, 5.0], [3.0, 5.0]]  # group0 split, group1 unsplit
        gt, gl, offs = ts.build_group_specs_per_tau(tau, lpt)
        self.assertEqual(gt.shape[0], 3)  # 2 cells + 1 cell
        self.assertEqual(offs, [0, 2])
        self.assertTrue(np.allclose(gl, [[3.0, 3.5], [3.5, 5.0], [3.0, 5.0]]))


class TestPerGroupLambdaOptimizer(unittest.TestCase):
    def test_each_group_finds_its_own_split(self):
        targets = [3.5, None, 4.2, 3.9]  # group1 should NOT split; others split at these cuts

        def score(tau, lam, flags, star, *, lambda_edges_per_tau=None):
            lpt = lambda_edges_per_tau
            rms = 1e8
            for k, tgt in enumerate(targets):
                lk = lpt[k]
                split = len(lk) >= 3
                if tgt is None:
                    rms += 0.0 if not split else 6e6
                else:
                    rms += 1e7 * (lk[1] - tgt) ** 2 if split else 8e6
            return {
                "rms": rms,
                "max_abs": 2 * rms,
                "int_q_pct": 0.1,
                "n_empty": 0,
                "n_groups": sum(len(x) - 1 for x in lpt),
            }

        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 3.8, 5.0],
            flags=[True] * 4,
            star="X",
            per_group_lambda=True,
            opt_tau=False,
            grow=False,
            method="cd",
            score_fn=score,
            max_evals=3000,
        )
        self.assertTrue(res["per_group_lambda"])
        self.assertLess(res["rms"], res["rms0"])
        got = res["lambda_edges_per_tau"]
        self.assertEqual(len(got[1]), 2)  # group 1 unsplit
        for k in (0, 2, 3):
            self.assertEqual(len(got[k]), 3)  # group k split (one interior cut)
            self.assertAlmostEqual(got[k][1], targets[k], delta=0.05)


if __name__ == "__main__":
    unittest.main()
