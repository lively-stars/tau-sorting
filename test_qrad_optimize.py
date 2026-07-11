"""Data-free unit tests for the Q_rad optimizer's search logic.

These never touch the ODF: they inject an analytic `score_fn` with a known optimum
(mirroring `test_build_split_band_index.py`'s data-free style) so the coordinate
descent / Nelder-Mead / greedy-flag / grow logic and the guardrails can be checked fast.
"""

from __future__ import annotations

import unittest

import numpy as np

import qrad_core as qc
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
            model="X",
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

    def test_warm_start_is_honored(self):
        # A re-run must resume from the passed per-group cuts, not reset to the shared box.
        seen = {}

        def score(tau, lam, flags, model, *, lambda_edges_per_tau=None):
            seen.setdefault("lpt0", [list(x) for x in lambda_edges_per_tau])
            return {
                "rms": 1e8,
                "max_abs": 2e8,
                "int_q_pct": 0.0,
                "n_empty": 0,
                "n_groups": sum(len(x) - 1 for x in lambda_edges_per_tau),
            }

        warm = [[3.0, 3.55, 5.0], [3.0, 5.0], [3.0, 4.4, 5.0], [3.0, 3.7, 5.0]]
        qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 3.8, 5.0],
            flags=[True] * 4,
            per_group_lambda=True,
            lambda_edges_per_tau=warm,
            opt_tau=False,
            opt_lambda=False,
            grow=False,
            score_fn=score,
            max_evals=50,
        )
        self.assertEqual(seen["lpt0"], warm)  # started from the warm-start cuts, not [3,3.8,5]×4


class TestPerTauLambdaCLI(unittest.TestCase):
    def test_parse_lambda_per_tau(self):
        got = ts.parse_lambda_per_tau(["3,3.82,5", "3 5", "3, 4.2, 5"])
        self.assertEqual(got, [[3.0, 3.82, 5.0], [3.0, 5.0], [3.0, 4.2, 5.0]])

    def test_parse_lambda_per_tau_rejects_bad(self):
        with self.assertRaises(Exception):
            ts.parse_lambda_per_tau(["5,3"])  # not increasing
        with self.assertRaises(Exception):
            ts.parse_lambda_per_tau(["3"])  # < 2 edges

    def test_filename_encodes_per_group_cuts(self):
        name = ts.build_kappa_dat_filename(
            nbands=21,
            n_splits=3,
            lambda_bin_edges=[3.0, 5.0],
            tau_bin_edges=[-0.63, 0.3488, 1.2275, 2.885, 7.0],
            lambda_edges_per_tau=[[3, 3.82, 5], [3, 3.65, 5], [3, 5], [3, 3.8, 5]],
        )
        self.assertIn("_pt_", name)
        self.assertIn("cuts_3.82-3.65-x-3.8", name)
        self.assertTrue(name.endswith(".dat"))


class TestConvertContinuum(unittest.TestCase):
    def test_dat_to_npy_ordering(self):
        import os
        import tempfile

        from typer.testing import CliRunner

        nb, nt, npr = 2, 3, 2  # nbins, nt, n_pressure
        data = np.arange(nb * nt * npr, dtype=float)  # .dat is (lambda, T, P) C-order: 0..11
        d = tempfile.mkdtemp()
        dat, out = os.path.join(d, "cont.dat"), os.path.join(d, "cont.npy")
        np.savetxt(dat, data)
        res = CliRunner().invoke(
            ts.app,
            ["convert-continuum", dat, out, "--nt", str(nt), "--np", str(npr), "--nbins", str(nb)],
        )
        self.assertEqual(res.exit_code, 0, res.output)
        got = np.load(out)
        self.assertEqual(got.shape, (nt, npr, nb))  # main's (nt, np, nbins) layout
        ref = data.reshape(nb, nt, npr)  # the .dat's native (lambda, T, P) view
        for t in range(nt):
            for p in range(npr):
                for lam in range(nb):
                    self.assertEqual(got[t, p, lam], ref[lam, t, p])


class TestConvertOdf(unittest.TestCase):
    def test_nc_to_npy(self):
        import os
        import tempfile

        try:
            from netCDF4 import Dataset
        except Exception:
            self.skipTest("netCDF4 not available")
        from typer.testing import CliRunner

        nt, npr, nb, nsb, numfp = 2, 2, 2, 2, 3
        d = tempfile.mkdtemp()
        nc, out = os.path.join(d, "odf.nc"), os.path.join(d, "odf.npy")
        with Dataset(nc, "w") as ds:
            for name, size in (("np", npr), ("nt", nt), ("nbins", nb), ("nsubbins", nsb), ("numfp", numfp)):
                ds.createDimension(name, size)
            ds.createVariable("ODF", "i2", ("nt", "np", "nbins", "nsubbins"))[:] = 1000  # 10**(1000/1000)=10
            ds.createVariable("FreqG", "f8", ("numfp",))[:] = [1.0, 2.0, 3.0]
            ds.createVariable("P", "f8", ("np",))[:] = [0.5, 1.5]
            ds.createVariable("T", "f8", ("nt",))[:] = [3.2, 4.0]
            ds.createVariable("subbin", "f8", ("nbins", "nsubbins"))[:] = 0.5
            ds.vturb = 2.0

        res = CliRunner().invoke(ts.app, ["convert-odf", nc, out])
        self.assertEqual(res.exit_code, 0, res.output)
        a = np.load(out, allow_pickle=True)
        self.assertEqual(int(a["nt"][0]), nt)
        self.assertEqual(int(a["nbins"][0]), nb)
        self.assertTrue(np.allclose(a["T"][0], [3.2, 4.0]))
        self.assertTrue(np.allclose(a["ODF"][0], 10.0))  # 10**(ODF/1000)


class TestStoppingAndWindow(unittest.TestCase):
    """Stopping conditions + scoring window, via an injected analytic score_fn (no ODF)."""

    @staticmethod
    def _flat(rms=5e7):
        def score(tau, lam, flags, model, *, lambda_edges_per_tau=None, window=None):
            return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": len(tau) - 1}

        return score

    def test_target_rms_stops_early(self):
        # score always returns rms below the target -> should stop with stop_reason='target_rms'
        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 5.0],
            flags=[True] * 4,
            target_rms=6e7,
            opt_lambda=False,
            opt_flags=False,
            grow=False,
            score_fn=self._flat(5e7),
            max_evals=1000,
            max_seconds=100,
        )
        self.assertEqual(res["stop_reason"], "target_rms")
        self.assertLess(res["n_evals"], 25)

    def test_plateau_stops(self):
        # constant rms -> no improvement -> plateau fires after plateau_evals
        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 5.0],
            flags=[True] * 4,
            plateau_evals=5,
            opt_lambda=False,
            opt_flags=False,
            grow=False,
            score_fn=self._flat(5e7),
            max_evals=1000,
            max_seconds=100,
        )
        self.assertEqual(res["stop_reason"], "plateau")

    def test_max_evals_stops(self):
        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 5.0],
            flags=[True] * 4,
            max_evals=8,
            grow=False,
            score_fn=self._flat(5e7),
            max_seconds=100,
        )
        self.assertEqual(res["stop_reason"], "max_evals")
        # a few un-budgeted checkpoint/final evals record the result, so allow a small overshoot
        self.assertLessEqual(res["n_evals"], 8 + 5)

    def test_window_forwarded_only_when_set(self):
        seen = []

        def score(tau, lam, flags, model, *, lambda_edges_per_tau=None, window="MISSING"):
            seen.append(window)
            return {"rms": 5e7, "max_abs": 1e8, "int_q_pct": 0.0, "n_empty": 0, "n_groups": len(tau) - 1}

        qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 5.0],
            flags=[True] * 4,
            window=(-2.0, 2.0),
            opt_tau=False,
            opt_lambda=False,
            opt_flags=False,
            grow=False,
            score_fn=score,
            max_evals=20,
        )
        self.assertTrue(all(w == (-2.0, 2.0) for w in seen))  # window passed through every call

    def test_window_absent_keeps_positional_signature(self):
        # a 4-positional score_fn (no window kwarg) must still work when window is None
        def score(tau, lam, flags, model):
            return {"rms": 5e7, "max_abs": 1e8, "int_q_pct": 0.0, "n_empty": 0, "n_groups": len(tau) - 1}

        res = qo.optimize_qrad(
            [-0.63, 0.35, 1.23, 2.89, 7.0],
            [3.0, 5.0],
            flags=[True] * 4,
            grow=False,
            score_fn=score,
            max_evals=15,
        )
        self.assertIn("stop_reason", res)


def _n_leaves(tree):
    return sum(1 for _ in qo._leaf_rects(tree["root"], (0.0, 1.0, 0.0, 1.0)))


def _tree_dev_score(tau_target=1.5, lam_target=3.8):
    """Analytic tree score_fn: rms grows with each internal cut's squared distance from its
    axis target, so coordinate descent should drive the cuts to (tau_target, lam_target)."""

    def score(tau, lam, flags, model, *, lambda_edges_per_tau=None, binning_tree=None, window=None):
        rms = 1.0e8

        def walk(node):
            nonlocal rms
            if node.get("leaf") or "axis" not in node:
                return
            tgt = tau_target if node["axis"] == "tau" else lam_target
            rms += 1.0e7 * (float(node["at"]) - tgt) ** 2
            walk(node["lo"])
            walk(node["hi"])

        walk(binning_tree["root"])
        return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": _n_leaves(binning_tree)}

    return score


def _tree_leafcount_score():
    """Analytic tree score_fn: rms = 1e8 / n_leaves, so grow keeps splitting until the cap."""

    def score(tau, lam, flags, model, *, lambda_edges_per_tau=None, binning_tree=None, window=None):
        n = _n_leaves(binning_tree)
        return {"rms": 1.0e8 / n, "max_abs": 1.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

    return score


class TestTreeOptimizer(unittest.TestCase):
    def test_tree_from_lpt_matches_per_tau(self):
        # the warm-start tree reproduces the exact rectangles (and DFS order) of the per-tau mode.
        tau = [-0.63, 0.35, 1.23, 7.0]
        lpt = [[3.0, 3.8, 5.0], [3.0, 5.0], [3.0, 4.2, 5.0]]
        tree = qo.tree_from_lpt(tau, lpt)
        rects = list(qo._leaf_rects(tree["root"], qo._root_rect(tree)))
        gte, gle, _off = ts.build_group_specs_per_tau(tau, lpt)
        self.assertEqual(len(rects), gte.shape[0])
        for i, (tlo, thi, llo, lhi) in enumerate(rects):
            self.assertAlmostEqual(tlo, gte[i][0])
            self.assertAlmostEqual(thi, gte[i][1])
            self.assertAlmostEqual(llo, gle[i][0])
            self.assertAlmostEqual(lhi, gle[i][1])

    def test_tree_position_refinement(self):
        res = qo.optimize_qrad(
            [-0.63, 0.5, 7.0],
            [3.0, 3.4, 5.0],
            flags=[True, True],
            tree=True,
            grow=False,
            score_fn=_tree_dev_score(),
            max_evals=5000,
        )
        self.assertTrue(res["tree"])
        self.assertLess(res["rms"], res["rms0"])
        devs = []

        def walk(node):
            if node.get("leaf") or "axis" not in node:
                return
            tgt = 1.5 if node["axis"] == "tau" else 3.8
            devs.append(abs(float(node["at"]) - tgt))
            walk(node["lo"])
            walk(node["hi"])

        walk(res["binning_tree"]["root"])
        self.assertTrue(devs and all(d < 0.12 for d in devs), devs)  # driven to the targets

    def test_tree_grow_respects_max(self):
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            tree=True,
            grow=True,
            max_groups=5,
            score_fn=_tree_leafcount_score(),
            max_evals=5000,
        )
        self.assertTrue(res["tree"])
        self.assertLessEqual(res["n_leaves"], 5)  # never exceeds the cap
        self.assertGreater(res["n_leaves"], 1)  # it grew
        self.assertLess(res["rms"], res["rms0"])

    def test_tree_result_is_feasible(self):
        res = qo.optimize_qrad(
            [-0.63, 0.5, 7.0],
            [3.0, 3.4, 5.0],
            flags=[True, True],
            tree=True,
            grow=True,
            max_groups=6,
            score_fn=_tree_dev_score(),
            max_evals=5000,
        )
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))


class TestTreeEquivalence(unittest.TestCase):
    """Gating tests pinning that the general guillotine-tree path is equivalent to the
    per-tau-group-lambda path it generalizes — at the membership level (data-free) and at the
    full .dat byte level (ODF-dependent). These are the P1a deletion gate."""

    # data-free: tree_from_lpt's assign_tree must match assign_per_tau_lambda exactly
    LPT_CASES = [
        # tau edges, per-tau-group lambda edges (mixed split/unsplit)
        ([-0.63, 0.35, 1.23, 7.0], [[3.0, 3.8, 5.0], [3.0, 5.0], [3.0, 4.2, 5.0]]),
        ([-0.63, 1.0, 7.0], [[3.0, 3.5, 5.0], [3.0, 5.0]]),  # split, unsplit
        (
            [-0.63, 0.3488, 1.2275, 2.885, 7.0],
            [[3.0, 3.82, 5.0], [3.0, 3.65, 5.0], [3.0, 5.0], [3.0, 3.8, 5.0]],
        ),
    ]

    def test_assign_tree_matches_per_tau_membership(self):
        # for several lpt binnings (incl. mixed split/unsplit tau groups) and thousands of
        # random (tau_rosseland, wavelength) points, tree membership == per-tau membership.
        rng = np.random.default_rng(7)
        n = 4000
        for tau_edges, lpt in self.LPT_CASES:
            with self.subTest(tau=tau_edges, lpt=lpt):
                tau = [float(e) for e in tau_edges]
                tree = qo.tree_from_lpt(tau, lpt)
                _gt, _gl, offs = ts.build_group_specs_per_tau(tau, lpt)
                tv = 10.0 ** (-rng.uniform(-1, 7, n))
                wl = 10.0 ** (rng.uniform(3, 5, n)) / 1e8
                bi_tree = ts.assign_tree(tv, wl, tree["root"], tree["window_tau"], tree["window_lam"])
                bi_lpt = ts.assign_per_tau_lambda(tv, wl, tau, lpt, offs)
                self.assertTrue(
                    np.array_equal(bi_tree, bi_lpt),
                    f"tree vs per-tau membership differ for tau={tau}, lpt={lpt}",
                )
                # sanity: most points land somewhere (not all rejected)
                self.assertGreater((bi_tree >= 0).sum(), n // 4)

    @staticmethod
    def _data_ready():
        from pathlib import Path

        repo = Path(__file__).resolve().parent
        odf_ok = (repo / "ODF_format.npy").exists() or (repo / "ODF_nc_format.nc").exists()
        return odf_ok and (repo / "continuumabs.dat").exists() and (repo / "models" / "G2_1D.dat").exists()

    def test_per_tau_lambda_and_tree_produce_identical_kappa_dat(self):
        # ODF-dependent hard gate: score_binning via lambda_edges_per_tau and via
        # binning_tree=tree_from_lpt(...) must give identical rms/members, and the serialized
        # kappa .dat (build_kappa_band_comparison output) must be byte-identical.
        if not self._data_ready():
            self.skipTest("ODF / continuum / models/G2_1D.dat not present")
        import tempfile
        from pathlib import Path

        from kappa_band_reader import read_kappa_4_band_comparison

        tau = [-0.63, 0.3488, 1.2275, 2.885, 7.0]
        lpt = [[3.0, 3.82, 5.0], [3.0, 3.65, 5.0], [3.0, 5.0], [3.0, 3.8, 5.0]]
        tree = qo.tree_from_lpt(tau, lpt)
        model = "G2_1D.dat"

        res_lpt = qc.score_binning(tau, [3.0, 5.0], None, model=model, lambda_edges_per_tau=lpt)
        res_tree = qc.score_binning(tau, [3.0, 5.0], None, model=model, binning_tree=tree)
        self.assertEqual(res_lpt["rms"], res_tree["rms"])
        self.assertEqual(res_lpt["n_bands"], res_tree["n_bands"])
        self.assertTrue(np.array_equal(res_lpt["members"], res_tree["members"]))
        self.assertTrue(np.array_equal(res_lpt["band_index"], res_tree["band_index"]))

        with tempfile.TemporaryDirectory() as td:
            p_lpt = Path(td) / "lpt.dat"
            p_tree = Path(td) / "tree.dat"
            qc.save_kappa_dat(tau, [3.0, 5.0], None, model=model, lambda_edges_per_tau=lpt, path=p_lpt)
            qc.save_kappa_dat(tau, [3.0, 5.0], None, model=model, binning_tree=tree, path=p_tree)
            # build_kappa_band_comparison output is byte-identical (same header, axes, data)
            self.assertEqual(p_lpt.read_bytes(), p_tree.read_bytes())
            back_lpt = read_kappa_4_band_comparison(p_lpt)
            back_tree = read_kappa_4_band_comparison(p_tree)
            self.assertTrue(np.array_equal(back_lpt.kap_mean, back_tree.kap_mean))
            self.assertTrue(np.array_equal(back_lpt.B_band, back_tree.B_band))


def _tree_bimodal_score():
    """Analytic tree score_fn with a shallow local min (lam cut at 1.0) and a deeper global
    min (lam cut at 3.0), separated by a barrier at p=2.5.

    Greedy grow only ever tries *midpoint* splits: from p=2.0 it descends to the shallow min
    at 1.0, and coordinate descent can't cross the barrier -> stuck. Beam tries the fractions
    0.35/0.5/0.65, so p=2.6 lands in the deep basin and the final polish drives it to 3.0.
    Tau splits and >=3 leaves are penalized, so the global optimum is exactly a 2-leaf lam
    split at 3.0 — reachable by beam, not by greedy."""
    BASE = 1e8

    def score(tau, lam, flags, model, *, binning_tree=None, window=None):
        t = binning_tree
        n = _n_leaves(t)
        has_tau = False
        lam_cuts = []

        def walk(node):
            nonlocal has_tau
            if node.get("leaf") or "axis" not in node:
                return
            if node["axis"] == "tau":
                has_tau = True
            else:
                lam_cuts.append(float(node["at"]))
            walk(node["lo"])
            walk(node["hi"])

        walk(t["root"])
        if has_tau:
            rms = BASE + 40.0 + 5.0 * (n - 1)
        elif n == 1:
            rms = BASE + 50.0
        elif n == 2:
            p = lam_cuts[0]
            rms = BASE + (1.0 + (p - 1.0) ** 2 if p <= 2.5 else (p - 3.0) ** 2)
        else:  # lam-only with >=3 leaves
            rms = BASE + 5.0 + 2.0 * n
        return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

    return score


class TestBeamSearch(unittest.TestCase):
    def test_beam_finds_global_where_greedy_is_stuck(self):
        # single-band seed (one leaf) over tau[0,4] x lam[0,4]; optimum = lam split at 3.0.
        score = _tree_bimodal_score()
        greedy = qo.optimize_qrad(
            [0.0, 4.0],
            [0.0, 4.0],
            flags=[True],
            tree=True,
            grow=True,
            method="cd",
            max_groups=4,
            score_fn=score,
            max_evals=5000,
        )
        beam = qo.optimize_qrad(
            [0.0, 4.0],
            [0.0, 4.0],
            flags=[True],
            tree=True,
            grow=True,
            method="beam",
            max_groups=4,
            score_fn=score,
            max_evals=5000,
        )
        self.assertTrue(beam["tree"])
        BASE = 1e8
        # beam reached the DEEP basin (rms near the global floor); greedy is stuck in the shallow one
        self.assertLess(beam["rms"], BASE + 0.5)
        self.assertGreater(greedy["rms"], BASE + 0.5)
        self.assertLess(beam["rms"], greedy["rms"] - 0.5)

        def lam_cuts(t):
            out = []

            def walk(node):
                if node.get("leaf") or "axis" not in node:
                    return
                if node["axis"] == "lam":
                    out.append(float(node["at"]))
                walk(node["lo"])
                walk(node["hi"])

            walk(t["root"])
            return out

        # basin membership: beam's cut is right of the barrier (deep), greedy's is left (shallow)
        bcuts, gcuts = lam_cuts(beam["binning_tree"]), lam_cuts(greedy["binning_tree"])
        self.assertTrue(bcuts and all(p > 2.5 for p in bcuts), bcuts)
        self.assertTrue(gcuts and all(p < 2.5 for p in gcuts), gcuts)

    def test_beam_respects_leaf_cap(self):
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            tree=True,
            grow=True,
            method="beam",
            max_groups=5,
            score_fn=_tree_leafcount_score(),
            max_evals=5000,
        )
        self.assertTrue(res["tree"])
        self.assertLessEqual(res["n_leaves"], 5)
        self.assertGreater(res["n_leaves"], 1)
        self.assertLess(res["rms"], res["rms0"])

    def test_beam_result_is_feasible(self):
        res = qo.optimize_qrad(
            [-0.63, 0.5, 7.0],
            [3.0, 3.4, 5.0],
            flags=[True, True],
            tree=True,
            grow=True,
            method="beam",
            max_groups=6,
            score_fn=_tree_dev_score(),
            max_evals=5000,
        )
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))

    def test_beam_warm_start_refines(self):
        # a re-run from a passed tree must keep refining it (not reset), and still be feasible.
        first = qo.optimize_qrad(
            [-0.63, 0.5, 7.0],
            [3.0, 3.4, 5.0],
            flags=[True, True],
            tree=True,
            grow=True,
            method="beam",
            max_groups=5,
            score_fn=_tree_dev_score(),
            max_evals=5000,
        )
        again = qo.optimize_qrad(
            [-0.63, 0.5, 7.0],
            [3.0, 3.4, 5.0],
            flags=[True, True],
            tree=True,
            grow=True,
            method="beam",
            max_groups=5,
            score_fn=_tree_dev_score(),
            max_evals=5000,
            binning_tree=first["binning_tree"],
        )
        self.assertTrue(qo._tree_feasible(again["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
        self.assertLessEqual(again["rms"], first["rms"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
