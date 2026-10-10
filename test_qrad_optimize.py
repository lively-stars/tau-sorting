"""Data-free unit tests for the Q_rad optimizer's search logic.

These never touch the ODF: they inject an analytic `score_fn` with a known optimum
(mirroring `test_build_split_band_index.py`'s data-free style) so the guillotine-tree
grow/polish logic and the guardrails can be checked fast. The optimizer is tree-only, so
every score_fn is invoked as ``score(None, None, None, model, binning_tree=tree, **kw)``.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import qrad_core as qc
import qrad_optimize as qo
import tausort as ts


def bowl(target_interior, *, reward_groups=False):
    """Analytic tree-aware score_fn: a quadratic bowl in the tree's interior TAU cut positions
    (min when each interior tau cut equals the corresponding target), optionally rewarding more
    leaves. The optimizer is tree-only, so this reads the cuts from `binning_tree` (the positional
    tau/lam/flags are None); it ignores lambda cuts and the scoring window."""
    target = np.asarray(target_interior, float)

    def score(tau, lam, flags, model, *, binning_tree=None, window=None):
        t = binning_tree
        tau_cuts = []

        def walk(node):
            if node.get("leaf") or "axis" not in node:
                return
            if node["axis"] == "tau":
                tau_cuts.append(float(node["at"]))
            walk(node["lo"])
            walk(node["hi"])

        walk(t["root"])
        n_leaves = sum(1 for _ in qo._leaf_rects(t["root"], qo._root_rect(t)))
        interior = np.asarray(sorted(tau_cuts), float)
        n = min(len(interior), len(target))
        rms = 1e8 + 1e7 * float(np.sum((interior[:n] - target[:n]) ** 2))
        if reward_groups:
            rms -= 1.5e6 * n_leaves
        return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.1, "n_empty": 0, "n_groups": n_leaves}

    return score


class TestGuardrails(unittest.TestCase):
    def test_infeasible_raises(self):
        # 1 group needs span >= min_gap; 0.1 < 0.5 -> ValueError
        with self.assertRaises(ValueError):
            qo.optimize_qrad([0.0, 0.1], [3.0, 5.0], flags=[True], min_gap_tau=0.5, score_fn=bowl([]))

    def test_wrong_flag_length_raises(self):
        with self.assertRaises(ValueError):
            qo.optimize_qrad([-0.63, 1.0, 7.0], [3.0, 5.0], flags=[True], score_fn=bowl([]))


class TestEmptyPenalty(unittest.TestCase):
    def test_empty_band_scores_worse(self):
        tree = qo.tree_from_lpt([-0.63, 7.0], [[3.0, 5.0]])
        ev0, _ = qo.make_evaluator(
            "X",
            score_fn=lambda t, l, f, s, *, binning_tree=None: {"rms": 1e8, "max_abs": 0, "int_q_pct": 0, "n_empty": 0},
        )
        ev1, _ = qo.make_evaluator(
            "X",
            score_fn=lambda t, l, f, s, *, binning_tree=None: {"rms": 1e8, "max_abs": 0, "int_q_pct": 0, "n_empty": 1},
        )
        c0 = ev0(binning_tree=tree)[0]
        c1 = ev1(binning_tree=tree)[0]
        self.assertGreater(c1, c0)


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
        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

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

        def score(tau, lam, flags, model, *, binning_tree=None, window="MISSING"):
            seen.append(window)
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            return {"rms": 5e7, "max_abs": 1e8, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

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
        # when window is None, score is NOT passed a window kwarg (only the now-mandatory binning_tree=)
        def score(tau, lam, flags, model, *, binning_tree=None):
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            return {"rms": 5e7, "max_abs": 1e8, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

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
    def test_tree_from_lpt_matches_descriptor(self):
        # tree_from_lpt's leaf rectangles (DFS order) must match build_group_specs_tree's rows.
        tau = [-0.63, 0.35, 1.23, 7.0]
        lpt = [[3.0, 3.8, 5.0], [3.0, 5.0], [3.0, 4.2, 5.0]]
        tree = qo.tree_from_lpt(tau, lpt)
        rects = list(qo._leaf_rects(tree["root"], qo._root_rect(tree)))
        gte, gle = ts.build_group_specs_tree(tree["root"], tree["window_tau"], tree["window_lam"])
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
    """Gating tests for the guillotine tree (now the sole grouping IR): data-free membership
    self-consistency (every assigned sub-bin lies inside its leaf rectangle), and an ODF-dependent
    byte-level check that the per-tau-lambda and explicit-tree qrad_core entry points agree."""

    # data-free: tree_from_lpt's assign_tree membership must be self-consistent with the descriptor
    LPT_CASES = [
        # tau edges, per-tau-group lambda edges (mixed split/unsplit)
        ([-0.63, 0.35, 1.23, 7.0], [[3.0, 3.8, 5.0], [3.0, 5.0], [3.0, 4.2, 5.0]]),
        ([-0.63, 1.0, 7.0], [[3.0, 3.5, 5.0], [3.0, 5.0]]),  # split, unsplit
        (
            [-0.63, 0.3488, 1.2275, 2.885, 7.0],
            [[3.0, 3.82, 5.0], [3.0, 3.65, 5.0], [3.0, 5.0], [3.0, 3.8, 5.0]],
        ),
    ]

    def test_assign_tree_membership_self_consistent(self):
        # for several lpt binnings (incl. mixed split/unsplit tau groups) and thousands of
        # random (tau_rosseland, wavelength) points, every assigned sub-bin lies inside its leaf
        # rectangle (>= lo, < hi) and out-of-window points are -1.
        rng = np.random.default_rng(7)
        n = 4000
        for tau_edges, lpt in self.LPT_CASES:
            with self.subTest(tau=tau_edges, lpt=lpt):
                tau = [float(e) for e in tau_edges]
                tree = qo.tree_from_lpt(tau, lpt)
                tv = 10.0 ** (-rng.uniform(-1, 7, n))
                wl = 10.0 ** (rng.uniform(3, 5, n)) / 1e8
                bi = ts.assign_tree(tv, wl, tree["root"], tree["window_tau"], tree["window_lam"])
                gte, gle = ts.build_group_specs_tree(tree["root"], tree["window_tau"], tree["window_lam"])
                # sanity: most points land somewhere (not all rejected)
                self.assertGreater((bi >= 0).sum(), n // 4)
                xs = np.log10(wl * 1e8)
                ys = -np.log10(np.clip(tv, 1e-300, None))
                for i in np.flatnonzero(bi >= 0):
                    te, le = gte[bi[i]], gle[bi[i]]
                    self.assertTrue(te[0] <= ys[i] < te[1], (i, te.tolist(), ys[i]))
                    self.assertTrue(le[0] <= xs[i] < le[1], (i, le.tolist(), xs[i]))

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
            grow_tol=0.0,  # isolate topology (midpoint-only) from the grow bar: accept any win
            beam_width=1,
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
            beam_width=3,
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
            beam_width=3,
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
            beam_width=3,
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
            beam_width=3,
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
            beam_width=3,
            max_groups=5,
            score_fn=_tree_dev_score(),
            max_evals=5000,
            binning_tree=first["binning_tree"],
        )
        self.assertTrue(qo._tree_feasible(again["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
        self.assertLessEqual(again["rms"], first["rms"] + 1e-6)

    def test_beam_coarsens_saturated_per_group_lambda_seed(self):
        # Regression: a per-group-lambda warm start that already meets/exceeds the leaf cap
        # (the webapp's default: 4 tau-groups x 2 lambda-cells = 8 leaves) used to saturate the
        # grow guard `_n_leaves < max_groups`, leaving the non-greedy beam search dead and the
        # result pinned at the seed leaf count (8), ignoring max_groups. The optimizer must
        # coarsen such a seed to the tau skeleton + lambda window so the beam can grow back out
        # -- the result must respect the cap and reach it (leaf-count score rewards splitting).
        res = qo.optimize_qrad(
            [-0.63, 0.3488, 1.2275, 2.885, 7.0],
            [3.0, 5.0],
            flags=[True, True, True, True],
            tree=True,
            grow=True,
            beam_width=3,
            max_groups=5,
            score_fn=_tree_leafcount_score(),
            max_evals=5000,
            lambda_edges_per_tau=[[3.0, 3.8, 5.0]] * 4,  # 4 tau-groups x 2 lambda-cells = 8 leaves
        )
        self.assertTrue(res["tree"])
        self.assertEqual(res["n_leaves"], 5)  # coarsened to 4, then beam grew to the cap (was 8, stuck)
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))

    def test_staged_seeding_splits_bottom_two(self):
        # Staged seeding: N=3 equally-spaced tau cuts -> tau polish -> bottom-two lambda split
        # (5 leaves) -> normal path. Data-free bowl rewards leaves, so the count must hold.
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            grow=False,
            initial_tau_bins=3,
            max_groups=8,
            score_fn=_tree_leafcount_score(),
            max_evals=5000,
        )
        self.assertTrue(res["tree"])
        self.assertEqual(res["n_leaves"], 5)  # 3 tau bins + 2 extra leaves from the bottom-two split
        self.assertLessEqual(res["n_leaves"], 8)
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
        tags = [h["tag"] for h in res["history"]]
        self.assertIn("staged-tau", tags)
        self.assertIn("staged-lambda", tags)

    def test_hard_sync_survives_grow_and_polish(self):
        # Synced bottom-two lambda cuts must stay equal through grow + polish, not just staging.
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            grow=True,
            beam_width=2,
            initial_tau_bins=3,
            max_groups=6,
            score_fn=_tree_dev_score(tau_target=1.5, lam_target=3.8),
            max_evals=3000,
        )
        by_sync: dict[str, list] = {}

        def walk(n):
            if n.get("leaf") or "axis" not in n:
                return
            if n.get("sync"):
                by_sync.setdefault(str(n["sync"]), []).append(round(float(n["at"]), 4))
            walk(n["lo"])
            walk(n["hi"])

        walk(res["binning_tree"]["root"])
        self.assertTrue(by_sync, "expected surviving synced cuts")
        for sid, ats in by_sync.items():
            self.assertEqual(len(set(ats)), 1, f"sync group {sid} desynced: {ats}")


class TestTauScan(unittest.TestCase):
    """Data-free tests for the LHS tau-cut seed scan (injected analytic bowl, no ODF)."""

    @staticmethod
    def _cuts(tree):
        return qo._tau_cuts_sorted(tree)

    def _bowl_cost(self, target):
        target = np.asarray(target, float)

        def cost_tree(tree):
            cuts = np.asarray(self._cuts(tree), float)
            n = min(len(cuts), len(target))
            return float(1e8 + 1e7 * np.sum((cuts[:n] - target[:n]) ** 2))

        return cost_tree

    def _bowl_score_fn(self, target):
        cost = self._bowl_cost(np.asarray(target, float))

        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            return {"rms": cost(binning_tree), "max_abs": 0.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        return score

    def test_scan_returns_separated_seeds_sorted(self):
        target = [1.0, 3.0]
        seeds = qo.scan_tau_seeds(
            64,
            [-0.63, 7.0],
            [3.0, 5.0],
            3,
            min_gap_tau=0.15,
            dtau=0.5,
            n_keep=5,
            cost_tree=self._bowl_cost(target),
            rng=0,
        )
        self.assertTrue(1 <= len(seeds) <= 5)  # dedup after the light polish may collapse seeds
        costs = [c for c, _ in seeds]
        self.assertEqual(costs, sorted(costs))  # best-first
        cuts = [self._cuts(t) for _, t in seeds]
        for c in cuts:  # right shape; pairwise separated by min_gap (the scan's contract)
            self.assertEqual(len(c), 2)
        for i in range(len(cuts)):
            for j in range(i + 1, len(cuts)):
                self.assertGreaterEqual(min(abs(a - b) for a, b in zip(cuts[i], cuts[j])), 0.15 - 1e-9)
        best = cuts[0]  # winner lands in the bowl minimum region
        self.assertTrue(all(abs(b - t) < 0.75 for b, t in zip(best, target)), best)

    def test_staged_scan_plan_order(self):
        target = [1.0, 3.0]
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            grow=False,
            initial_tau_bins=3,
            initial_tau_scan=64,
            max_groups=8,
            score_fn=self._bowl_score_fn(target),
            max_evals=5000,
        )
        tags = [h["tag"] for h in res["history"]]
        self.assertEqual(tags, sorted(tags, key=tags.index))  # no duplicates by construction
        order = ["start", "tau-scan", "staged-tau", "staged-lambda"]
        idx = [tags.index(t) for t in order]
        self.assertEqual(idx, sorted(idx))  # plan order preserved
        self.assertLessEqual(res["rms"], res["rms0"])  # converged scan winner: polish must never worsen

    def test_seed_reproducible(self):
        kw = dict(
            tau_window=[-0.63, 7.0],
            lam_window=[3.0, 5.0],
            n_bins=3,
            min_gap_tau=0.15,
            dtau=0.5,
            n_keep=5,
            rng=7,
        )
        target = [1.0, 3.0]
        a = qo.scan_tau_seeds(64, cost_tree=self._bowl_cost(target), **kw)
        b = qo.scan_tau_seeds(64, cost_tree=self._bowl_cost(target), **kw)
        self.assertEqual([c for c, _ in a], [c for c, _ in b])
        self.assertEqual([self._cuts(t) for _, t in a], [self._cuts(t) for _, t in b])

    def test_seeds_differ_across_rng(self):
        target = [1.0, 3.0]
        a = qo.scan_tau_seeds(
            64,
            [-0.63, 7.0],
            [3.0, 5.0],
            3,
            min_gap_tau=0.15,
            dtau=0.5,
            n_keep=5,
            cost_tree=self._bowl_cost(target),
            rng=0,
        )
        b = qo.scan_tau_seeds(
            64,
            [-0.63, 7.0],
            [3.0, 5.0],
            3,
            min_gap_tau=0.15,
            dtau=0.5,
            n_keep=5,
            cost_tree=self._bowl_cost(target),
            rng=1,
        )
        self.assertNotEqual([self._cuts(t) for _, t in a], [self._cuts(t) for _, t in b])


class TestBottomScan(unittest.TestCase):
    """Data-free tests for the flipped bottom-region (L, t_lo, t_hi) scan (no ODF)."""

    @staticmethod
    def _triple_cost(target):
        tL, ta, tb = (float(v) for v in target)

        def cost_tree(tree):
            L, a, b = qo._bottom_triple(tree)
            return float(1e8 + 1e7 * ((L - tL) ** 2 + (a - ta) ** 2 + (b - tb) ** 2))

        return cost_tree

    def _triple_score_fn(self, target):
        cost = self._triple_cost(np.asarray(target, float))

        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            try:
                rms = cost(binning_tree)
            except (KeyError, TypeError):
                rms = 1e8 + 1e7 * n  # non-flipped seed shape (tau-only staging): never the winner
            return {"rms": rms, "max_abs": 0.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        return score

    def test_scan_returns_separated_triples_sorted(self):
        target = [3.9, 0.5, 1.5]
        out = qo.scan_bottom_columns(
            64,
            [-0.63, 7.0],
            [3.0, 5.0],
            2.87,
            min_gap_tau=0.15,
            min_gap_lam=0.10,
            n_keep=3,
            cost_tree=self._triple_cost(target),
            rng=0,
        )
        self.assertTrue(1 <= len(out) <= 3)
        costs = [c for c, _ in out]
        self.assertEqual(costs, sorted(costs))  # best-first
        triples = [qo._bottom_triple(t) for _, t in out]
        for L, a, b in triples:  # every triple feasible on its own axes
            self.assertTrue(L - 3.0 >= 0.10 - 1e-9 and 5.0 - L >= 0.10 - 1e-9)
            for v in (a, b):
                self.assertTrue(v - -0.63 >= 0.15 - 1e-9 and 2.87 - v >= 0.15 - 1e-9)
        for i in range(len(triples)):
            for j in range(i + 1, len(triples)):
                sep = (
                    abs(triples[i][0] - triples[j][0]) >= 0.10 - 1e-9
                    or abs(triples[i][1] - triples[j][1]) >= 0.15 - 1e-9
                    or abs(triples[i][2] - triples[j][2]) >= 0.15 - 1e-9
                )
                self.assertTrue(sep, (triples[i], triples[j]))  # OR-separated
        best = triples[0]  # winner lands in the bowl minimum region
        self.assertTrue(all(abs(b - t) < 0.6 for b, t in zip(best, target)), best)

    def test_scan_needs_cost_tree(self):
        with self.assertRaises(ValueError):
            qo.scan_bottom_columns(8, [-0.63, 7.0], [3.0, 5.0], 2.87, cost_tree=None)
        self.assertEqual(qo.scan_bottom_columns(0, [-0.63, 7.0], [3.0, 5.0], 2.87, cost_tree=lambda t: 1.0), [])

    def test_staged_scan_on_yields_flipped_shape(self):
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            grow=False,
            initial_tau_bins=3,
            staged_lambda_scan=64,
            max_groups=8,
            score_fn=self._triple_score_fn([3.9, 0.5, 1.5]),
            max_evals=5000,
        )
        self.assertEqual(res["n_leaves"], 5)
        root = res["binning_tree"]["root"]
        self.assertEqual(root["axis"], "tau")  # flipped: tau@top at the root
        self.assertTrue(root["hi"].get("leaf"))  # top leaf spans the full lambda width
        bot = root["lo"]
        self.assertEqual(bot["axis"], "lam")  # bottom region splits on lambda
        self.assertEqual(bot["lo"]["axis"], "tau")
        self.assertEqual(bot["hi"]["axis"], "tau")  # per-column tau cuts
        rects = list(qo._leaf_rects(res["binning_tree"]["root"], qo._root_rect(res["binning_tree"])))
        lhis = [r[3] for r in rects if abs(r[1] - float(root["at"])) < 1e-9]
        self.assertTrue(lhis and abs(max(lhis) - 5.0) < 1e-9 and abs(min(r[2] for r in rects) - 3.0) < 1e-9)
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))

    def test_staged_scan_off_keeps_sync_path(self):
        # scan=0 must preserve the current mid-window bottom-lam pair (synced cut survives).
        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            grow=False,
            initial_tau_bins=3,
            max_groups=8,
            score_fn=_tree_leafcount_score(),
            max_evals=5000,
        )
        self.assertEqual(res["n_leaves"], 5)
        by_sync: dict[str, list] = {}

        def walk(n):
            if n.get("leaf") or "axis" not in n:
                return
            if n.get("sync"):
                by_sync.setdefault(str(n["sync"]), []).append(round(float(n["at"]), 4))
            walk(n["lo"])
            walk(n["hi"])

        walk(res["binning_tree"]["root"])
        self.assertIn("bottom-lam", by_sync)
        self.assertEqual(len(set(by_sync["bottom-lam"])), 1)


class TestBottomAllocScan(unittest.TestCase):
    """Data-free tests for the per-column tau-depth allocation scan (no ODF)."""

    _TL = 3.83
    _T1 = 1.02
    _T2 = 2.09
    _BASE = 1e8

    @staticmethod
    def _alloc_cost(target):
        TL, T1, T2 = (float(v) for v in target)

        def cost_tree(tree):
            L, lo, hi = qo._bottom_alloc(tree)
            if len(lo) == 2 and len(hi) == 0:  # depth stacked in the low-lambda column (the winner)
                return 1e8 + 1e7 * ((L - TL) ** 2 + (lo[0] - T1) ** 2 + (lo[1] - T2) ** 2)
            return 1e8 + 1e7 * 4.0  # symmetric / other allocations: high floor

        return cost_tree

    def _alloc_score_fn(self, target):
        cost = self._alloc_cost(np.asarray(target, float))

        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            try:
                rms = cost(binning_tree)
            except (KeyError, TypeError):
                rms = 1e8 + 1e7 * 9.0  # non-flipped seed shape (tau-only staging)
            return {"rms": rms, "max_abs": 0.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        return score

    def test_alloc_tree_shape(self):
        t = qo._bottom_alloc_tree([-0.63, 7.0], [3.0, 5.0], 4.37, 3.83, [1.02, 2.09], [])
        root = t["root"]
        self.assertEqual(root["axis"], "tau")
        self.assertAlmostEqual(float(root["at"]), 4.37)
        self.assertTrue(root["hi"].get("leaf"))  # deep leaf spans the full lambda width
        bot = root["lo"]
        self.assertEqual(bot["axis"], "lam")
        self.assertAlmostEqual(float(bot["at"]), 3.83)
        stack = bot["lo"]
        self.assertEqual(stack["axis"], "tau")  # dense column holds the tau stack
        self.assertAlmostEqual(float(stack["at"]), 1.02)
        self.assertEqual(stack["hi"]["axis"], "tau")
        self.assertAlmostEqual(float(stack["hi"]["at"]), 2.09)
        self.assertTrue(bot["hi"].get("leaf"))  # sparse column undivided
        self.assertEqual(qo._n_leaves(t), 5)
        self.assertTrue(qo._tree_feasible(t, qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
        L, lo, hi = qo._bottom_alloc(t)  # extractor round-trips the allocation
        self.assertAlmostEqual(L, 3.83)
        self.assertEqual((len(lo), len(hi)), (2, 0))

    def test_scan_reaches_asymmetric_basin(self):
        out = qo.scan_bottom_allocated(
            64,
            [-0.63, 7.0],
            [3.0, 5.0],
            4.37,
            min_gap_tau=0.15,
            min_gap_lam=0.10,
            n_keep=3,
            allocations=[(0, 2), (1, 1), (2, 0)],
            cost_tree=self._alloc_cost([self._TL, self._T1, self._T2]),
            rng=0,
        )
        self.assertTrue(1 <= len(out) <= 3)
        costs = [c for c, _ in out]
        self.assertEqual(costs, sorted(costs))  # best-first
        self.assertLess(out[0][0], 1e8 + 2e7)  # well below the wrong-allocation floor
        L, lo, hi = qo._bottom_alloc(out[0][1])
        self.assertEqual((len(lo), len(hi)), (2, 0))  # winner is the asymmetric allocation
        for c, t in out:
            self.assertTrue(qo._tree_feasible(t, qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
            self.assertEqual(qo._n_leaves(t), 5)

    def test_coarse_grid_cannot_represent_winner_tau(self):
        # The winner's tau1=1.02 is invisible to the coarse 0.5 grid (it skips 0.87 -> 1.37, so no
        # point is within 0.1 of 1.02); the fine 0.1 grid has 0.97/1.07. This is the mechanism the
        # staged-lambda fine scan fixes: the coarse scan literally cannot seed the winner's tau stack.
        coarse = qo._grid_points(-0.63, 4.37, 0.5)[1:-1]
        fine = qo._grid_points(-0.63, 4.37, 0.1)[1:-1]
        self.assertGreater(min(abs(g - 1.02) for g in coarse), 0.1)
        self.assertLess(min(abs(g - 1.02) for g in fine), 0.06)

    def test_scan_needs_cost_tree(self):
        with self.assertRaises(ValueError):
            qo.scan_bottom_allocated(8, [-0.63, 7.0], [3.0, 5.0], 4.37, cost_tree=None)
        self.assertEqual(qo.scan_bottom_allocated(0, [-0.63, 7.0], [3.0, 5.0], 4.37, cost_tree=lambda t: 1.0), [])

    def test_staged_scan_allocation_reaches_winner_basin(self):
        # Ablation: the bowl's global min sits at the asymmetric (2,0) allocation. The symmetric
        # one-cut-per-column scan (scan_allocation=False) can only build the (1,1) floor, while
        # scan_allocation=True seeds the winner directly -- no downstream topology cliff needed.
        kw = dict(
            flags=[True],
            grow=False,
            initial_tau_bins=3,
            staged_lambda_scan=64,
            max_groups=5,
            score_fn=self._alloc_score_fn([self._TL, self._T1, self._T2]),
            max_evals=5000,
        )
        sym = qo.optimize_qrad([-0.63, 7.0], [3.0, 5.0], scan_allocation=False, seed=0, **kw)
        asym = qo.optimize_qrad([-0.63, 7.0], [3.0, 5.0], scan_allocation=True, seed=0, **kw)
        self.assertLess(asym["rms"], sym["rms"])
        self.assertLess(asym["rms"], 1e8 + 2e7)  # reached the asymmetric bowl, not the 4e7 floor
        root = asym["binning_tree"]["root"]
        self.assertEqual(root["axis"], "tau")
        bot = root["lo"]
        self.assertEqual(bot["axis"], "lam")
        self.assertTrue(qo._is_leaf(bot["hi"]))  # sparse column undivided
        self.assertEqual(bot["lo"]["axis"], "tau")  # dense column holds the stack
        L, lo, hi = qo._bottom_alloc(asym["binning_tree"])
        self.assertEqual((len(lo), len(hi)), (2, 0))
        self.assertEqual(asym["n_leaves"], 5)
        self.assertTrue(qo._tree_feasible(asym["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))


class TestDeterministicFineGrid(unittest.TestCase):
    """Data-free tests for the deterministic photospheric fine-grid scan (no ODF)."""

    _TL = 3.83
    _T1 = 1.02
    _T2 = 2.09

    @staticmethod
    def _alloc_cost(target):
        TL, T1, T2 = (float(v) for v in target)

        def cost_tree(tree):
            L, lo, hi = qo._bottom_alloc(tree)
            if len(lo) == 2 and len(hi) == 0:  # depth stacked in the low-lambda column (the winner)
                return 1e8 + 1e7 * ((L - TL) ** 2 + (lo[0] - T1) ** 2 + (lo[1] - T2) ** 2)
            return 1e8 + 1e7 * 4.0  # symmetric / other allocations: high floor

        return cost_tree

    def test_fine_grid_beats_coarse_position(self):
        # The winner's tau1=1.02 falls between coarse-grid points (0.87/1.37), so the coarse
        # winner sits >= 0.15 off in tau1; the fine grid (dtau=0.2 over [0.4, 2.2]) lands on 1.0.
        cost = self._alloc_cost([self._TL, self._T1, self._T2])
        coarse_tree = qo._bottom_alloc_tree([-0.63, 7.0], [3.0, 5.0], 4.37, 3.75, [0.87, 2.37], [])
        coarse_cost = cost(coarse_tree)
        out = qo.scan_bottom_fine_grid(
            [-0.63, 7.0],
            [3.0, 5.0],
            4.37,
            min_gap_tau=0.15,
            min_gap_lam=0.10,
            n_keep=3,
            allocations=[(2, 0), (0, 2)],
            cost_tree=cost,
        )
        self.assertTrue(1 <= len(out) <= 3)
        costs = [c for c, _ in out]
        self.assertEqual(costs, sorted(costs))  # best-first
        self.assertLess(out[0][0], coarse_cost)  # the grid sampled inside the narrow well
        L, lo, hi = qo._bottom_alloc(out[0][1])
        self.assertEqual((len(lo), len(hi)), (2, 0))  # winner is the asymmetric allocation
        self.assertLess(abs(L - self._TL), 0.06)
        self.assertLess(abs(lo[0] - self._T1), 0.11)
        for c, t in out:
            self.assertTrue(qo._tree_feasible(t, qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
            self.assertEqual(qo._n_leaves(t), 5)

    def test_fine_grid_is_deterministic(self):
        cost = self._alloc_cost([self._TL, self._T1, self._T2])
        kw = dict(
            tau_window=[-0.63, 7.0],
            lam_window=[3.0, 5.0],
            t_top=4.37,
            min_gap_tau=0.15,
            min_gap_lam=0.10,
            n_keep=3,
            allocations=[(2, 0), (0, 2)],
            cost_tree=cost,
        )
        a = qo.scan_bottom_fine_grid(**kw)
        b = qo.scan_bottom_fine_grid(**kw)
        self.assertEqual([c for c, _ in a], [c for c, _ in b])  # no RNG anywhere
        self.assertEqual([qo._tree_signature(t) for _, t in a], [qo._tree_signature(t) for _, t in b])

    def test_fine_grid_needs_cost_tree(self):
        with self.assertRaises(ValueError):
            qo.scan_bottom_fine_grid([-0.63, 7.0], [3.0, 5.0], 4.37, cost_tree=None)


class TestStagedPortfolio(unittest.TestCase):
    """Data-free tests for the multi-seed staged portfolio (no ODF)."""

    @staticmethod
    def _bowl_score(target):
        target = np.asarray(target, float)

        def cost_tree(tree):
            cuts = np.asarray(qo._tau_cuts_sorted(tree), float)
            n = min(len(cuts), len(target))
            return float(1e8 + 1e7 * np.sum((cuts[:n] - target[:n]) ** 2))

        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            return {"rms": cost_tree(binning_tree), "max_abs": 0.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        return score

    def _run(self, **kw):
        base = dict(
            flags=[True],
            grow=False,
            beam_width=1,
            initial_tau_bins=3,
            initial_tau_scan=32,
            staged_lambda_scan=32,
            max_groups=8,
            score_fn=self._bowl_score([1.0, 3.0]),
            max_evals=5000,
        )
        base.update(kw)
        return qo.optimize_qrad([-0.63, 7.0], [3.0, 5.0], **base)

    def test_single_seed_matches_default(self):
        # n_staged_seeds=1 must reproduce today's single-seed path: same tags, same rms, same tree.
        a = self._run(seed=3)
        b = self._run(seed=3, n_staged_seeds=1)
        self.assertEqual([h["tag"] for h in a["history"]], [h["tag"] for h in b["history"]])
        self.assertEqual(a["rms"], b["rms"])
        self.assertEqual(qo._tree_signature(a["binning_tree"]), qo._tree_signature(b["binning_tree"]))

    def test_portfolio_never_loses(self):
        # Best-of-K over distinct rng streams must beat-or-tie any single stream on the same bowl.
        singles = [self._run(seed=s)["rms"] for s in range(5)]
        port = self._run(seed=0, n_staged_seeds=5)
        self.assertLessEqual(port["rms"], min(singles))
        tags = [h["tag"] for h in port["history"]]
        self.assertIn("staged-tau:0", tags)
        self.assertIn("staged-lambda:0", tags)

    def test_single_basin_signature_equal(self):
        # One global basin (seed scans off): every member stages the same fixed seed tree,
        # so the winner's signature equals the single-seed result.
        a = self._run(seed=3, initial_tau_scan=0, staged_lambda_scan=0)
        b = self._run(seed=3, initial_tau_scan=0, staged_lambda_scan=0, n_staged_seeds=3)
        self.assertEqual(qo._tree_signature(a["binning_tree"]), qo._tree_signature(b["binning_tree"]))


class TestMainGroupingDispatch(unittest.TestCase):
    """Regression guard for the main() rewrite (P3): every CLI grouping mode resolves to the
    per-tau-group lambda-edge list (``lpt``) feeding the single guillotine-tree IR, and the tree
    membership produced by main()'s path (``_resolve_grouping_inputs`` -> ``tree_from_lpt`` ->
    ``assign_tree``) is self-consistent with the descriptor. Data-free."""

    @staticmethod
    def _subbins(rng, n, ylo, yhi, xlo, xhi):
        y = rng.uniform(ylo, yhi, n)
        x = rng.uniform(xlo, xhi, n)
        return 10.0 ** (-y), 10.0**x / 1e8

    def _verify(self, gi, tau_edges, seed):
        # main()'s grouping path: lpt -> tree_from_lpt -> assign_tree + build_group_specs_tree.
        tree = qo.tree_from_lpt(list(tau_edges), gi["lpt"])
        tw, lw = tree["window_tau"], tree["window_lam"]
        rng = np.random.default_rng(seed)
        n = 6000
        tv, wl = self._subbins(rng, n, tw[0], tw[1], lw[0], lw[1])
        bi = ts.assign_tree(tv, wl, tree["root"], tw, lw)
        gte, gle = ts.build_group_specs_tree(tree["root"], [tw[0], tw[1]], lw)
        # sampling across the whole window -> almost everything is assigned
        self.assertGreater((bi >= 0).sum(), n * 0.9)
        # every assigned sub-bin lies inside its leaf rectangle (>= lo, < hi)
        xs = np.log10(wl * 1e8)
        ys = -np.log10(np.clip(tv, 1e-300, None))
        for i in np.flatnonzero(bi >= 0):
            te, le = gte[bi[i]], gle[bi[i]]
            self.assertTrue(te[0] <= ys[i] < te[1], (i, te.tolist(), ys[i]))
            self.assertTrue(le[0] <= xs[i] < le[1], (i, le.tolist(), xs[i]))
        return gte.shape[0]

    def test_single_cell_uniform(self):
        tau = [-0.63, 0.15, 1.5, 3.8, 7.0]
        gi = ts._resolve_grouping_inputs(tau, [3.0, 5.0], None, [], [])
        self.assertEqual(gi["mode"], "uniform")
        self.assertIsNone(gi["split_flags"])
        self.assertIsNone(gi["lambda_edges_per_tau"])
        self.assertEqual(gi["lpt"], [[3.0, 5.0]] * (len(tau) - 1))
        n_groups = self._verify(gi, tau, seed=1)
        self.assertEqual(n_groups, len(tau) - 1)  # 1 lambda cell x n_tau

    def test_split_flag(self):
        tau = [-0.63, 0.35, 1.23, 2.89, 7.0]  # n_tau = 4
        gi = ts._resolve_grouping_inputs(tau, [3.0, 3.8, 5.0], "1010", [], [])
        self.assertEqual(gi["mode"], "split-lambda")
        self.assertEqual(gi["split_flags"], [True, False, True, False])
        # flagged groups subdivide into the lambda cells; unsplit groups span [3, 5]
        self.assertEqual(
            gi["lpt"],
            [[3.0, 3.8, 5.0], [3.0, 5.0], [3.0, 3.8, 5.0], [3.0, 5.0]],
        )
        n_groups = self._verify(gi, tau, seed=2)
        self.assertEqual(n_groups, 2 * 2 + 2 * 1)  # 2 flagged (2 cells) + 2 unsplit (1)

    def test_per_tau_lambda(self):
        tau = [-0.63, 0.3488, 1.2275, 2.885, 7.0]  # n_tau = 4
        specs = ["3,3.82,5", "3,3.65,5", "3,5", "3,3.8,5"]
        gi = ts._resolve_grouping_inputs(tau, [3.0, 5.0], None, specs, [])
        self.assertEqual(gi["mode"], "per-tau-lambda")
        self.assertIsNone(gi["split_flags"])
        self.assertEqual(gi["lambda_bin_edges"], [3.0, 5.0])  # outer window
        self.assertEqual(gi["n_lambda"], 1)
        self.assertEqual(
            gi["lpt"],
            [[3.0, 3.82, 5.0], [3.0, 3.65, 5.0], [3.0, 5.0], [3.0, 3.8, 5.0]],
        )
        n_groups = self._verify(gi, tau, seed=3)
        self.assertEqual(n_groups, 2 + 2 + 1 + 2)  # (len(edges_k) - 1) per group

    def test_columns(self):
        lam = [3.0, 3.8, 5.0]
        specs = ["-0.63,0.1,1.0,3.2,7", "-0.63,0.8,1.6,2.5,7"]
        gi = ts._resolve_grouping_inputs([-0.63, 7.0], lam, None, [], specs)
        self.assertEqual(gi["mode"], "columns")
        self.assertEqual(gi["tree_kind"], "columns")
        self.assertIsNone(gi["split_flags"])
        self.assertIsNone(gi["lambda_edges_per_tau"])
        self.assertEqual(
            gi["tau_per_lambda"],
            [[-0.63, 0.1, 1.0, 3.2, 7.0], [-0.63, 0.8, 1.6, 2.5, 7.0]],
        )
        # columns path: tree_from_columns -> assign_tree + build_group_specs_tree.
        tree = qo.tree_from_columns(list(lam), gi["tau_per_lambda"])
        tw, lw = tree["window_tau"], tree["window_lam"]
        rng = np.random.default_rng(4)
        n = 6000
        tv, wl = self._subbins(rng, n, tw[0], tw[1], lw[0], lw[1])
        bi = ts.assign_tree(tv, wl, tree["root"], tw, lw)
        gte, gle = ts.build_group_specs_tree(tree["root"], [tw[0], tw[1]], lw)
        self.assertGreater((bi >= 0).sum(), n * 0.9)
        xs = np.log10(wl * 1e8)
        ys = -np.log10(np.clip(tv, 1e-300, None))
        for i in np.flatnonzero(bi >= 0):
            te, le = gte[bi[i]], gle[bi[i]]
            self.assertTrue(te[0] <= ys[i] < te[1], (i, te.tolist(), ys[i]))
            self.assertTrue(le[0] <= xs[i] < le[1], (i, le.tolist(), xs[i]))
        self.assertEqual(gte.shape[0], 4 + 4)  # (len(edges_k) - 1) per column


class TestFlatParamShim(unittest.TestCase):
    """The flat-input params (flags / per_group_lambda) no longer drive separate optimizers —
    they seed a guillotine tree (the shim in optimize_qrad). Calling optimize_qrad WITHOUT
    tree=True must still return a tree result whose leaf rectangles match tree_from_lpt of the
    equivalent flat inputs. grow is off so the structure is exactly the seed; a constant
    tree-aware score_fn means the polish never moves a cut (no strict improvement), so the
    seed rectangles are preserved verbatim."""

    @staticmethod
    def _const_score(tau, lam, flags, model, *, binning_tree=None, window=None):
        n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
        return {"rms": 1e8, "max_abs": 2e8, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

    @staticmethod
    def _rects(tree):
        return [
            (round(tlo, 4), round(thi, 4), round(llo, 4), round(lhi, 4))
            for tlo, thi, llo, lhi in qo._leaf_rects(tree["root"], qo._root_rect(tree))
        ]

    def test_flags_seed_matches_tree_from_lpt(self):
        tau = [-0.63, 0.35, 1.23, 7.0]
        lam = [3.0, 3.8, 5.0]
        flags = [True, False, True]
        lmin, lmax = lam[0], lam[-1]
        expected = qo.tree_from_lpt(tau, [list(lam) if f else [lmin, lmax] for f in flags])
        res = qo.optimize_qrad(
            tau,
            lam,
            flags=flags,
            grow=False,
            opt_tau=False,
            opt_lambda=False,
            opt_flags=False,
            beam_width=1,
            score_fn=self._const_score,
            max_evals=200,
        )
        self.assertTrue(res["tree"])
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
        self.assertEqual(self._rects(res["binning_tree"]), self._rects(expected))

    def test_per_group_lambda_seed_matches_tree_from_lpt(self):
        tau = [-0.63, 0.35, 1.23, 7.0]
        lam = [3.0, 3.8, 5.0]
        expected = qo.tree_from_lpt(tau, [list(lam) for _ in range(len(tau) - 1)])
        res = qo.optimize_qrad(
            tau,
            lam,
            flags=[True] * (len(tau) - 1),
            per_group_lambda=True,
            grow=False,
            opt_tau=False,
            opt_lambda=False,
            opt_flags=False,
            beam_width=1,
            score_fn=self._const_score,
            max_evals=200,
        )
        self.assertTrue(res["tree"])
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))
        self.assertEqual(self._rects(res["binning_tree"]), self._rects(expected))


class TestSaveLoadTree(unittest.TestCase):
    """save_tree/load_tree JSON round-trip: dict-equal + score-equal, validation, seed precedence."""

    @staticmethod
    def _leafcount_score(seen):
        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            seen.append(qo._tree_signature(binning_tree))
            n = sum(1 for _ in qo._leaf_rects(binning_tree["root"], qo._root_rect(binning_tree)))
            return {"rms": 1.0e8 / n, "max_abs": 1.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        return score

    def test_round_trip_identical_dict_and_score(self):
        tree = qo.tree_from_splits(
            [-0.63, 7.0],
            [3.0, 5.0],
            [{"axis": "tau", "tau": 1.23456789, "lam": 4.0}, {"axis": "lam", "tau": 0.0, "lam": 3.81234567}],
        )
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "tree.json")
            qo.save_tree(tree, p)
            raw = json.loads(Path(p).read_text())
            self.assertEqual(raw["window_tau"], [-0.63, 7.0])
            # 6-decimal rounding on save (not the 4-decimal wire payload)
            self.assertIn(1.234568, [raw["root"]["at"], raw["root"]["lo"].get("at"), raw["root"]["hi"].get("at")])
            back = qo.load_tree(p)
            self.assertEqual(back, raw)
            seen = []
            ev, _ = qo.make_evaluator("X", score_fn=self._leafcount_score(seen))
            c0, r0 = ev(binning_tree=back)
            c1, r1 = ev(binning_tree=raw)
            self.assertEqual(c0, c1)
            self.assertEqual(r0["rms"], r1["rms"])

    def test_load_validation_rejects_bad_files(self):
        with tempfile.TemporaryDirectory() as d:
            bad_json = Path(d) / "bad.json"
            bad_json.write_text("{not json")
            with self.assertRaises(ValueError):
                qo.load_tree(str(bad_json))
            missing = Path(d) / "missing.json"
            missing.write_text(json.dumps({"window_tau": [-0.63, 7.0]}))
            with self.assertRaises(ValueError):
                qo.load_tree(str(missing))
            bad_axis = Path(d) / "axis.json"
            bad_axis.write_text(
                json.dumps(
                    {
                        "window_tau": [-0.63, 7.0],
                        "window_lam": [3.0, 5.0],
                        "root": {"axis": "rho", "at": 1.0, "lo": {"leaf": True}, "hi": {"leaf": True}},
                    }
                )
            )
            with self.assertRaises(ValueError):
                qo.load_tree(str(bad_axis))
            bad_gap = Path(d) / "gap.json"
            bad_gap.write_text(
                json.dumps(
                    {
                        "window_tau": [-0.63, 7.0],
                        "window_lam": [3.0, 5.0],
                        "root": {"axis": "tau", "at": -0.62, "lo": {"leaf": True}, "hi": {"leaf": True}},
                    }
                )
            )
            with self.assertRaises(ValueError):
                qo.load_tree(str(bad_gap))

    def test_loaded_tree_wins_over_flags_seed(self):
        # grow=False refine-only: the result tree must equal the loaded tree (modulo the
        # 4-decimal result rounding), even though flags describe a different seed shape.
        warm = qo.tree_from_splits([-0.63, 7.0], [3.0, 5.0], [{"axis": "tau", "tau": 2.0, "lam": 4.0}])
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "warm.json")
            qo.save_tree(warm, p)
            loaded = qo.load_tree(p)
            seen = []
            res = qo.optimize_qrad(
                [-0.63, 0.35, 1.23, 7.0],
                [3.0, 3.8, 5.0],
                flags=[True, False, True],
                grow=False,
                beam_width=1,
                binning_tree=loaded,
                initial_tau_bins=None,
                score_fn=self._leafcount_score(seen),
                max_evals=200,
            )
            self.assertEqual(
                qo._tree_signature(res["binning_tree"]),
                qo._tree_signature(qo._round_tree(loaded)),
            )
            # staged seeding skipped when a tree is loaded: no staged tags in history
            tags = [h["tag"] for h in res["history"]]
            self.assertNotIn("staged-tau", tags)
            self.assertNotIn("staged-lambda", tags)


class TestTopologyTransplant(unittest.TestCase):
    """Data-free tests for the cross-column depth transplant (+ nested-removal support, no ODF)."""

    @staticmethod
    def _symmetric_seed(lam_at=1.0):
        # Loser-like staging: a shared lam cut with one tau cut per column (4 leaves).
        return {
            "window_tau": [0.0, 4.0],
            "window_lam": [0.0, 4.0],
            "root": {
                "axis": "lam",
                "at": float(lam_at),
                "lo": {"axis": "tau", "at": 1.0, "lo": {"leaf": True}, "hi": {"leaf": True}},
                "hi": {"axis": "tau", "at": 3.0, "lo": {"leaf": True}, "hi": {"leaf": True}},
            },
        }

    @staticmethod
    def _tau_cuts(node):
        out = []

        def walk(n):
            if n.get("leaf") or "axis" not in n:
                return
            if n["axis"] == "tau":
                out.append(float(n["at"]))
            walk(n["lo"])
            walk(n["hi"])

        walk(node)
        return sorted(out)

    @staticmethod
    def _budget():
        return qo._Budget(max_evals=10**9, max_seconds=1e12, state={"n_evals": 0}, t0=0.0)

    def _transplant_kwargs(self, cost, **over):
        kw = dict(
            cost_tree=cost,
            cfg=None,
            budget=self._budget(),
            max_groups=4,
            min_gap_tau=qo.MIN_GAP_TAU,
            min_gap_lam=qo.MIN_GAP_LAM,
            positions=(0.35, 0.5, 0.65),
        )
        kw.update(over)
        return kw

    def test_transplant_moves_depth_across_columns(self):
        # Mechanism: the first candidate collapses lo's tau cut and cuts hi's wide leaf at the
        # 0.35 fraction (1.05); the bowl minimum sits exactly at the resulting nested shape
        # (hi keeps its tau@3.0 cut, the new tau@1.05 cut nests inside), so it is adopted at once.
        seed = self._symmetric_seed(lam_at=3.0)
        BASE = 1e8
        target = np.asarray([1.05, 3.0])

        def cost(tree):
            got = np.asarray(self._tau_cuts(tree["root"]), float)
            return BASE + 1e7 * float(np.sum((got - target[: len(got)]) ** 2))

        calls = []
        best = [cost(seed)]

        def adopt(cand, new_path, joint_paths):
            c = cost(cand)
            calls.append((qo._tree_signature(cand), tuple(new_path), tuple(joint_paths), c))
            if c < best[0] - 1e-9:
                best[0] = c
                return True
            return False

        won = qo._try_transplant(seed, adopt=adopt, **self._transplant_kwargs(cost))
        self.assertTrue(won)
        self.assertTrue(calls)
        sig, new_path, joint, _c = calls[0]
        self.assertEqual(new_path, ("hi", "lo"))  # tau cut transplanted lo -> hi column
        self.assertEqual(joint, ((),))  # shared lam cut jointly polished
        expected = self._symmetric_seed(lam_at=3.0)
        expected["root"]["lo"] = {"leaf": True}
        expected["root"]["hi"]["lo"] = {"axis": "tau", "at": 1.05, "lo": {"leaf": True}, "hi": {"leaf": True}}
        self.assertEqual(sig, qo._tree_signature(expected))
        self.assertEqual(len(sig), 4)  # collapse freed 1 slot, the re-cut spent it

    def test_transplant_needs_joint_lambda_move(self):
        # End-to-end: the bowl's lam optimum is conditional on the shape (1.0 for the symmetric
        # seed, 1.35 once transplanted), so the transplanted shape at the OLD lam position costs
        # MORE than the seed -- only the transplant's joint lam polish crosses the gap.
        seed = self._symmetric_seed(lam_at=1.0)
        BASE = 1e8

        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            t = binning_tree
            n = sum(1 for _ in qo._leaf_rects(t["root"], qo._root_rect(t)))
            root = t["root"]
            cuts = self._tau_cuts(root["hi"]) if root.get("axis") == "lam" and not qo._is_leaf(root["hi"]) else []
            if root.get("axis") == "lam" and qo._is_leaf(root["lo"]) and cuts:
                L = float(root["at"])
                rms = BASE + 1e7 * ((L - 1.35) ** 2 + (cuts[0] - 1.05) ** 2)
            else:
                L = float(root["at"]) if root.get("axis") == "lam" else 1.0
                rms = BASE + 1e7 * ((L - 1.0) ** 2 + 0.1)
            return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        res = qo.optimize_qrad(
            [0.0, 4.0],
            [0.0, 4.0],
            flags=[True],
            binning_tree=seed,
            grow=True,
            beam_width=2,  # topology search runs (it is skipped for greedy beam_width == 1)
            max_groups=4,  # cap saturated: SPLIT disabled, only depth moves can win
            score_fn=score,
            max_evals=5000,
        )
        self.assertLess(res["rms"], res["rms0"])  # never-worsens, and the basin was reached
        root = res["binning_tree"]["root"]
        self.assertEqual(root["axis"], "lam")
        self.assertTrue(qo._is_leaf(root["lo"]))  # donor column collapsed to a leaf
        self.assertEqual(root["hi"]["axis"], "tau")  # donee column holds the tau stack
        self.assertAlmostEqual(float(root["at"]), 1.35, delta=0.06)
        self.assertAlmostEqual(float(root["hi"]["lo"]["at"]), 1.05, delta=0.06)
        self.assertEqual(res["n_leaves"], 4)
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))

    def test_nested_removal_skips_guarded(self):
        seed = self._symmetric_seed()
        plain = list(qo._iter_removable_with_path(seed["root"], qo._root_rect(seed)))
        self.assertEqual([f for _, _, f in plain], [1, 1, 3])  # two-leaf fast path first, root collapse last
        seed["root"]["lo"]["frozen"] = True  # lo column skeleton fixed
        got = list(qo._iter_removable_with_path(seed["root"], qo._root_rect(seed)))
        self.assertEqual([(p, f) for p, _, f in got], [(("hi",), 1)])  # root covers frozen: excluded too
        for path, _rect, freed in got:
            node = qo._node_at_path(seed["root"], path)
            self.assertEqual(qo._count_leaves(node) - 1, freed)
        both = self._symmetric_seed()
        both["root"]["lo"]["sync"] = "s"
        both["root"]["hi"]["sync"] = "s"
        self.assertEqual(list(qo._iter_removable_with_path(both["root"], qo._root_rect(both))), [])

    def test_transplant_respects_guards_and_cap(self):
        def never(cand, new_path, joint_paths):
            self.fail(f"adopt called for a guarded/capped tree: {new_path}")

        frozen = self._symmetric_seed()
        frozen["root"]["lo"]["frozen"] = True
        frozen["root"]["hi"]["frozen"] = True
        self.assertFalse(qo._try_transplant(frozen, adopt=never, **self._transplant_kwargs(lambda t: 1.0)))
        synced = self._symmetric_seed()
        synced["root"]["lo"]["sync"] = "s"
        synced["root"]["hi"]["sync"] = "s"
        self.assertFalse(qo._try_transplant(synced, adopt=never, **self._transplant_kwargs(lambda t: 1.0)))
        capped = self._symmetric_seed()
        self.assertFalse(  # 4 leaves, collapse frees 1, re-cut spends 1 -> 4 > 3: no room
            qo._try_transplant(capped, adopt=never, **self._transplant_kwargs(lambda t: 1.0, max_groups=3))
        )

    def test_transplant_keeps_frozen_lambda_fixed(self):
        # A frozen shared lam cut is jointly "polished" (a no-op) but never moved.
        seed = self._symmetric_seed(lam_at=3.0)
        seed["root"]["frozen"] = True
        BASE = 1e8

        def cost(tree):
            if qo._is_leaf(tree["root"]["lo"]) and not qo._is_leaf(tree["root"]["hi"]):
                t = float(tree["root"]["hi"]["lo"]["at"])
                return BASE + 1e7 * (t - 1.05) ** 2
            return BASE + 1e7 * 5.0

        best = [cost(seed)]

        def adopt(cand, new_path, joint_paths):
            c = cost(cand)
            if c < best[0] - 1e-9:
                best[0] = c
                return True
            return False

        self.assertTrue(qo._try_transplant(seed, adopt=adopt, **self._transplant_kwargs(cost)))
        self.assertEqual(float(seed["root"]["at"]), 3.0)  # input untouched (candidates are copies)

    # Hardcoded seed-3 constants: the 5-group winner (rms 4.8733e7) from the seed5g
    # RNG sweep. Root tau@TOP frozen (as staged freeze leaves it), dense column
    # tau-stacked at T1/T2 behind lam@L, sparse side + top leaf undivided.
    _SEED3_TOP = 4.37
    _SEED3_L = 3.83
    _SEED3_T1 = 1.02
    _SEED3_T2 = 2.0925

    @staticmethod
    def _seed3_loser():
        # 5-leaf loser (cap-saturated: SPLIT dead). S-lo holds a single tau cut at T1
        # (already at target), S-hi a single tau cut; lam at 4.13 (0.3 off basin); root
        # at 4.084 so the 0.35-fraction re-cut of the donee upper leaf lands at T2.
        # Donor = S-hi tau (two-leaf collapse frees 1 slot: 5-1+1=5 ok); donee = S-lo
        # upper leaf. The re-cut hits T2 nearly exactly, so the transplant's joint-L
        # walk only needs L -- which REALLOC's single-cut gate never moves (see bowl).
        return {
            "window_tau": [-0.63, 7.0],
            "window_lam": [3.0, 5.0],
            "root": {
                "axis": "tau",
                "at": 4.084,
                "lo": {
                    "axis": "lam",
                    "at": 4.13,
                    "lo": {"axis": "tau", "at": 1.02, "lo": {"leaf": True}, "hi": {"leaf": True}},
                    "hi": {"axis": "tau", "at": 1.47, "lo": {"leaf": True}, "hi": {"leaf": True}},
                },
                "hi": {"leaf": True},
                "frozen": True,
            },
        }

    def test_seed3_basin_reached_via_transplant(self):
        # Regression guard for the seed-3 basin: an analytic bowl whose minimum sits
        # exactly at the hardcoded seed-3 tiling must be reached from loser staging.
        # The wall sits at BASE-0.45e7 (95.5e6): the transplanted shape at old L costs
        # ~95.9e6 (rejected raw), but its joint-L walk dips to ~95.2e6 (passes); the
        # same tree through REALLOC's single-cut gate never moves L and stalls at
        # ~95.9e6 (rejected). Margins ~0.3e6, deterministic (analytic, no noise).
        seed = self._seed3_loser()
        BASE = 1e8

        def score(tau, lam, flags, model, *, binning_tree=None, window=None):
            t = binning_tree
            n = sum(1 for _ in qo._leaf_rects(t["root"], qo._root_rect(t)))
            root = t["root"]
            lo = root.get("lo") if root.get("axis") == "tau" else None
            # ONLY the nested stack scores the basin (smooth quadratic, no wall):
            # second tau cut inside the donee upper leaf.
            nested = (
                n == 5
                and root.get("axis") == "tau"
                and isinstance(lo, dict)
                and lo.get("axis") == "lam"
                and isinstance(lo.get("lo"), dict)
                and lo["lo"].get("axis") == "tau"
                and isinstance(lo["lo"].get("hi"), dict)
                and lo["lo"]["hi"].get("axis") == "tau"
                and qo._is_leaf(lo.get("hi", {}))
                and qo._is_leaf(root.get("hi", {}))
            )
            if nested:
                L = float(lo["at"])
                c = sorted(self._tau_cuts(lo))
                rms = (
                    BASE
                    - 0.5e7
                    + 1e7 * ((L - self._SEED3_L) ** 2 + (c[0] - self._SEED3_T1) ** 2 + (c[1] - self._SEED3_T2) ** 2)
                )
            else:
                rms = BASE - 0.45e7 if n == 5 else BASE + (0.0 if n < 5 else 5e7)
            return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        res = qo.optimize_qrad(
            [-0.63, 7.0],
            [3.0, 5.0],
            flags=[True],
            binning_tree=seed,
            grow=True,
            beam_width=2,  # topology search runs (skipped for greedy beam_width == 1)
            max_groups=5,  # cap saturated: SPLIT dead, only depth moves can win
            score_fn=score,
            max_evals=5000,
        )
        self.assertLess(res["rms"], res["rms0"])  # basin reached via the transplant cliff
        root = res["binning_tree"]["root"]
        self.assertEqual(root["axis"], "tau")
        # Root frozen at the loser's 4.084: the transplant reallocates depth below it.
        lo = root["lo"]
        self.assertEqual(lo["axis"], "lam")
        self.assertTrue(qo._is_leaf(lo["hi"]))  # sparse side undivided
        self.assertAlmostEqual(float(lo["at"]), self._SEED3_L, delta=0.1)
        stack = lo["lo"]
        self.assertEqual(stack["axis"], "tau")  # dense column holds the tau stack
        self.assertEqual(stack["hi"]["axis"], "tau")  # NESTED: second cut inside the upper leaf
        got = sorted(self._tau_cuts(lo))
        self.assertEqual(len(got), 2)  # depth transplanted into the dense column
        self.assertAlmostEqual(got[0], self._SEED3_T1, delta=0.1)
        self.assertAlmostEqual(got[1], self._SEED3_T2, delta=0.1)
        self.assertEqual(res["n_leaves"], 5)
        self.assertTrue(qo._tree_feasible(res["binning_tree"], qo.MIN_GAP_TAU, qo.MIN_GAP_LAM))


if __name__ == "__main__":
    unittest.main()
