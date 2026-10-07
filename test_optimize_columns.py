"""Data-free unit tests for the columns-constrained Q_rad optimizer.

All tests inject an analytic ``score_fn`` (no ODF): the cost is read off the
``binning_tree`` the optimizer passes (lambda-cut / tau-cut targets plus an optional
per-leaf reward), so the suite runs in seconds.
"""

import unittest

import qrad_optimize as qo
import tausort as ts


def _leaf_rects(tree):
    return list(qo._leaf_rects(tree["root"], qo._root_rect(tree)))


def columns_bowl(lam_targets, tau_targets_per_col, *, reward_leaf=0.0):
    """Analytic columns score_fn: quadratic bowls on the lambda cut and on each column's
    tau cuts (columns identified by their lambda span), minus a per-leaf reward (drives grow)."""

    def score(tau, lam, flags, model, *, binning_tree=None, window=None):
        tree = binning_tree
        rects = _leaf_rects(tree)
        lam_cuts = []

        def walk(node):
            if node.get("leaf") or "axis" not in node:
                return
            if node["axis"] == "lam":
                lam_cuts.append(float(node["at"]))
            walk(node["lo"])
            walk(node["hi"])

        walk(tree["root"])
        rms = 1.0e8
        for got, tgt in zip(sorted(lam_cuts), lam_targets):
            rms += 1.0e7 * (got - tgt) ** 2
        # interior tau cuts of each column = that column's leaf lo-edges minus the window lo
        tlo = tree["window_tau"][0]
        for ci, llo_c in enumerate(sorted({r[2] for r in rects})):
            if ci >= len(tau_targets_per_col):
                break
            t_cuts = sorted(t for t in {r[0] for r in rects if r[2] == llo_c} if t > tlo + 1e-9)
            for got, tt in zip(t_cuts, tau_targets_per_col[ci]):
                rms += 1.0e7 * (got - tt) ** 2
        rms -= reward_leaf * len(rects)
        return {"rms": rms, "max_abs": 2 * rms, "int_q_pct": 0.0, "n_empty": 0, "n_groups": len(rects)}

    return score


class TestColumnsStaysInFamily(unittest.TestCase):
    def test_result_matches_tree_from_columns(self):
        lam = [3.0, 3.4, 5.0]
        cols = [[-0.63, 0.5, 7.0], [-0.63, 2.0, 7.0]]
        res = qo.optimize_columns(
            lam,
            cols,
            score_fn=columns_bowl([3.8], [[1.5], [2.5]]),
            max_evals=2000,
            grow=False,
        )
        self.assertTrue(res.get("columns"))
        self.assertLess(res["rms"], res["rms0"])  # tilted problem: strict improvement
        expected = qo.tree_from_columns(res["lambda_edges"], res["tau_per_lambda"])
        self.assertEqual(qo._tree_signature(res["binning_tree"]), qo._tree_signature(expected))
        # lambda-major DFS leaf order (not just the rect set): band order depends on it
        got_rects = list(ts._iter_leaf_rects(res["binning_tree"]["root"], tuple(qo._root_rect(res["binning_tree"]))))
        exp_rects = list(ts._iter_leaf_rects(expected["root"], tuple(qo._root_rect(expected))))
        self.assertEqual(got_rects, exp_rects)
        # shared outer tau window preserved
        tlo, thi = res["tau_per_lambda"][0][0], res["tau_per_lambda"][0][-1]
        for c in res["tau_per_lambda"]:
            self.assertEqual((c[0], c[-1]), (tlo, thi))
        # lambda-cut count fixed, driven toward the target
        self.assertEqual(len(res["lambda_edges"]), len(lam))
        self.assertLess(abs(res["lambda_edges"][1] - 3.8), abs(3.4 - 3.8))


class TestColumnsGrow(unittest.TestCase):
    def test_grow_adds_group_where_rewarded(self):
        lam = [3.0, 5.0]
        cols = [[-0.63, 7.0]]
        res = qo.optimize_columns(
            lam,
            cols,
            score_fn=columns_bowl([], [[]], reward_leaf=2.0e6),
            max_evals=3000,
            max_groups=3,
            grow_tol_rel=0.0,
        )
        self.assertGreater(res["n_leaves"], 1)  # grew
        self.assertLessEqual(res["n_leaves"], 3)  # respects the cap
        self.assertLess(res["rms"], res["rms0"])
        expected = qo.tree_from_columns(res["lambda_edges"], res["tau_per_lambda"])
        self.assertEqual(qo._tree_signature(res["binning_tree"]), qo._tree_signature(expected))

    def test_no_grow_without_reward(self):
        # flat score: no structural move pays, so the single column stays a single leaf
        def flat(tau, lam, flags, model, *, binning_tree=None, window=None):
            n = len(_leaf_rects(binning_tree))
            return {"rms": 1.0e8, "max_abs": 1.0, "int_q_pct": 0.0, "n_empty": 0, "n_groups": n}

        res = qo.optimize_columns([3.0, 5.0], [[-0.63, 7.0]], score_fn=flat, max_evals=500, grow_tol_rel=0.0)
        self.assertEqual(res["n_leaves"], 1)


class TestColumnsInvalidSeeds(unittest.TestCase):
    def test_count_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "tau columns for"):
            qo.optimize_columns([3.0, 5.0], [[-0.63, 7.0], [-0.63, 7.0]], score_fn=columns_bowl([], [[]]))

    def test_window_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "outer tau window"):
            qo.optimize_columns(
                [3.0, 3.8, 5.0],
                [[-0.63, 1.0, 7.0], [-0.5, 1.0, 7.0]],
                score_fn=columns_bowl([3.8], [[1.0], [1.0]]),
            )

    def test_infeasible_gap_raises(self):
        with self.assertRaisesRegex(ValueError, "need span"):
            qo.optimize_columns(
                [3.0, 3.05],
                [[-0.63, 7.0]],
                min_gap_lam=0.10,
                score_fn=columns_bowl([3.02], [[]]),
            )


if __name__ == "__main__":
    unittest.main()
