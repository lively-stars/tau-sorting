"""Unit tests for per-lambda-column tau stacks (columns mode).

Data-free: parse/validate, tree_from_columns structure + error paths, resolver
dispatch, and the tree membership self-consistency (assign_tree rectangles match
the descriptor). Leaf order is lambda-major (column 0's tau stack first).
"""

import unittest

import numpy as np

import qrad_optimize as qo
import tausort as ts
from tausort import (
    assign_tree,
    build_group_specs_tree,
    parse_tau_per_lambda,
)


class TestParseTauPerLambda(unittest.TestCase):
    def test_valid(self):
        got = parse_tau_per_lambda(["-0.63,0.1,1.0,3.2,7", "-0.63 0.8 1.6 2.5 7"])
        self.assertEqual(got, [[-0.63, 0.1, 1.0, 3.2, 7.0], [-0.63, 0.8, 1.6, 2.5, 7.0]])

    def test_rejects_bad(self):
        import typer

        with self.assertRaises(typer.BadParameter):
            parse_tau_per_lambda(["7,-0.63"])  # not increasing
        with self.assertRaises(typer.BadParameter):
            parse_tau_per_lambda(["7"])  # < 2 edges


class TestTreeFromColumns(unittest.TestCase):
    def test_leaf_rectangles_lambda_major(self):
        tree = qo.tree_from_columns(
            [3.0, 3.8, 5.0],
            [[-0.63, 0.1, 1.0, 3.2, 7.0], [-0.63, 0.8, 1.6, 2.5, 7.0]],
        )
        self.assertEqual(tree["window_tau"], [-0.63, 7.0])
        self.assertEqual(tree["window_lam"], [3.0, 5.0])
        rects = list(ts._iter_leaf_rects(tree["root"], (-0.63, 7.0, 3.0, 5.0)))
        self.assertEqual(len(rects), 8)
        # column 0's tau stack first, then column 1's
        self.assertEqual(
            [tuple(round(v, 2) for v in r) for r in rects],
            [
                (-0.63, 0.1, 3.0, 3.8),
                (0.1, 1.0, 3.0, 3.8),
                (1.0, 3.2, 3.0, 3.8),
                (3.2, 7.0, 3.0, 3.8),
                (-0.63, 0.8, 3.8, 5.0),
                (0.8, 1.6, 3.8, 5.0),
                (1.6, 2.5, 3.8, 5.0),
                (2.5, 7.0, 3.8, 5.0),
            ],
        )

    def test_single_column_degenerate(self):
        tree = qo.tree_from_columns([3.0, 5.0], [[-0.63, 7.0]])
        rects = list(ts._iter_leaf_rects(tree["root"], (-0.63, 7.0, 3.0, 5.0)))
        self.assertEqual(rects, [(-0.63, 7.0, 3.0, 5.0)])

    def test_rejects_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "tau columns for"):
            qo.tree_from_columns([3.0, 5.0], [[-0.63, 7.0], [-0.63, 7.0]])

    def test_rejects_window_mismatch(self):
        with self.assertRaisesRegex(ValueError, "outer tau window"):
            qo.tree_from_columns([3.0, 3.8, 5.0], [[-0.63, 1.0, 7.0], [-0.5, 1.0, 7.0]])

    def test_rejects_bad_lambda(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            qo.tree_from_columns([3.0, 3.8, 3.8, 5.0], [[-0.63, 7.0]] * 3)


class TestColumnsMembership(unittest.TestCase):
    def test_assign_matches_descriptor(self):
        lam = [3.0, 3.8, 5.0]
        cols = [[-0.63, 0.1, 1.0, 3.2, 7.0], [-0.63, 0.8, 1.6, 2.5, 7.0]]
        tree = qo.tree_from_columns(lam, cols)
        tw, lw, root = tree["window_tau"], tree["window_lam"], tree["root"]
        rng = np.random.default_rng(7)
        y = rng.uniform(tw[0], tw[1], 4000)
        x = rng.uniform(lw[0], lw[1], 4000)
        bi = assign_tree(10.0 ** (-y), 10.0**x / 1e8, root, tw, lw)
        gte, gle = build_group_specs_tree(root, [tw[0], tw[1]], lw)
        self.assertGreater((bi >= 0).sum(), 3600)
        for i in np.flatnonzero(bi >= 0):
            te, le = gte[bi[i]], gle[bi[i]]
            self.assertTrue(te[0] <= y[i] < te[1])
            self.assertTrue(le[0] <= x[i] < le[1])
        self.assertEqual(gte.shape[0], 8)

    def test_t_junction_point_lands_hi_side(self):
        # point exactly on column 0's tau edge 1.0 but inside column 1's [0.8, 1.6] cell
        cols = [[-0.63, 1.0, 7.0], [-0.63, 0.8, 1.6, 7.0]]
        tree = qo.tree_from_columns([3.0, 3.8, 5.0], cols)
        tw, lw, root = tree["window_tau"], tree["window_lam"], tree["root"]
        bi = assign_tree(np.array([10.0**-1.0]), np.array([10.0**4.0 / 1e8]), root, tw, lw)
        gte, gle = build_group_specs_tree(root, [tw[0], tw[1]], lw)
        g = int(bi[0])
        self.assertTrue(gle[g, 0] >= 3.8)  # column 1
        self.assertTrue(gte[g, 0] <= 1.0 < gte[g, 1])


class TestColumnsResolver(unittest.TestCase):
    def test_resolve_and_reject(self):
        import typer

        gi = ts._resolve_grouping_inputs([-0.63, 7.0], [3.0, 3.8, 5.0], None, [], ["-0.63,0.1,7", "-0.63,0.8,7"])
        self.assertEqual(gi["mode"], "columns")
        self.assertEqual(gi["tree_kind"], "columns")
        self.assertEqual(gi["tau_per_lambda"], [[-0.63, 0.1, 7.0], [-0.63, 0.8, 7.0]])
        with self.assertRaises(typer.BadParameter):
            ts._resolve_grouping_inputs([-0.63, 7.0], [3.0, 3.8, 5.0], "11", [], ["-0.63,7"])
        with self.assertRaises(typer.BadParameter):
            ts._resolve_grouping_inputs([-0.63, 7.0], [3.0, 3.8, 5.0], None, [], ["-0.63,7"])
        with self.assertRaises(typer.BadParameter):
            ts._resolve_grouping_inputs([-0.63, 7.0], [3.0, 3.8, 5.0], None, [], ["-0.63,7", "-0.5,7"])


if __name__ == "__main__":
    unittest.main()
