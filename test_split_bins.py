"""Unit tests for the box + ordered-split-list binning (click-to-split UI).

Data-free: tree_from_splits geometry + error paths, and the grouping-update
contract the frontend relies on — appending a split grows the leaf/group count
by exactly one and the assign_tree membership matches the new descriptor.
"""

import unittest

import numpy as np

import qrad_optimize as qo
import tausort as ts
from tausort import assign_tree, build_group_specs_tree


class TestTreeFromSplits(unittest.TestCase):
    def test_empty_box_is_one_group(self):
        tree = qo.tree_from_splits([-0.63, 7.0], [3.0, 5.0], [])
        self.assertEqual(tree["window_tau"], [-0.63, 7.0])
        self.assertEqual(tree["window_lam"], [3.0, 5.0])
        rects = list(ts._iter_leaf_rects(tree["root"], (-0.63, 7.0, 3.0, 5.0)))
        self.assertEqual(rects, [(-0.63, 7.0, 3.0, 5.0)])

    def test_user_example_two_tau_then_one_lam(self):
        # Box tau [-1,7] x lam [3,5]; tau cuts at 0 and 2 (full-height), then a
        # lambda cut at (tau=1.0, lam=3.5) splits only the tau-[0,2] bin.
        tree = qo.tree_from_splits(
            [-1.0, 7.0],
            [3.0, 5.0],
            [
                {"axis": "tau", "tau": 0.0, "lam": 4.0},
                {"axis": "tau", "tau": 2.0, "lam": 4.0},
                {"axis": "lam", "tau": 1.0, "lam": 3.5},
            ],
        )
        rects = [tuple(round(v, 6) for v in r) for r in ts._iter_leaf_rects(tree["root"], (-1.0, 7.0, 3.0, 5.0))]
        self.assertEqual(
            rects,
            [
                (-1.0, 0.0, 3.0, 5.0),
                (0.0, 2.0, 3.0, 3.5),
                (0.0, 2.0, 3.5, 5.0),
                (2.0, 7.0, 3.0, 5.0),
            ],
        )

    def test_order_matters_lam_first(self):
        # Same cuts, lambda first: the later tau cut only spans one lambda side.
        tree = qo.tree_from_splits(
            [-1.0, 7.0],
            [3.0, 5.0],
            [
                {"axis": "lam", "tau": 0.0, "lam": 4.0},
                {"axis": "tau", "tau": 0.0, "lam": 4.5},
            ],
        )
        rects = sorted(tuple(round(v, 6) for v in r) for r in ts._iter_leaf_rects(tree["root"], (-1.0, 7.0, 3.0, 5.0)))
        self.assertEqual(
            rects,
            [
                (-1.0, 0.0, 4.0, 5.0),
                (-1.0, 7.0, 3.0, 4.0),
                (0.0, 7.0, 4.0, 5.0),
            ],
        )

    def test_rejects_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "tau window must be increasing"):
            qo.tree_from_splits([1.0, 1.0], [3.0, 5.0], [])
        with self.assertRaisesRegex(ValueError, "lambda window must be increasing"):
            qo.tree_from_splits([-1.0, 7.0], [3.0, 3.0], [])
        with self.assertRaisesRegex(ValueError, "axis must be 'tau' or 'lam'"):
            qo.tree_from_splits([-1.0, 7.0], [3.0, 5.0], [{"axis": "x", "tau": 0.0, "lam": 4.0}])
        with self.assertRaisesRegex(ValueError, "outside the box"):
            qo.tree_from_splits([-1.0, 7.0], [3.0, 5.0], [{"axis": "tau", "tau": 99.0, "lam": 4.0}])
        with self.assertRaisesRegex(ValueError, "min_gap"):
            qo.tree_from_splits([-1.0, 7.0], [3.0, 5.0], [{"axis": "tau", "tau": -0.99, "lam": 4.0}])


class TestSplitGroupingUpdate(unittest.TestCase):
    """Appending one split grows the grouping by exactly one leaf, and the new
    leaf rectangles tile the box without overlap (the plot-update contract:
    after a click the recompute must show n+1 group boxes)."""

    def _groups(self, splits):
        tree = qo.tree_from_splits([-0.63, 7.0], [3.0, 5.0], splits)
        tw, lw, root = tree["window_tau"], tree["window_lam"], tree["root"]
        gte, gle = build_group_specs_tree(root, [tw[0], tw[1]], lw)
        return tree, gte, gle

    def test_each_split_adds_one_group(self):
        splits = []
        _, gte, _ = self._groups(splits)
        self.assertEqual(gte.shape[0], 1)
        clicks = [
            {"axis": "tau", "tau": 2.0, "lam": 3.6},
            {"axis": "tau", "tau": 0.35, "lam": 4.0},
            {"axis": "lam", "tau": 0.8, "lam": 3.8},
        ]
        for i, c in enumerate(clicks):
            splits = [*splits, c]
            _, gte, _ = self._groups(splits)
            self.assertEqual(gte.shape[0], 2 + i, f"after split {i + 1}")

    def test_leaves_tile_the_box(self):
        _, gte, gle = self._groups(
            [
                {"axis": "tau", "tau": 0.0, "lam": 4.0},
                {"axis": "tau", "tau": 2.0, "lam": 4.0},
                {"axis": "lam", "tau": 1.0, "lam": 3.5},
            ]
        )
        area = sum((gte[g, 1] - gte[g, 0]) * (gle[g, 1] - gle[g, 0]) for g in range(gte.shape[0]))
        box = (7.0 - -0.63) * (5.0 - 3.0)
        self.assertAlmostEqual(area, box, places=9)

    def test_assign_matches_new_descriptor(self):
        tree, gte, gle = self._groups([{"axis": "tau", "tau": 2.0, "lam": 3.6}])
        tw, lw, root = tree["window_tau"], tree["window_lam"], tree["root"]
        self.assertEqual(gte.shape[0], 2)
        rng = np.random.default_rng(11)
        y = rng.uniform(tw[0], tw[1], 2000)
        x = rng.uniform(lw[0], lw[1], 2000)
        bi = assign_tree(10.0 ** (-y), 10.0**x / 1e8, root, tw, lw)
        self.assertTrue((bi >= 0).all())
        self.assertEqual(set(np.unique(bi)), {0, 1})
        for i in range(0, 2000, 97):
            te, le = gte[bi[i]], gle[bi[i]]
            self.assertTrue(te[0] <= y[i] < te[1])
            self.assertTrue(le[0] <= x[i] < le[1])


if __name__ == "__main__":
    unittest.main()
