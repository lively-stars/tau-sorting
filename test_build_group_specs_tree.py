"""Unit tests for the general 2D guillotine binning (build_group_specs_tree / assign_tree).

A guillotine tree recursively cuts the (-log10 tau, log10 lambda) plane along either axis;
its leaves are the (tau, lambda) groups. These tests check the core invariants — DFS-order
alignment between the descriptor and the assignment, empty-leaf id alignment, the
digitize(right=False) boundary convention, and out-of-window handling — without any ODF data.
"""

import unittest

import numpy as np

from tausort import assign_tree, build_group_specs_tree


def _subbins(points):
    """Build (tau_rosseland, wavelength[cm]) arrays whose (x=log10 lambda[A], y=-log10 tau)
    equal the given (x, y) points. Uses the exact inverse of assign_tree's coordinate maps:
    wl_cm = 10**x / 1e8 (so log10(wl*1e8)=x), tau = 10**(-y) (so -log10(tau)=y)."""
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    return 10.0 ** (-y), 10.0**x / 1e8


# tree: cut tau@1.23 -> {lo: cut lam@3.8 -> (leaf0, leaf1), hi: leaf2}
# leaves in DFS order: 0=[tau -0.63..1.23]x[lam 3..3.8], 1=[..1.23]x[3.8..5], 2=[1.23..7]x[3..5]
_TREE = {
    "axis": "tau",
    "at": 1.23,
    "lo": {"axis": "lam", "at": 3.8, "lo": {"leaf": True}, "hi": {"leaf": True}},
    "hi": {"leaf": True},
}
_WIN_TAU = [-0.63, 7.0]
_WIN_LAM = [3.0, 5.0]


class TestGuillotineBinning(unittest.TestCase):
    def test_descriptor_dfs_order(self):
        gte, gle = build_group_specs_tree(_TREE, _WIN_TAU, _WIN_LAM)
        self.assertEqual(gte.shape, (3, 2))
        np.testing.assert_allclose(gte, [[-0.63, 1.23], [-0.63, 1.23], [1.23, 7.0]])
        np.testing.assert_allclose(gle, [[3.0, 3.8], [3.8, 5.0], [3.0, 5.0]])

    def test_assign_matches_descriptor_containment(self):
        # every assigned sub-bin must land in a group whose rectangle contains its (x, y),
        # with the right=False convention (>= lo, < hi), and -1 iff outside the root rect.
        pts = [(3.4, 0.5), (4.2, 0.5), (3.4, 3.0), (4.2, 3.0), (2.5, 1.0), (3.5, 8.0)]
        tau, wl = _subbins(pts)
        g = assign_tree(tau, wl, _TREE, _WIN_TAU, _WIN_LAM)
        gte, gle = build_group_specs_tree(_TREE, _WIN_TAU, _WIN_LAM)
        expected = [0, 1, 2, 2, -1, -1]
        self.assertEqual(g.tolist(), expected)
        for i, (x, y) in enumerate(pts):
            if g[i] < 0:
                continue
            te, le = gte[g[i]], gle[g[i]]
            self.assertTrue(te[0] <= y < te[1], f"pt {i} y={y} not in tau {te}")
            self.assertTrue(le[0] <= x < le[1], f"pt {i} x={x} not in lam {le}")

    def test_empty_leaf_id_alignment(self):
        # tree: cut lam@3.8 -> {lo: leaf0, hi: cut tau@1.23 -> (leaf1, leaf2)}.
        # leaf1 (x in [3.8,5], y<1.23) is deliberately empty; a point in leaf2 must still get
        # id 2 (the counter must consume the empty leaf's id).
        tree = {
            "axis": "lam",
            "at": 3.8,
            "lo": {"leaf": True},
            "hi": {"axis": "tau", "at": 1.23, "lo": {"leaf": True}, "hi": {"leaf": True}},
        }
        pts = [(3.4, 0.5), (4.2, 3.0)]  # -> leaf0, leaf2 ; nothing in leaf1
        tau, wl = _subbins(pts)
        g = assign_tree(tau, wl, tree, _WIN_TAU, _WIN_LAM)
        self.assertEqual(g.tolist(), [0, 2])
        gte, _ = build_group_specs_tree(tree, _WIN_TAU, _WIN_LAM)
        self.assertEqual(gte.shape[0], 3)  # empty leaf still occupies a descriptor row

    def test_right_false_boundary(self):
        # a point exactly on a cut lands in the HI child (digitize right=False). Use cut
        # positions that round-trip exactly (integer log values): lam@4.0, tau@1.0.
        tree = {
            "axis": "lam",
            "at": 4.0,
            "lo": {"axis": "tau", "at": 1.0, "lo": {"leaf": True}, "hi": {"leaf": True}},
            "hi": {"leaf": True},
        }
        # p0 exactly on lam cut 4.0 -> hi child -> leaf 2 ; p1 exactly on tau cut 1.0 (x<4) -> hi -> leaf1
        pts = [(4.0, 0.5), (3.5, 1.0)]
        tau, wl = _subbins(pts)
        g = assign_tree(tau, wl, tree, _WIN_TAU, _WIN_LAM)
        self.assertEqual(g.tolist(), [2, 1])

    def test_single_leaf_tree(self):
        # a bare leaf assigns every in-window sub-bin to group 0.
        leaf = {"leaf": True}
        pts = [(3.5, 1.0), (4.5, 5.0), (6.0, 1.0)]  # last is outside lam window -> -1
        tau, wl = _subbins(pts)
        g = assign_tree(tau, wl, leaf, _WIN_TAU, _WIN_LAM)
        self.assertEqual(g.tolist(), [0, 0, -1])
        gte, gle = build_group_specs_tree(leaf, _WIN_TAU, _WIN_LAM)
        np.testing.assert_allclose(gte, [[-0.63, 7.0]])
        np.testing.assert_allclose(gle, [[3.0, 5.0]])

    def test_membership_window_diverges_from_descriptor_clamped_top(self):
        # Invariant #4: assign_tree scores membership against the RAW (un-clamped) tau window,
        # while build_group_specs_tree reports the CLAMPED top edge. When raw_top < clamped_top,
        # a sub-bin at y in (raw_top, clamped_top) is a member of group 0 (raw window) yet lies
        # BELOW the descriptor group-0 rectangle's top edge — the intended containment mismatch
        # that qrad_core.score_binning relies on (membership raw, descriptor clamped).
        raw_top, clamped_top, hi = 0.0, 0.5, 7.0  # raw_top < clamped_top
        # simple 1-split tree: cut tau@3.0 -> lo=group0 [..,3], hi=group1 [3,hi]
        tree = {"axis": "tau", "at": 3.0, "lo": {"leaf": True}, "hi": {"leaf": True}}
        y, x = 0.25, 4.0  # y in (raw_top, clamped_top), and y < 3.0 so -> group 0
        tau_r, wl = _subbins([(x, y)])

        # membership uses the RAW window -> the point is in group 0
        g_raw = assign_tree(tau_r, wl, tree, [raw_top, hi], _WIN_LAM)
        self.assertEqual(g_raw.tolist(), [0])

        # descriptor uses the CLAMPED top -> group 0's rectangle starts at clamped_top > y,
        # so the descriptor does NOT contain the point it just assigned.
        gte, _gle = build_group_specs_tree(tree, [clamped_top, hi], _WIN_LAM)
        self.assertEqual(gte[0, 0], clamped_top)
        self.assertGreater(clamped_top, y)
        self.assertFalse(gte[0, 0] <= y < gte[0, 1])  # descriptor group 0 excludes y


if __name__ == "__main__":
    unittest.main()
