"""Unit tests for build_split_band_index (split-resolved band index).

build_split_band_index subdivides each tau-group into n_splits opacity segments
(low/mid/high) and assigns each member sub-bin the combined band k*n_splits+seg.
These tests check the core invariants without needing the large ODF data files.

The grouping fixtures (which (lambda, tau) group each sub-bin belongs to) are now
built through the guillotine tree — `qrad_optimize.tree_from_lpt` -> `assign_tree`
-> `build_group_specs_tree` — the single grouping IR. `build_split_band_index`
itself only consumes band_index + descriptors, so it is unchanged.
"""

import unittest

import numpy as np

import qrad_optimize as qo
from tausort import (
    assign_tree,
    build_group_specs_tree,
    build_split_band_index,
    parse_split_lambda,
)


def _make_group(member_indices, n, *, lo=-4.0, hi=1.0):
    """A non-empty sorted_per_bin entry with an increasing bot-sorted curve.

    sort_idx_bot is identity, so members are already in ascending-opacity order;
    sorted_weighted_kappa_bot is a smooth increasing positive curve that
    analyze_group can segment into low/mid/high. `lo`/`hi` set the log10 dynamic
    range (max/min = 10**(hi-lo)); default 1e5x.
    """
    member_indices = np.asarray(member_indices, dtype=np.int64)
    assert member_indices.size == n
    return {
        "member_indices": member_indices,
        "sort_idx_bot": np.arange(n, dtype=np.int64),
        "sorted_weighted_kappa_bot": np.logspace(lo, hi, n),
    }


class TestBuildSplitBandIndex(unittest.TestCase):
    def test_partition_and_factorization(self):
        n_subbin_points = 1000
        n_splits = 3
        # Two well-populated, disjoint groups.
        g0 = np.arange(0, 200)
        g1 = np.arange(400, 700)
        sorted_per_bin = {0: _make_group(g0, g0.size), 1: _make_group(g1, g1.size)}

        idx = build_split_band_index(sorted_per_bin, n_subbin_points, n_groups=2, n_splits=n_splits)

        self.assertEqual(idx.shape, (n_subbin_points,))
        self.assertEqual(idx.dtype, np.int32)

        for k, members in ((0, g0), (1, g1)):
            assigned = np.flatnonzero(idx // n_splits == k)
            # Union of the group's splits == original group members (clean partition).
            np.testing.assert_array_equal(np.sort(assigned), np.sort(members))
            bands = idx[members]
            # Every member maps to this group...
            self.assertTrue(np.all(bands // n_splits == k))
            # ...and to a valid split id, with all three splits actually used.
            segs = bands % n_splits
            self.assertTrue(np.all((segs >= 0) & (segs < n_splits)))
            self.assertEqual(set(segs.tolist()), {0, 1, 2})

        # Sub-bins outside any group stay unassigned.
        untouched = np.setdiff1d(np.arange(n_subbin_points), np.concatenate([g0, g1]))
        self.assertTrue(np.all(idx[untouched] == -1))

    def test_small_group_all_to_low(self):
        g = np.arange(10, 18)  # n = 8 <= 10
        idx = build_split_band_index({0: _make_group(g, g.size)}, 100, n_groups=1, n_splits=3)
        # Whole group collapses to split 0 (low) -> band 0.
        self.assertTrue(np.all(idx[g] == 0))
        self.assertEqual(np.flatnonzero(idx >= 0).tolist(), g.tolist())

    def test_empty_and_missing_groups(self):
        g1 = np.arange(50, 250)
        sorted_per_bin = {
            0: {"empty": True, "members": 0},  # explicitly empty
            1: _make_group(g1, g1.size),
            # group 2 missing from dict entirely
        }
        idx = build_split_band_index(sorted_per_bin, 500, n_groups=3, n_splits=3)
        # Only group 1's bands (3,4,5) appear; groups 0 and 2 contribute nothing.
        assigned_bands = set(idx[idx >= 0].tolist())
        self.assertTrue(assigned_bands.issubset({3, 4, 5}))
        np.testing.assert_array_equal(np.sort(np.flatnonzero(idx >= 0)), g1)

    def test_combined_index_within_range(self):
        g0 = np.arange(0, 150)
        g1 = np.arange(150, 320)
        idx = build_split_band_index(
            {0: _make_group(g0, g0.size), 1: _make_group(g1, g1.size)},
            400,
            n_groups=2,
            n_splits=3,
        )
        # Must stay < n_groups*n_splits so calculate_tau_bin_opacities accepts it.
        self.assertTrue(np.all(idx < 2 * 3))
        self.assertTrue(np.all(idx >= -1))

    def test_min_opacity_delta_collapses_narrow_group(self):
        # A group with only a 10x opacity range (max/min = 10) collapses to a single
        # band (split 0) when min_opacity_delta=20 demands a wider dynamic range.
        g = np.arange(0, 200)
        sorted_per_bin = {0: _make_group(g, g.size, lo=-1.0, hi=0.0)}  # 10x range
        idx = build_split_band_index(sorted_per_bin, 1000, n_groups=1, n_splits=3, min_opacity_delta=20.0)
        # Every member lands on band 0 (split 0); splits 1 and 2 are never used.
        self.assertTrue(np.all(idx[g] == 0))
        self.assertEqual(set(idx[idx >= 0].tolist()), {0})

    def test_min_opacity_delta_splits_wide_group(self):
        # The same threshold (20) leaves a wide-range group (1e5x) split into all three.
        g = np.arange(0, 200)
        sorted_per_bin = {0: _make_group(g, g.size)}  # 1e5x range
        idx = build_split_band_index(sorted_per_bin, 1000, n_groups=1, n_splits=3, min_opacity_delta=20.0)
        segs = set(idx[g].tolist())
        self.assertEqual(segs, {0, 1, 2})

    def test_min_opacity_delta_default_always_splits(self):
        # Default (1.0) preserves the original behaviour: even the 10x group is split.
        g = np.arange(0, 200)
        sorted_per_bin = {0: _make_group(g, g.size, lo=-1.0, hi=0.0)}
        idx_default = build_split_band_index(sorted_per_bin, 1000, n_groups=1, n_splits=3)
        idx_one = build_split_band_index(sorted_per_bin, 1000, n_groups=1, n_splits=3, min_opacity_delta=1.0)
        np.testing.assert_array_equal(idx_default, idx_one)
        # and the default does split into all three segments
        self.assertEqual(set(idx_default[g].tolist()), {0, 1, 2})


class TestLambdaGrouping(unittest.TestCase):
    """Uniform per-cell grouping via the guillotine tree (the single grouping IR)."""

    def test_single_cell_matches_tau_index(self):
        # With one lambda cell, the group index is exactly the tau index.
        # lambda edges [3, 5] put every wavelength in the single cell; tau edges give 3 groups.
        rng = np.linspace(1e3, 9e4, 50)  # Angstrom; log10 in (3, 5)
        wl_cm = rng * 1e-8
        # -log10(tau) spanning the edges so all three tau groups get members
        neg_logtau = np.linspace(0.0, 1.4, 50)
        tau = 10.0 ** (-neg_logtau)
        tau_edges = [-0.1, 0.5, 1.0, 1.5]
        n_tau = len(tau_edges) - 1
        tree = qo.tree_from_lpt(tau_edges, [[3.0, 5.0]] * n_tau)
        g = assign_tree(tau, wl_cm, tree["root"], tree["window_tau"], tree["window_lam"])
        # All assigned, values are the plain tau index in [0, 3).
        self.assertTrue(np.all(g >= 0))
        self.assertTrue(np.all(g < 3))
        self.assertEqual(set(g.tolist()), {0, 1, 2})

    def test_two_cells_partition_consistent(self):
        # Two lambda cells split at log10 lambda = 4 (10000 A); same 2 tau groups each.
        # The tree enumerates tau-major, so points in different cells of the same tau group
        # land in different leaves — verify via the descriptor rectangle containment.
        wl_cm = np.array([3.0e3, 5.0e4]) * 1e-8  # log10 ~3.48 (cell 0), ~4.70 (cell 1)
        tau = np.array([10.0 ** (-1.2), 10.0 ** (-1.2)])  # -log10(tau)=1.2 -> tau group 1
        tau_edges = [-0.1, 0.5, 1.5]
        tree = qo.tree_from_lpt(tau_edges, [[3.0, 4.0, 5.0]] * 2)
        g = assign_tree(tau, wl_cm, tree["root"], tree["window_tau"], tree["window_lam"])
        gte, gle = build_group_specs_tree(tree["root"], tree["window_tau"], tree["window_lam"])
        # 2 tau groups x 2 lambda cells = 4 leaves; the two points are in tau-group 1 but
        # different lambda cells -> different leaves.
        self.assertEqual(gte.shape[0], 4)
        self.assertNotEqual(g[0], g[1])
        for i in range(2):
            te, le = gte[g[i]], gle[g[i]]
            self.assertTrue(te[0] <= -np.log10(tau[i]) < te[1])
            self.assertTrue(le[0] <= np.log10(wl_cm[i] * 1e8) < le[1])

    def test_out_of_range_dropped(self):
        wl_cm = np.array([1.0e2, 3.0e3]) * 1e-8  # log10 ~2.0 (below 3 -> dropped), ~3.48 (cell 0)
        tau = np.array([10.0 ** (-0.2), 10.0 ** (-0.2)])  # -log10(tau)=0.2 -> tau group 0
        tau_edges = [-0.1, 0.5, 1.5]
        tree = qo.tree_from_lpt(tau_edges, [[3.0, 5.0]] * 2)
        g = assign_tree(tau, wl_cm, tree["root"], tree["window_tau"], tree["window_lam"])
        self.assertEqual(g[0], -1)  # outside lambda window
        self.assertEqual(g[1], 0)  # tau group 0, cell 0 -> first leaf


class TestSplitLambdaFlags(unittest.TestCase):
    """Split-flag grouping via the guillotine tree (flags -> per-tau lambda edges -> tree)."""

    @staticmethod
    def _lpt(flags, lam):
        lmin, lmax = lam[0], lam[-1]
        return [list(lam) if f else [lmin, lmax] for f in flags]

    def test_group_specs_count_and_rectangles(self):
        # 3 tau groups, 2 lambda cells, flags [True, False, True] -> 5 groups tau-major.
        tau = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        tree = qo.tree_from_lpt(tau, self._lpt([True, False, True], lam))
        gte, gle = build_group_specs_tree(tree["root"], tree["window_tau"], tree["window_lam"])
        self.assertEqual(gte.shape, (5, 2))
        # tau-major: k0 split (g0=cell0, g1=cell1), k1 single (g2=whole), k2 split (g3, g4)
        np.testing.assert_allclose(gle[0], [3.0, 4.0])
        np.testing.assert_allclose(gle[1], [4.0, 5.0])
        np.testing.assert_allclose(gle[2], [3.0, 5.0])  # unsplit spans the whole range
        np.testing.assert_allclose(gle[3], [3.0, 4.0])
        np.testing.assert_allclose(gle[4], [4.0, 5.0])
        # tau ranges follow the shared edges.
        np.testing.assert_allclose(gte[2], [0.5, 1.0])  # k1

    def test_assign_split_vs_unsplit(self):
        tau_edges = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        tree = qo.tree_from_lpt(tau_edges, self._lpt([True, False, True], lam))
        # wavelengths: cell0 (~10^3.5 A), cell1 (~10^4.5 A); tau slots k1 (unsplit), k0 (split)
        wl_cm = np.array([3162.0, 31623.0, 3162.0, 31623.0]) * 1e-8
        tau = 10.0 ** (-np.array([0.7, 0.7, 0.2, 0.2]))  # k1, k1, k0, k0
        g = assign_tree(tau, wl_cm, tree["root"], tree["window_tau"], tree["window_lam"])
        # tau-major: k0 split -> g0(cell0)/g1(cell1); k1 single -> g2; k2 split -> g3/g4.
        # unsplit slot k1 -> both lambda cells collapse to group 2; split slot k0 -> 0 / 1.
        np.testing.assert_array_equal(g, [2, 2, 0, 1])

    def test_all_true_matches_uniform_count(self):
        # All-True flags reproduce the uniform-split group count (n_tau * n_lambda).
        tau = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        tree = qo.tree_from_lpt(tau, self._lpt([True, True, True], lam))
        gte, _gle = build_group_specs_tree(tree["root"], tree["window_tau"], tree["window_lam"])
        self.assertEqual(gte.shape[0], 3 * 2)

    def test_all_false_is_one_per_tau(self):
        tau = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        tree = qo.tree_from_lpt(tau, self._lpt([False, False, False], lam))
        gte, gle = build_group_specs_tree(tree["root"], tree["window_tau"], tree["window_lam"])
        self.assertEqual(gte.shape[0], 3)
        # every group spans the full lambda range
        self.assertTrue(np.all(gle[:, 0] == 3.0) and np.all(gle[:, 1] == 5.0))

    def test_parse_split_lambda(self):
        self.assertEqual(parse_split_lambda("00111100"), [False, False, True, True, True, True, False, False])
        self.assertEqual(parse_split_lambda("true,false,1,0"), [True, False, True, False])
        self.assertEqual(parse_split_lambda("T F t f"), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
