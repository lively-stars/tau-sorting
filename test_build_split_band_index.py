"""Unit tests for build_split_band_index (split-resolved band index).

build_split_band_index subdivides each tau-group into n_splits opacity segments
(low/mid/high) and assigns each member sub-bin the combined band k*n_splits+seg.
These tests check the core invariants without needing the large ODF data files.
"""

import unittest

import numpy as np

from tausort import (
    assign_split_lambda,
    assign_tau_to_bin,
    build_group_index_maps,
    build_group_specs_split_lambda,
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
    def test_offsets_and_counts(self):
        # cell 0 has 3 tau groups, cell 1 has 2 -> offsets [0, 3], n_groups 5
        per_cell = [[-0.6, 0.0, 1.0, 7.0], [-0.6, 2.0, 7.0]]
        offsets, n_groups, g2cell, g2tau = build_group_index_maps(per_cell)
        self.assertEqual(offsets, [0, 3])
        self.assertEqual(n_groups, 5)
        np.testing.assert_array_equal(g2cell, [0, 0, 0, 1, 1])
        np.testing.assert_array_equal(g2tau, [0, 1, 2, 0, 1])

    def test_single_cell_matches_tau_only(self):
        # With one lambda cell, the group index is exactly the tau index.
        # lambda edges [3, 5] put every wavelength in cell 0; tau edges give 3 groups.
        rng = np.linspace(1e3, 9e4, 50)  # Angstrom; log10 in (3, 5)
        wl_cm = rng * 1e-8
        # -log10(tau) spanning the edges so all three tau groups get members
        neg_logtau = np.linspace(0.0, 1.4, 50)
        tau = 10.0 ** (-neg_logtau)
        tau_edges = [-0.1, 0.5, 1.0, 1.5]
        g = assign_tau_to_bin(tau, wl_cm, tau_edges_per_lambda=[tau_edges], lambda_bin_edges=[3.0, 5.0])
        # All assigned, values are the plain tau index in [0, 3).
        self.assertTrue(np.all(g >= 0))
        self.assertTrue(np.all(g < 3))
        self.assertEqual(set(g.tolist()), {0, 1, 2})

    def test_two_cells_flatten_with_offset(self):
        # Two lambda cells split at log10 lambda = 4 (10000 A); each has 2 tau groups.
        # Build wavelengths in both cells and a tau that lands in tau-group 1 of each.
        wl_cm = np.array([3.0e3, 5.0e4]) * 1e-8  # log10 ~3.48 (cell 0), ~4.70 (cell 1)
        tau = np.array([10.0 ** (-1.2), 10.0 ** (-1.2)])  # -log10(tau)=1.2 -> tau group 1
        per_cell = [[-0.1, 0.5, 1.5], [-0.1, 0.5, 1.5]]  # 2 tau groups each
        g = assign_tau_to_bin(tau, wl_cm, tau_edges_per_lambda=per_cell, lambda_bin_edges=[3.0, 4.0, 5.0])
        # cell 0 tau-group 1 -> g = 0*2 + 1 = 1 ; cell 1 tau-group 1 -> g = 2 + 1 = 3
        np.testing.assert_array_equal(g, [1, 3])

    def test_out_of_range_dropped(self):
        wl_cm = np.array([1.0e2, 3.0e3]) * 1e-8  # log10 ~2.0 (below 3 -> dropped), ~3.48 (cell 0)
        tau = np.array([10.0 ** (-0.2), 10.0 ** (-0.2)])  # -log10(tau)=0.2 -> tau group 0
        per_cell = [[-0.1, 0.5, 1.5]]
        g = assign_tau_to_bin(tau, wl_cm, tau_edges_per_lambda=per_cell, lambda_bin_edges=[3.0, 5.0])
        self.assertEqual(g[0], -1)  # outside lambda range
        self.assertEqual(g[1], 0)  # cell 0, tau group 0


class TestSplitLambdaFlags(unittest.TestCase):
    def test_group_specs_count_and_order(self):
        # 3 tau groups, 2 lambda cells, flags [True, False, True] -> 5 groups slot-major.
        tau = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        gt, gl, s2cg, s2sg = build_group_specs_split_lambda(tau, lam, [True, False, True])
        self.assertEqual(gt.shape, (5, 2))
        self.assertEqual(gl.shape, (5, 2))
        # slot-major: k0 split (g0,g1), k1 single (g2), k2 split (g3,g4)
        np.testing.assert_array_equal(s2cg, [[0, 1], [-1, -1], [3, 4]])
        np.testing.assert_array_equal(s2sg, [-1, 2, -1])
        # unsplit group spans the whole lambda range; split groups are confined to a cell.
        np.testing.assert_allclose(gl[2], [3.0, 5.0])
        np.testing.assert_allclose(gl[0], [3.0, 4.0])
        np.testing.assert_allclose(gl[1], [4.0, 5.0])
        # tau ranges follow the shared edges.
        np.testing.assert_allclose(gt[2], [0.5, 1.0])

    def test_assign_split_vs_unsplit(self):
        tau_edges = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        _gt, _gl, s2cg, s2sg = build_group_specs_split_lambda(tau_edges, lam, [True, False, True])
        # wavelengths: cell0 (~10^3.5 A), cell1 (~10^4.5 A); tau slots k1 (unsplit), k0 (split)
        wl_cm = np.array([3162.0, 31623.0, 3162.0, 31623.0]) * 1e-8
        tau = 10.0 ** (-np.array([0.7, 0.7, 0.2, 0.2]))  # k1, k1, k0, k0
        g = assign_split_lambda(tau, wl_cm, tau_edges, lam, s2cg, s2sg)
        # unsplit slot k1 -> both lambda cells collapse to group 2; split slot k0 -> 0 / 1
        np.testing.assert_array_equal(g, [2, 2, 0, 1])

    def test_all_true_matches_uniform_count(self):
        # All-True flags reproduce the uniform-split group count (n_tau * n_lambda).
        tau = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        gt, _gl, _s2cg, _s2sg = build_group_specs_split_lambda(tau, lam, [True, True, True])
        self.assertEqual(gt.shape[0], 3 * 2)

    def test_all_false_is_one_per_tau(self):
        tau = [-0.1, 0.5, 1.0, 1.5]
        lam = [3.0, 4.0, 5.0]
        gt, gl, _s2cg, s2sg = build_group_specs_split_lambda(tau, lam, [False, False, False])
        self.assertEqual(gt.shape[0], 3)
        # every group spans the full lambda range
        self.assertTrue(np.all(gl[:, 0] == 3.0) and np.all(gl[:, 1] == 5.0))
        np.testing.assert_array_equal(s2sg, [0, 1, 2])

    def test_parse_split_lambda(self):
        self.assertEqual(parse_split_lambda("00111100"), [False, False, True, True, True, True, False, False])
        self.assertEqual(parse_split_lambda("true,false,1,0"), [True, False, True, False])
        self.assertEqual(parse_split_lambda("T F t f"), [True, False, True, False])


if __name__ == "__main__":
    unittest.main()
