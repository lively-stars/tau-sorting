"""Unit tests for polygon (non-rectangular) tau/lambda bins.

Data-free except TestPolygonParity (ODF-gated): parse/validate, slab
decomposition, assignment, group specs, layer-index seam, plus the
single-span parity that proves the generalized gather changed nothing.
"""

import json
import unittest

import numpy as np

import qrad_core as qc
from tausort import (
    assign_polygons,
    build_group_specs_polygons,
    build_kappa_dat_filename,
    decompose_rectilinear_polygon,
    layer_indices_for_span,
    parse_bins_json,
)


def _subbins(points):
    """Build (tau_rosseland, wavelength[cm]) arrays whose (x=log10 lambda[A],
    y=-log10 tau) equal the given (x, y) points (inverse of assign maps)."""
    x = np.array([p[0] for p in points], dtype=np.float64)
    y = np.array([p[1] for p in points], dtype=np.float64)
    return 10.0 ** (-y), 10.0**x / 1e8


def _spec(*bins):
    """Build a spec JSON string from per-bin (tau, lam) vertex lists."""
    return json.dumps({"bins": [{"vertices": [{"tau": t, "lam": x} for t, x in v]} for v in bins]})


_L_RECT = [(0.0, 3.0), (0.0, 4.0), (1.0, 4.0), (1.0, 5.0), (2.0, 5.0), (2.0, 3.0)]
_L_RECTS = [(0.0, 1.0, 3.0, 4.0), (1.0, 2.0, 3.0, 5.0)]


class TestParseAndValidate(unittest.TestCase):
    def test_valid_l_shape(self):
        bins = parse_bins_json(_spec(_L_RECT))
        self.assertEqual(len(bins), 1)
        self.assertEqual(bins[0]["vertices"].shape, (6, 2))
        self.assertEqual(len(bins[0]["rects"]), 2)

    def test_rejects_non_rectilinear_edge(self):
        bad = [(0.0, 3.0), (1.0, 4.0), (1.0, 5.0), (0.0, 5.0)]
        with self.assertRaisesRegex(ValueError, "bin 0.*non-rectilinear"):
            parse_bins_json(_spec(bad))

    def test_rejects_zero_length_edge(self):
        bad = [(0.0, 3.0), (0.0, 3.0), (1.0, 3.0), (1.0, 5.0), (0.0, 5.0)]
        with self.assertRaisesRegex(ValueError, "bin 0.*zero-length"):
            parse_bins_json(_spec(bad))

    def test_rejects_too_few_vertices(self):
        with self.assertRaisesRegex(ValueError, "bin 1.*>= 4"):
            parse_bins_json(_spec(_L_RECT, [(0.0, 3.0), (0.0, 4.0), (1.0, 4.0)]))

    def test_rejects_bowtie(self):
        # self-crossing boundary: the sweep's slab area (3.5) mismatches |shoelace| (5.5).
        bow = [(0.0, 3.0), (0.0, 5.0), (2.0, 5.0), (2.0, 4.0), (0.5, 4.0), (0.5, 6.0), (1.5, 6.0), (1.5, 3.0)]
        with self.assertRaisesRegex(ValueError, "bin 0.*self-intersecting or degenerate"):
            parse_bins_json(_spec(bow))

    def test_rejects_overlapping_bins(self):
        a = [(0.0, 3.0), (0.0, 5.0), (2.0, 5.0), (2.0, 3.0)]
        b = [(1.0, 3.0), (1.0, 5.0), (3.0, 5.0), (3.0, 3.0)]
        with self.assertRaisesRegex(ValueError, "bins 0 and 1 overlap"):
            parse_bins_json(_spec(a, b))

    def test_rejects_bad_vertex_keys(self):
        spec = json.dumps({"bins": [{"vertices": [{"tau": 0.0, "lam": 3.0, "extra": 1.0}] * 4}]})
        with self.assertRaisesRegex(ValueError, "bin 0.*exactly keys"):
            parse_bins_json(spec)

    def test_rejects_non_finite(self):
        bad = [(0.0, 3.0), (0.0, float("inf")), (1.0, 5.0), (1.0, 3.0)]
        with self.assertRaisesRegex(ValueError, "bin 0.*non-finite"):
            parse_bins_json(_spec(bad))


class TestDecompose(unittest.TestCase):
    def test_l_shape_two_rects(self):
        rects = decompose_rectilinear_polygon(np.asarray(_L_RECT, dtype=np.float64))
        self.assertEqual(len(rects), 2)
        np.testing.assert_allclose(rects, _L_RECTS)

    def test_staircase_three_rects(self):
        stair = [(0.0, 3.0), (0.0, 4.0), (1.0, 4.0), (1.0, 4.5), (2.0, 4.5), (2.0, 5.0), (3.0, 5.0), (3.0, 3.0)]
        rects = decompose_rectilinear_polygon(np.asarray(stair, dtype=np.float64))
        np.testing.assert_allclose(rects, [(0.0, 1.0, 3.0, 4.0), (1.0, 2.0, 3.0, 4.5), (2.0, 3.0, 3.0, 5.0)])

    def test_rectangle_one_rect(self):
        rects = decompose_rectilinear_polygon(np.asarray([(0.0, 3.0), (0.0, 5.0), (2.0, 5.0), (2.0, 3.0)]))
        np.testing.assert_allclose(rects, [(0.0, 2.0, 3.0, 5.0)])

    def test_collinear_vertices_tolerated(self):
        split = [(0.0, 3.0), (0.0, 4.0), (0.0, 5.0), (2.0, 5.0), (2.0, 3.0)]
        rects = decompose_rectilinear_polygon(np.asarray(split, dtype=np.float64))
        np.testing.assert_allclose(rects, [(0.0, 2.0, 3.0, 5.0)])


class TestAssignPolygons(unittest.TestCase):
    def test_inside_each_slab(self):
        bins = parse_bins_json(_spec(_L_RECT))
        tau, wl = _subbins([(3.5, 0.5), (4.5, 1.5)])
        idx, span = assign_polygons(tau, wl, bins)
        self.assertEqual(idx.tolist(), [0, 0])
        np.testing.assert_allclose(span, [[0.0, 1.0], [1.0, 2.0]])

    def test_shared_bin_edge_goes_hi_side(self):
        a = [(0.0, 3.0), (0.0, 4.0), (1.0, 4.0), (1.0, 3.0)]
        b = [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0)]
        bins = parse_bins_json(_spec(a, b))
        tau, wl = _subbins([(3.5, 1.0)])  # exactly on the shared tau edge
        idx, span = assign_polygons(tau, wl, bins)
        self.assertEqual(idx.tolist(), [1])
        np.testing.assert_allclose(span, [[1.0, 2.0]])

    def test_outside_is_minus_one(self):
        bins = parse_bins_json(_spec(_L_RECT))
        tau, wl = _subbins([(2.5, 1.0), (4.5, 0.5)])
        idx, span = assign_polygons(tau, wl, bins)
        self.assertEqual(idx.tolist(), [-1, -1])
        np.testing.assert_allclose(span, [[0.0, 0.0], [0.0, 0.0]])

    def test_empty_bin_keeps_id(self):
        a = [(0.0, 3.0), (0.0, 4.0), (1.0, 4.0), (1.0, 3.0)]
        far = [(5.0, 6.0), (5.0, 7.0), (6.0, 7.0), (6.0, 6.0)]
        b = [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0)]
        bins = parse_bins_json(_spec(a, far, b))
        tau, wl = _subbins([(3.5, 0.5), (3.5, 1.5)])
        idx, _ = assign_polygons(tau, wl, bins)
        self.assertEqual(idx.tolist(), [0, 2])


class TestGroupSpecsPolygons(unittest.TestCase):
    def test_bbox_and_round_trip(self):
        bins = parse_bins_json(_spec(_L_RECT, [(3.0, 3.0), (3.0, 5.0), (4.0, 5.0), (4.0, 3.0)]))
        gte, gle, concat, counts = build_group_specs_polygons(bins, None)
        np.testing.assert_allclose(gte, [[0.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(gle, [[3.0, 5.0], [3.0, 5.0]])
        self.assertEqual(counts.tolist(), [6, 4])
        self.assertEqual(concat.shape, (10, 2))
        np.testing.assert_allclose(concat[:6], bins[0]["vertices"])
        np.testing.assert_allclose(concat[6:], bins[1]["vertices"])

    def test_clamp_lo_diverges_from_membership(self):
        # raw membership keeps a sub-bin below the clamped descriptor lo —
        # the same raw/clamped split as the tree path.
        bins = parse_bins_json(_spec([(0.0, 3.0), (0.0, 5.0), (2.0, 5.0), (2.0, 3.0)]))
        tau, wl = _subbins([(4.0, 0.25)])
        idx, _ = assign_polygons(tau, wl, bins)
        self.assertEqual(idx.tolist(), [0])
        gte, _, _, _ = build_group_specs_polygons(bins, 0.5)
        self.assertEqual(float(gte[0, 0]), 0.5)
        self.assertFalse(gte[0, 0] <= 0.25 < gte[0, 1])

    def test_fully_below_clamp_raises(self):
        bins = parse_bins_json(_spec([(0.0, 3.0), (0.0, 5.0), (1.0, 5.0), (1.0, 3.0)]))
        with self.assertRaisesRegex(ValueError, "bin 0 lies entirely below the atmosphere clamp"):
            build_group_specs_polygons(bins, 2.0)


class TestLayerIndicesForSpan(unittest.TestCase):
    def test_span_maps_to_searchsorted_pair(self):
        tau_full = np.array([0.0, 0.01, 0.1, 1.0, 10.0, 100.0])
        # (y_lo, y_hi) -> searchsorted(10**-y_hi), searchsorted(10**-y_lo)
        jt, jb = layer_indices_for_span(0.0, 1.0, tau_full)  # tau 0.1 .. 1.0
        self.assertEqual((jt, jb), (2, 3))
        jt, jb = layer_indices_for_span(-1.0, 3.0, tau_full)  # straddles the profile
        self.assertEqual((jt, jb), (1, 4))


class TestPolygonFilename(unittest.TestCase):
    def _bins(self):
        a = [(0.0, 3.0), (0.0, 4.0), (1.0, 4.0), (1.0, 3.0)]
        b = [(1.0, 3.0), (1.0, 4.0), (2.0, 4.0), (2.0, 3.0)]
        c = [(0.0, 4.0), (0.0, 5.0), (2.0, 5.0), (2.0, 4.0)]
        return parse_bins_json(_spec(a, b, c))

    def test_polygon_filename_shape_stable(self):
        bins = self._bins()
        fn1 = build_kappa_dat_filename(nbands=9, n_splits=3, bins=bins)
        fn2 = build_kappa_dat_filename(nbands=9, n_splits=3, bins=bins)
        self.assertEqual(fn1, fn2)
        self.assertRegex(fn1, r"^kappa_9band_poly3_sp3_[0-9a-f]{8}\.dat$")

    def test_polygon_filename_moves_with_cut(self):
        bins = self._bins()
        fn1 = build_kappa_dat_filename(nbands=9, n_splits=3, bins=bins)
        moved = parse_bins_json(
            _spec(
                [(0.0, 3.0), (0.0, 4.1), (1.0, 4.1), (1.0, 3.0)],
                [(1.0, 3.0), (1.0, 4.1), (2.0, 4.1), (2.0, 3.0)],
                [(0.0, 4.1), (0.0, 5.0), (2.0, 5.0), (2.0, 4.1)],
            )
        )
        fn2 = build_kappa_dat_filename(nbands=9, n_splits=3, bins=moved)
        self.assertNotEqual(fn1, fn2)


class TestPolygonParity(unittest.TestCase):
    @staticmethod
    def _data_ready():
        from pathlib import Path

        repo = Path(__file__).resolve().parent
        odf_ok = (repo / "ODF_format.npy").exists() or (repo / "ODF_nc_format.nc").exists()
        return odf_ok and (repo / "continuumabs.dat").exists() and (repo / "models" / "G2_1D.dat").exists()

    def test_polygons_reproduce_per_tau_lambda_dat(self):
        # ODF-gated: rectangles-as-polygons must give a byte-identical .dat to
        # the lambda_edges_per_tau path (single-span parity for the gather).
        if not self._data_ready():
            self.skipTest("ODF / continuum / models/G2_1D.dat not present")
        import tempfile
        from pathlib import Path

        from kappa_band_reader import read_kappa_4_band_comparison

        tau = [-0.63, 2.885, 7.0]
        lpt = [[3.0, 3.8, 5.0], [3.0, 5.0]]
        bins = parse_bins_json(
            _spec(
                [(-0.63, 3.0), (-0.63, 3.8), (2.885, 3.8), (2.885, 3.0)],
                [(-0.63, 3.8), (-0.63, 5.0), (2.885, 5.0), (2.885, 3.8)],
                [(2.885, 3.0), (2.885, 5.0), (7.0, 5.0), (7.0, 3.0)],
            )
        )
        model = "G2_1D.dat"
        res_lpt = qc.score_binning(tau, [3.0, 5.0], None, model=model, lambda_edges_per_tau=lpt)
        res_poly = qc.score_binning(tau, [3.0, 5.0], None, model=model, bins=bins)
        self.assertEqual(res_lpt["rms"], res_poly["rms"])
        self.assertTrue(np.array_equal(res_lpt["band_index"], res_poly["band_index"]))
        with tempfile.TemporaryDirectory() as td:
            p_lpt = Path(td) / "lpt.dat"
            p_poly = Path(td) / "poly.dat"
            qc.save_kappa_dat(tau, [3.0, 5.0], None, model=model, lambda_edges_per_tau=lpt, path=p_lpt)
            qc.save_kappa_dat(tau, [3.0, 5.0], None, model=model, bins=bins, path=p_poly)
            self.assertEqual(p_lpt.read_bytes(), p_poly.read_bytes())
            back_lpt = read_kappa_4_band_comparison(p_lpt)
            back_poly = read_kappa_4_band_comparison(p_poly)
            self.assertTrue(np.array_equal(back_lpt.kap_mean, back_poly.kap_mean))


if __name__ == "__main__":
    unittest.main()
