"""Data-free test pinning score_binning's grouping precedence.

Stubs inv_for/reference with minimal fakes and stops the pipeline at assign_tree,
recording which tree the branch selected. Precedence: binning_tree > tau_per_lambda
> lambda_edges_per_tau > flags.
"""

import unittest
from unittest import mock

import numpy as np

import qrad_core as qc


def _fake_inv():
    tau_ross = np.array([0.01, 0.1, 1.0, 10.0])
    return {
        "odf": object(),
        "cont": object(),
        "atm": object(),
        "tau_ross": tau_ross,
        "max_height_idx": 1,
        "tau_at_lam1": np.array([0.05, 0.5]),
        "wl_centers": np.array([1e-6, 2e-6]),
    }


class _Stop(Exception):
    def __init__(self, root):
        super().__init__("stop")
        self.root = root


def _run(**kw):
    seen = {}

    def _assign(tau_r, wl, root, tw, lw):
        raise _Stop(root)

    with (
        mock.patch.object(qc, "inv_for", return_value=_fake_inv()),
        mock.patch.object(qc, "reference", return_value={}),
        mock.patch.object(qc.ts, "assign_tree", side_effect=_assign),
    ):
        try:
            qc.score_binning(
                kw.pop("tau_edges", [-0.63, 7.0]),
                kw.pop("lambda_edges", [3.0, 5.0]),
                kw.pop("flags", [True]),
                model="fake",
                **kw,
            )
        except _Stop as s:
            seen["root"] = s.root
    return seen["root"]


class TestScorerPrecedence(unittest.TestCase):
    def test_tree_wins_over_all(self):
        import qrad_optimize as qo

        tree = qo.tree_from_columns([3.0, 5.0], [[-0.63, 7.0]])
        root = _run(
            binning_tree=tree,
            tau_per_lambda=[[-0.63, 1.0, 7.0]],
            lambda_edges_per_tau=[[3.0, 5.0]],
        )
        self.assertIs(root, tree["root"])

    def test_columns_win_over_lpt_and_flags(self):
        # two columns -> lambda cut at the root (a tau-root would mean the lpt branch won)
        root = _run(
            lambda_edges=[3.0, 3.8, 5.0],
            tau_per_lambda=[[-0.63, 1.0, 7.0], [-0.63, 2.0, 7.0]],
            lambda_edges_per_tau=[[3.0, 5.0]],
        )
        self.assertEqual(root.get("axis"), "lam")

    def test_lpt_wins_over_flags(self):
        # two tau groups -> tau cut at the root (a lam-root would mean the flags branch won)
        root = _run(
            tau_edges=[-0.63, 1.0, 7.0],
            flags=[True, True],
            lambda_edges_per_tau=[[3.0, 3.8, 5.0], [3.0, 5.0]],
        )
        self.assertEqual(root.get("axis"), "tau")


if __name__ == "__main__":
    unittest.main()
