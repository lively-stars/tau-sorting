"""Data-free routing tests for the webapp columns-optimizer path.

Covers that a job started with ``columns: True`` threads into
``qrad_optimize.optimize_columns`` (warm-started from the TAU_COLS editor state)
with the same budgets/gaps/metric/hooks as the general path, and that the
general path still threads into ``optimize_qrad`` untouched. No ODF/model data:
both optimizer entry points are monkeypatched.
"""

import importlib.util
import time
import unittest
from pathlib import Path
from unittest import mock

import qrad_optimize as qopt

_SPEC = importlib.util.spec_from_file_location(
    "webapp_server_under_test", Path(__file__).resolve().parent / "webapp" / "server.py"
)
server = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(server)

LAM = [3.0, 3.8, 5.0]
TAU = [-0.63, 7.0]
COLS = [[-0.63, 1.0, 7.0], [-0.63, 2.0, 7.0]]


def _base_opt(**over):
    opt = {
        "opt_tau": True,
        "opt_lambda": True,
        "opt_flags": True,
        "grow": True,
        "tau_per_lambda": [list(c) for c in COLS],
        "lambda_edges_per_tau": None,
        "columns": False,
        "tree": True,
        "binning_tree": None,
        "metric": "rms",
        "beam_width": 3,
        "max_seconds": 5.0,
        "max_evals": 10,
        "max_groups": 4,
        "window": None,
        "target_rms": None,
        "plateau_evals": 0,
        "plateau_rel": 0.005,
        "grow_tol_rel": 0.01,
        "min_gap_tau": 0.15,
        "min_gap_lam": 0.10,
        "min_opacity_delta": 1.0,
    }
    opt.update(over)
    return opt


class TestColumnsRouting(unittest.TestCase):
    def setUp(self):
        server._QOPT.update(
            running=False,
            cancel=False,
            history=[],
            result=None,
            error=None,
            t0=time.perf_counter(),
            n_evals=0,
            best=None,
            rms0=None,
            groups=0,
            model="",
            diagram=None,
        )

    def test_columns_true_routes_to_optimize_columns(self):
        seen = {}

        def fake_columns(lambda_edges, tau_per_lambda, **kw):
            seen["lambda_edges"] = list(lambda_edges)
            seen["tau_per_lambda"] = [list(c) for c in tau_per_lambda]
            seen["kw"] = kw
            # exercise the shared progress hooks the thread installs
            kw["on_progress"]("start", 3.0, 2, 0)
            kw["on_eval"](1, 2.0, {"rms": 2.0, "n_groups": 2})
            return {"columns": True, "binning_tree": {"root": {}}, "rms": 2.0, "rms0": 3.0}

        with mock.patch.object(qopt, "optimize_columns", side_effect=fake_columns) as m_col:
            with mock.patch.object(qopt, "optimize_qrad") as m_gen:
                server._run_qrad_opt(list(TAU), list(LAM), [False], "G2_1D.dat", _base_opt(columns=True))
        m_col.assert_called_once()
        m_gen.assert_not_called()
        self.assertEqual(seen["lambda_edges"], LAM)
        self.assertEqual(seen["tau_per_lambda"], COLS)  # warm-started from the TAU_COLS editor state
        self.assertTrue(seen["kw"]["grow"])
        self.assertEqual(seen["kw"]["model"], "G2_1D.dat")
        for key in (
            "metric",
            "max_seconds",
            "max_evals",
            "max_groups",
            "min_gap_tau",
            "min_gap_lam",
            "grow_tol_rel",
            "window",
            "target_rms",
            "plateau_evals",
            "plateau_rel",
            "min_opacity_delta",
            "on_eval",
            "on_progress",
            "should_stop",
        ):
            self.assertIn(key, seen["kw"], f"columns call missing shared knob/hook {key!r}")
        # status shape: result carried through, thread marked done, hooks streamed progress
        self.assertTrue(server._QOPT["result"]["columns"])
        self.assertFalse(server._QOPT["running"])
        self.assertIsNone(server._QOPT["error"])
        self.assertEqual(server._QOPT["rms0"], 3.0)
        self.assertEqual(server._QOPT["best"], 2.0)

    def test_columns_false_keeps_general_path(self):
        with mock.patch.object(qopt, "optimize_qrad", return_value={"columns": False}) as m_gen:
            with mock.patch.object(qopt, "optimize_columns") as m_col:
                server._run_qrad_opt(list(TAU), list(LAM), [False], "G2_1D.dat", _base_opt(columns=False))
        m_gen.assert_called_once()
        m_col.assert_not_called()
        self.assertFalse(server._QOPT["running"])
        self.assertIsNone(server._QOPT["error"])


if __name__ == "__main__":
    unittest.main()
