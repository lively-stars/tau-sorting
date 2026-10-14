# Optimal binning trees

Guillotine-tree tilings (`{"window_tau", "window_lam", "root"}` JSON, loadable
via `qrad_optimize.load_tree` / `--load-tree`) behind the paper's claimed optima.
Setup throughout: staged path, `models/G2_1D.dat`, window log10(tau_Ross) in
`[-5, 3]`, smooth_window 15 (cap-5 verify and cap-6 runs; Oct-8 sweep used 7).

| file | rms | provenance |
| --- | --- | --- |
| `best_436.json` | **4.37e7** (all-time best) | Oct-12 verify, seed 1, cap 5 |
| `seed5g_3.json` | 4.87e7 | Oct-8 sweep, seed 3, cap 5 (REALLOC cliff evals 570→676) |
| `cap2_best.json` | 1.81e8 | seed 0, `--max-groups 2 --initial-tau-bins 2` |
| `cap3_best.json` | 1.24e8 | freeze study, seed 0, cap 3 |
| `cap4_best.json` | 9.98e7 | freeze study, seed 1, cap 4 |
| `cap6_best.json` | 4.41e7 | freeze study, seed 0, cap 6 (same allocation as cap 5 + idle top leaf) |

Re-score any tree with:
```bash
uv run python -c "import json, qrad_core; print(qrad_core.score_binning(None,None,None,'G2_1D.dat',binning_tree=json.load(open('trees/best_436.json')))['rms'])"
```
