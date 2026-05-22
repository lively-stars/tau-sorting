#!/usr/bin/env bash
# Regenerate the four sorted-opacity plots referenced in README.md.
#
# Each invocation of `tausort.py main` writes
# `sorted_weighted_opacity_per_tau_bin.jpg` to the project root; we rename
# it into plots/ between runs.
#
# Usage:
#   ./scripts/regenerate_readme_plots.sh

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p plots

run_variant() {
    local label="$1"
    local refine_flag="$2"
    shift 2
    local edges=("$@")

    local edge_args=()
    for e in "${edges[@]}"; do
        edge_args+=("--tau-bin-edges" "$e")
    done

    echo "==> $label"
    uv run python tausort.py main "${edge_args[@]}" "$refine_flag" > "/tmp/plot_${label}.log" 2>&1

    if [[ ! -f sorted_weighted_opacity_per_tau_bin.jpg ]]; then
        echo "  ERROR: sorted_weighted_opacity_per_tau_bin.jpg not produced for $label" >&2
        echo "  Last 30 log lines:" >&2
        tail -30 "/tmp/plot_${label}.log" >&2
        exit 1
    fi

    mv sorted_weighted_opacity_per_tau_bin.jpg "plots/sorted_${label}.jpg"
    echo "  -> plots/sorted_${label}.jpg"
}

# 4-bin edges (5 values)
edges_4bin=(-0.63 -0.1 1.5 3.8 7.0)
# 8-bin edges from a prior --optimize-high-overlap run (9 values)
edges_8bin=(-0.63 -0.3 -0.15 0.0 0.25 0.7 1.5 3.9 7.0)

run_variant "4bin_refinemid"    "--refine-mid"    "${edges_4bin[@]}"
run_variant "4bin_no_refinemid" "--no-refine-mid" "${edges_4bin[@]}"
run_variant "8bin_refinemid"    "--refine-mid"    "${edges_8bin[@]}"
run_variant "8bin_no_refinemid" "--no-refine-mid" "${edges_8bin[@]}"

echo
echo "All 4 plots regenerated under plots/."
ls -la plots/sorted_*.jpg
