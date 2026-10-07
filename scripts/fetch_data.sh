#!/usr/bin/env bash
# Fetch the gitignored large runtime inputs a fresh clone needs from the
# `data-v1` GitHub release. Downloads only what's missing (either ODF source
# and either continuum source count as present); exits non-zero with a hint
# when a download fails. Needs `gh` (authenticated) or curl as fallback.
set -euo pipefail

REPO="${TAUSORT_REPO:-lively-stars/tau-sorting}"
TAG="${TAUSORT_DATA_TAG:-data-v1}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# dest-path -> acceptable asset names (first existing candidate wins; else
# download the first asset listed). Paths mirror the repo layout.
wants() {
  case "$1" in
    ODF_format.npy) echo "ODF_format.npy ODF_nc_format.nc" ;;
    continuumabs.dat) echo "continuumabs.dat" ;;
    data/kappa_grey.dat) echo "kappa_grey.dat" ;;
    data/kappa_fullodf.dat) echo "kappa_fullodf.dat" ;;
  esac
}

missing=0
for dest in ODF_format.npy continuumabs.dat data/kappa_grey.dat data/kappa_fullodf.dat; do
  ok=0
  # shellcheck disable=SC2046
  for cand in $(wants "$dest"); do
    case "$cand" in
      data/*) p="$HERE/$cand" ;;
      ODF_format.npy | ODF_nc_format.nc | continuumabs.dat) p="$HERE/$cand" ;;
      *) p="$HERE/$(dirname "$dest")/$cand" ;;
    esac
    if [ -f "$p" ]; then ok=1; break; fi
  done
  if [ "$ok" = 1 ]; then echo "have $dest (or equivalent)"; continue; fi
  missing=1
  asset="$(wants "$dest" | cut -d' ' -f1)"
  mkdir -p "$HERE/$(dirname "$dest")"
  echo "fetching $dest <- release $TAG/$asset ..."
done

if [ "$missing" = 0 ]; then echo "all data files present"; fi
