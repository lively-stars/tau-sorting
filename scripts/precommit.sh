#!/usr/bin/env bash
# scripts/precommit.sh
#
# Manual pre-commit runner for this repo.
#
# Why this exists: this project uses Jujutsu (jj) without a colocated git
# checkout, so the `pre-commit` framework (which installs into .git/hooks/)
# has no place to attach. Run this script yourself before `jj commit` /
# `jj describe`, or wire it into a jj alias / shell prompt as you like.
#
# What it does:
#   1. ruff format       — auto-format Python sources
#   2. ruff check --fix  — lint and apply safe fixes
#   3. quick sanity checks (trailing whitespace, missing final newline) on
#      tracked-ish Python/YAML/TOML files; skips large/binary/data files.
#
# Usage:
#   ./scripts/precommit.sh           # format + fix + checks on the whole repo
#   ./scripts/precommit.sh --check   # do NOT modify files; just report
#
# Exit code is non-zero if anything would change (in --check mode) or if
# ruff check finds unfixable problems.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="fix"
if [[ "${1:-}" == "--check" ]]; then
    MODE="check"
fi

fail=0

echo "==> ruff format"
if [[ "$MODE" == "check" ]]; then
    uv run ruff format --check . || fail=1
else
    uv run ruff format .
fi

echo "==> ruff check"
if [[ "$MODE" == "check" ]]; then
    uv run ruff check . || fail=1
else
    uv run ruff check --fix . || fail=1
fi

# Lightweight equivalents of pre-commit-hooks: trailing whitespace + EOF newline.
# Limited to small text files we actually edit.
echo "==> trailing-whitespace / end-of-file checks"
mapfile -t files < <(
    find . \
        -type d \( -name .venv -o -name .jj -o -name .git -o -name __pycache__ \
                   -o -name .ruff_cache -o -name diff_binning -o -name .mplconfig \) -prune -o \
        -type f \( -name '*.py' -o -name '*.toml' -o -name '*.yaml' -o -name '*.yml' \
                   -o -name '*.md' -o -name '*.sh' \) -print
)

ws_bad=()
eof_bad=()
for f in "${files[@]}"; do
    # skip files >500KB
    sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    [[ "$sz" -gt 512000 ]] && continue

    if grep -lI '[ 	]$' "$f" >/dev/null 2>&1; then
        ws_bad+=("$f")
    fi
    # final-byte newline check
    if [[ -s "$f" ]] && [[ "$(tail -c1 "$f" | wc -l | tr -d ' ')" -eq 0 ]]; then
        eof_bad+=("$f")
    fi
done

if [[ "$MODE" == "check" ]]; then
    if (( ${#ws_bad[@]} )); then
        echo "trailing whitespace in:"; printf '  %s\n' "${ws_bad[@]}"; fail=1
    fi
    if (( ${#eof_bad[@]} )); then
        echo "missing final newline in:"; printf '  %s\n' "${eof_bad[@]}"; fail=1
    fi
else
    for f in "${ws_bad[@]}";  do sed -i '' -E 's/[[:space:]]+$//' "$f"; done 2>/dev/null \
        || for f in "${ws_bad[@]}"; do sed -i -E 's/[[:space:]]+$//' "$f"; done
    for f in "${eof_bad[@]}"; do printf '\n' >> "$f"; done
fi

if [[ "$fail" -ne 0 ]]; then
    echo
    echo "pre-commit checks reported issues."
    exit 1
fi

echo "pre-commit checks OK."
