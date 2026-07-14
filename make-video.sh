#!/usr/bin/env bash
#
# make_video.sh - Turn a folder of step_XXXX_*.png frames into an mp4.
#
# Usage:
#   ./make_video.sh -i <input_dir> -o <output.mp4> [-r fps] [-p pattern]
#
# Example:
#   ./make_video.sh -i ./frames -o out.mp4 -r 24
#
set -euo pipefail

INPUT_DIR=""
OUTPUT_FILE="output.mp4"
FPS=30
# glob pattern to find images; adjust if your extension differs
PATTERN="step_*.png"

usage() {
    echo "Usage: $0 -i <input_dir> [-o output.mp4] [-r fps] [-p glob_pattern]"
    exit 1
}

while getopts "i:o:r:p:h" opt; do
    case "$opt" in
        i) INPUT_DIR="$OPTARG" ;;
        o) OUTPUT_FILE="$OPTARG" ;;
        r) FPS="$OPTARG" ;;
        p) PATTERN="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: input directory is required (-i)."
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: '$INPUT_DIR' is not a directory."
    exit 1
fi

command -v ffmpeg >/dev/null 2>&1 || { echo "Error: ffmpeg not found in PATH."; exit 1; }

# Collect files matching the pattern
shopt -s nullglob
FILES=("$INPUT_DIR"/$PATTERN)
shopt -u nullglob

if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Error: no files matching '$PATTERN' found in '$INPUT_DIR'."
    exit 1
fi

# Extract step number from each filename (the number right after "step_")
# and sort numerically by that value. This is safer than a plain lexical
# sort, since it doesn't depend on consistent zero-padding.
TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

for f in "${FILES[@]}"; do
    base="$(basename "$f")"
    if [[ "$base" =~ step_([0-9]+) ]]; then
        step="${BASH_REMATCH[1]}"
        # strip leading zeros for numeric sort correctness (base 10 forced)
        step_num=$((10#$step))
        printf '%s\t%s\n' "$step_num" "$f" >> "$TMP_LIST"
    else
        echo "Warning: skipping '$base' (no step_NNNN found)"
    fi
done

if [[ ! -s "$TMP_LIST" ]]; then
    echo "Error: no files matched the expected step_NNNN naming pattern."
    exit 1
fi

# Sort numerically by step number
SORTED_LIST="$(sort -n -k1,1 "$TMP_LIST")"

# Build a temp directory of sequentially-numbered symlinks so ffmpeg's
# %06d pattern can consume them, regardless of gaps in the original steps.
FRAME_DIR="$(mktemp -d)"
trap 'rm -f "$TMP_LIST"; rm -rf "$FRAME_DIR"' EXIT

i=0
ext=""
while IFS=$'\t' read -r step_num filepath; do
    if [[ -z "$ext" ]]; then
        ext="${filepath##*.}"
    fi
    printf -v idx "%06d" "$i"
    ln -s "$(realpath "$filepath")" "$FRAME_DIR/frame_${idx}.${ext}"
    i=$((i + 1))
done <<< "$SORTED_LIST"

echo "Found $i frames. Encoding at ${FPS} fps -> $OUTPUT_FILE"

ffmpeg -y -framerate "$FPS" \
    -i "$FRAME_DIR/frame_%06d.${ext}" \
    -c:v libx264 -pix_fmt yuv420p \
    -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    "$OUTPUT_FILE"

echo "Done: $OUTPUT_FILE"
