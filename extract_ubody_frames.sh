#!/bin/bash

INPUT_DIR="data/UBody/videos/videos"
OUTPUT_DIR="data/UBody/frames"

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" -type f -name "*.mp4" | while IFS= read -r VIDEO; do
    # Normalize full path
    ABS_VIDEO=$(realpath "$VIDEO")

    # Compute relative path from INPUT_DIR and extract subfolder path and base filename
    REL_PATH=${ABS_VIDEO#"$INPUT_DIR"/}
    REL_DIR=$(dirname "$REL_PATH")
    BASENAME=$(basename "$REL_PATH" .mp4)

    OUT_DIR="$OUTPUT_DIR/$REL_DIR/$BASENAME"
    mkdir -p "$OUT_DIR"

    if [[ -f "$VIDEO" ]]; then
        ffmpeg -hide_banner -loglevel error -i "$VIDEO" -vf fps=1 "$OUT_DIR/frame_%04d.png"
        echo "✅ Extracted frames from $VIDEO -> $OUT_DIR"
    else
        echo "❌ Skipped (not found): $VIDEO"
    fi
done
