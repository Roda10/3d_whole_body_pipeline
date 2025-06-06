#!/bin/bash
set -e

UBODY_DIR="data/UBody"

# echo "📁 Creating download directory: $UBODY_DIR"
# mkdir -p "$UBODY_DIR"

# # Step 1: Download files
# echo "⬇️  Downloading videos.zip..."
# rclone copy "gdrive:Ubody/videos.zip" "$UBODY_DIR" --drive-shared-with-me --progress

# echo "⬇️  Downloading annotations.zip..."
# rclone copy "gdrive:Ubody/annotations.zip" "$UBODY_DIR" --drive-shared-with-me --progress

# echo "⬇️  Downloading splits.zip..."
# rclone copy "gdrive:Ubody/splits.zip" "$UBODY_DIR" --drive-shared-with-me --progress

# Step 2: Unzip files
echo "📦 Unzipping videos.zip..."
unzip -o "$UBODY_DIR/videos.zip" -d "$UBODY_DIR/videos"

echo "📦 Unzipping annotations.zip..."
unzip -o "$UBODY_DIR/annotations.zip" -d "$UBODY_DIR/annotations"

echo "📦 Unzipping splits.zip..."
unzip -o "$UBODY_DIR/splits.zip" -d "$UBODY_DIR/splits"

echo "✅ Download and extraction complete!"
