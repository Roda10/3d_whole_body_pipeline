#!/bin/bash
# ======================================================
# Setup Script for 3D Whole Body Pipeline
# ======================================================

echo "🔄 Initializing and updating submodules..."

# Ensure we're in repo root
cd "$(dirname "$0")/.."

# Sync submodule URLs (useful if you changed HTTPS → SSH)
git submodule sync

# Initialize and update all submodules recursively
git submodule update --init --recursive

echo "✅ Submodules are ready!"
echo "You can now run the pipeline."
