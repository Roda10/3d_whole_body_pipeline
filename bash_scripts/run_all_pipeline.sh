#!/bin/bash

# Unified 3D Pose Pipeline Runner with Fusion
# ===========================================

# === Input image from argument (with fallback) ===
INPUT_IMAGE=${1:-"data/EHF/01_img.jpg"}
BASE_OUTPUT_DIR="reference_pipeline"

echo "🚀 Running unified pipeline for image: $INPUT_IMAGE"
echo "📂 Output base directory: $BASE_OUTPUT_DIR"
echo ""

# === Step 1: Run main adapter pipeline ===
python main.py --input_image "$INPUT_IMAGE" --output_dir "$BASE_OUTPUT_DIR"
echo ""

# === Step 2: Detect latest run folder ===
LATEST_RUN=$(ls -dt "$BASE_OUTPUT_DIR"/run_* | head -n 1)
echo "📁 Detected latest run directory: $LATEST_RUN"
echo ""

# === Step 3: Run parameter analysis ===
# echo "🔍 Running parameter analyzer ..."
# python analysis_tools/parameter_analyzer.py --results_dir "$LATEST_RUN"
# echo ""

# === Step 4: Run coordinate alignment analysis ===
echo "🧭 Running coordinate analyzer ..."
python analysis_tools/coordinate_analyzer_fixed.py "$LATEST_RUN"
echo ""

# === Step 5: Run parameter fusion ===
echo "🧬 Running direct parameter fusion ..."
python fusion/direct_parameter_fusion.py --results_dir "$LATEST_RUN"
echo ""

# === Step 6: Visualize fused mesh ===
echo "🎨 Running fusion visualizer ..."
python fusion/fusion_visualizer.py --results_dir "$LATEST_RUN"
echo ""

echo "✅ Pipeline complete for image: $INPUT_IMAGE"