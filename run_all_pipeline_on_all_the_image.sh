#!/bin/bash

# ==============================================================================
# Unified Pipeline Runner (Upgraded for Parallel & Limited Batch Processing)
#
# This script can be called in three ways:
# 1. With a single image file:
#    ./run_all_pipeline.sh /path/to/your/image.jpg
#
# 2. With a directory to process ALL images:
#    ./run_all_pipeline.sh /path/to/your/images_folder/
#
# 3. With a directory AND a limit on the number of images:
#    ./run_all_pipeline.sh /path/to/your/images_folder/ 10
# ==============================================================================

# --- Configuration ---
# source /opt/conda/etc/profile.d/conda.sh
# conda activate unified_pose_env
# OUTPUT_BASE_DIR="pipeline_results"
# mkdir -p "$OUTPUT_BASE_DIR"

# --- Argument Validation ---
if [[ "$#" -eq 0 ]] || [[ "$#" -gt 2 ]]; then
    echo "Usage: $0 [path_to_image.jpg | path_to_images_folder] [OPTIONAL: number_of_images]"
    exit 1
fi

INPUT_PATH=$1
# The second argument is the limit, default to a very large number if not provided.
LIMIT=${2:-999999} 

# ==============================================================================
# Function to process a SINGLE image file
# (This function does not need any changes)
# ==============================================================================
process_single_image() {
    local IMAGE_FILE=$1
    echo "======================================================================"
    echo "🚀 Processing Image: $IMAGE_FILE"
    echo "======================================================================"

    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    IMAGE_BASENAME=$(basename -- "$IMAGE_FILE")
    IMAGE_STEM="${IMAGE_BASENAME%.*}"
    RUN_DIR="$OUTPUT_BASE_DIR/run_${TIMESTAMP}_${IMAGE_STEM}_$$"
    mkdir -p "$RUN_DIR"
    echo "📂 Outputting to: $RUN_DIR"

    SMPLESTX_OUT_DIR="$RUN_DIR/smplestx_results"; WILOR_OUT_DIR="$RUN_DIR/wilor_results"; EMOCA_OUT_DIR="$RUN_DIR/emoca_results"
    mkdir -p "$SMPLESTX_OUT_DIR" "$WILOR_OUT_DIR/temp_input" "$EMOCA_OUT_DIR/temp_input"

    cp "$IMAGE_FILE" "$WILOR_OUT_DIR/temp_input/"
    cp "$IMAGE_FILE" "$EMOCA_OUT_DIR/temp_input/"

    echo "▶️  Starting pipeline components for $IMAGE_STEM..."
    python adapters/smplestx_adapter.py --input_image "$IMAGE_FILE" --output_dir "$SMPLESTX_OUT_DIR" --multi_person &> "$RUN_DIR/smplestx.log" &
    SMPLESTX_PID=$!
    ( cd external/WiLoR || exit; python wilor_adapter.py --img_folder "../../$WILOR_OUT_DIR/temp_input" --out_folder "../../$WILOR_OUT_DIR" --rescale_factor 2.0 &> "../../$RUN_DIR/wilor.log" ) &
    WILOR_PID=$!
    python adapters/emoca_adapter.py --input_folder "$EMOCA_OUT_DIR/temp_input" --output_folder "$EMOCA_OUT_DIR" --mode detail &> "$RUN_DIR/emoca.log" &
    EMOCA_PID=$!
    
    wait $SMPLESTX_PID; wait $WILOR_PID; wait $EMOCA_PID
    echo "✅ All components finished for $IMAGE_STEM."

    echo "🧬 Running fusion for $IMAGE_STEM..."
    python fusion/direct_parameter_fusion.py --results_dir "$RUN_DIR" &> "$RUN_DIR/fusion.log"
    
    echo "✅ Fusion complete for $IMAGE_FILE"
}
export -f process_single_image

# ==============================================================================
# Main Execution Logic
# ==============================================================================
if [ -d "$INPUT_PATH" ]; then
    # --- Input is a DIRECTORY ---
    echo "📂 Input is a directory. Preparing to process up to $LIMIT images."
    
    # --- PARALLEL EXECUTION CONFIGURATION ---
    # On a g2-standard-12, 4 parallel jobs is a good starting point.
    NUM_PARALLEL_JOBS=4
    echo "🛠️  Configured to run $NUM_PARALLEL_JOBS jobs in parallel."

    # --- THE MODIFICATION IS HERE ---
    # Find all images, sort them, take the first N using 'head', then pipe to xargs.
    find "$INPUT_PATH" -maxdepth 1 -type f \( -iname \*.jpg -o -iname \*.jpeg -o -iname \*.png \) | \
        sort | head -n "$LIMIT" | xargs -I {} -P "$NUM_PARALLEL_JOBS" bash -c 'process_single_image "{}"'
    # --- END OF MODIFICATION ---

    echo ""
    echo "🎉 Batch processing complete."

elif [ -f "$INPUT_PATH" ]; then
    # --- Input is a single FILE ---
    if [ "$LIMIT" -ne 999999 ]; then
        echo "⚠️  Warning: A number limit was provided, but the input is a single file. Processing just the one file."
    fi
    echo "📄 Input is a single file."
    process_single_image "$INPUT_PATH"
    echo ""
    echo "🎉 Processing complete for $INPUT_PATH"

else
    # --- Input is not valid ---
    echo "❌ Error: Input path '$INPUT_PATH' is not a valid file or directory."
    exit 1
fi