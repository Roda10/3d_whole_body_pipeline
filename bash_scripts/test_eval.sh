#!/bin/bash

# Test evaluation on 10 frames with drift analysis
timestamp=$(date +%Y%m%d_%H%M%S)
out_dir="evaluation_results/ehf_test_${timestamp}"
mkdir -p "$out_dir"

echo "Starting evaluation on test frames..."
echo "Output directory: $out_dir"

# Define frames array
frames=("01" "02" "03")

# Process each frame and copy results
for frame in ${frames[@]}; do
    echo "Processing frame ${frame}..."
    # Run evaluation with verbose output
    python3 evaluation/ehf_fusion_evaluator.py --frame_id "${frame}" --verbose_output
    
    # Get latest evaluation directory
    latest_eval=$(ls -dt evaluation_results/ehf_compatible_opt_* | head -n 1)
    
    # Copy and rename the results
    cp "${latest_eval}/evaluation_comparison_results.json" "${out_dir}/frame_${frame}_result.json"
done

# Run drift analysis
echo -e "\nAnalyzing results..."
python3 evaluation/frame_validator.py --results_dir "${out_dir}" --frames "${frames[@]}" --save
echo "Done! Results saved in ${out_dir}"

echo "Evaluation complete. Results saved in $OUTPUT_DIR"