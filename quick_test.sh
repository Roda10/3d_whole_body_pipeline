# Run a single persistence pipeline
./quick_start.sh run data/full_images/test2.jpg

# Find the latest run directory
latest_run=$(ls -td pipeline_results_persistence/run_* | head -1)
echo "=== ACTUAL FILE STRUCTURE CREATED ==="
echo "Latest run: $latest_run"

# Show COMPLETE file structure 
find "$latest_run" -type f | sort

echo ""
echo "=== SPECIFIC PATTERNS THE EVALUATOR LOOKS FOR ==="
echo "SMPLest-X patterns:"
find "$latest_run" -name "*smplx_params*" 2>/dev/null || echo "  ❌ No smplx_params files found"

echo "WiLoR patterns:"
find "$latest_run" -name "*parameters.json" 2>/dev/null || echo "  ❌ No parameters.json files found"

echo "EMOCA patterns:"
find "$latest_run" -name "codes.json" 2>/dev/null || echo "  ❌ No codes.json files found"