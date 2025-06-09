# 4. Show me the structures
echo "=== Pipeline Structure ==="
find reference_pipeline -name "*.json" | head -5

echo "=== Coordinate Analysis Output ==="
ls -la reference_pipeline/run_*/coordinate_analysis_summary.json

echo "=== Fusion Input/Output ==="
ls -la reference_pipeline/run_*/fusion_results/