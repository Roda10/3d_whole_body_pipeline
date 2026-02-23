#!/bin/bash
# cleanup_codebase.sh - Prune redundant files

set -e

echo "🧹 Starting codebase cleanup..."

# Create archive directories
echo "📁 Creating archive directories..."
mkdir -p archive/old_evaluators
mkdir -p archive/old_fusion
mkdir -p archive/old_docs
mkdir -p archive/unused_analysis

# Phase 1: Archive redundant evaluators
echo "📦 Archiving redundant evaluators..."
mv evaluation/archive/* archive/old_evaluators/ 2>/dev/null || echo "   ⚠️  No archive evaluators found"

# Phase 2: Archive duplicate fusion
echo "📦 Archiving duplicate fusion implementations..."
[ -f fusion/direct_parameter_fusion_SOTA.py ] && mv fusion/direct_parameter_fusion_SOTA.py archive/old_fusion/

# Phase 3: Archive excessive documentation (keep essential ones)
echo "📦 Archiving excessive documentation..."
mv 00_START_HERE.md archive/old_docs/ 2>/dev/null || true
mv DOCUMENTATION_SUMMARY.txt archive/old_docs/ 2>/dev/null || true
mv QUICK_REFERENCE.md archive/old_docs/ 2>/dev/null || true
mv QUICK_START.md archive/old_docs/ 2>/dev/null || true
mv README_DOCUMENTATION.md archive/old_docs/ 2>/dev/null || true
mv COMPLETE_ANALYSIS.md archive/old_docs/ 2>/dev/null || true
mv ARCHITECTURE.md archive/old_docs/ 2>/dev/null || true
mv FUSION_GUIDE.md archive/old_docs/ 2>/dev/null || true
mv Fusion_Architecture archive/old_docs/ 2>/dev/null || true
mv FUSION_GUIDE archive/old_docs/ 2>/dev/null || true

# Phase 4: Remove unused analysis tools
echo "🗑️  Removing unused analysis tools..."
[ -f analysis_tools/parameter_analyzer.py ] && mv analysis_tools/parameter_analyzer.py archive/unused_analysis/
[ -f analysis_tools/validate_hand_transformations.py ] && mv analysis_tools/validate_hand_transformations.py archive/unused_analysis/

# Phase 5: Clean up empty archive directory in evaluation
echo "🗑️  Cleaning evaluation/archive..."
[ -d evaluation/archive ] && rmdir evaluation/archive 2>/dev/null || echo "   ⚠️  evaluation/archive not empty or already removed"

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "   - Archived evaluators: $(ls -1 archive/old_evaluators/ 2>/dev/null | wc -l) files"
echo "   - Archived fusion: $(ls -1 archive/old_fusion/ 2>/dev/null | wc -l) files"
echo "   - Archived docs: $(ls -1 archive/old_docs/ 2>/dev/null | wc -l) files"
echo "   - Archived analysis: $(ls -1 archive/unused_analysis/ 2>/dev/null | wc -l) files"
echo ""
echo "🎯 Remaining core files:"
echo "   ├── main.py"
echo "   ├── adapters/{smplestx,wilor,emoca}_adapter.py"
echo "   ├── analysis_tools/coordinate_analyzer_fixed.py"
echo "   ├── fusion/direct_parameter_fusion.py"
echo "   ├── evaluation/ehf_fusion_evaluator.py"
echo "   └── bash_scripts/run_all_pipeline.sh"