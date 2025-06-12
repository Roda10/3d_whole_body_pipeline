#!/bin/bash
# Comprehensive Codebase Structure Scanner
# This script maps out the entire project structure

echo "🔍 COMPREHENSIVE CODEBASE STRUCTURE ANALYSIS"
echo "=============================================="
echo "Time: $(date)"
echo "Directory: $(pwd)"
echo ""

# Basic project structure
echo "📁 TOP-LEVEL DIRECTORY STRUCTURE:"
echo "=================================="
ls -la
echo ""

# Key directories
echo "📂 KEY DIRECTORIES:"
echo "=================="
for dir in data external pretrained_models human_models services evaluation fusion analysis_tools pipeline_results* evaluation_results*; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/ exists"
        echo "   Files: $(find "$dir" -type f | wc -l)"
        echo "   Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
    else
        echo "❌ $dir/ missing"
    fi
done
echo ""

# Python scripts
echo "🐍 PYTHON SCRIPTS:"
echo "=================="
echo "Main scripts:"
find . -maxdepth 1 -name "*.py" -type f | sort

echo ""
echo "Service scripts:"
find services -name "*.py" -type f 2>/dev/null | sort

echo ""
echo "Evaluation scripts:"
find evaluation -name "*.py" -type f 2>/dev/null | sort

echo ""
echo "Fusion scripts:"
find fusion -name "*.py" -type f 2>/dev/null | sort

echo ""
echo "Analysis scripts:"
find analysis_tools -name "*.py" -type f 2>/dev/null | sort
echo ""

# Configuration files
echo "⚙️  CONFIGURATION FILES:"
echo "======================="
find . -name "*.py" -path "*/config*" -o -name "config*.py" -o -name "*.yaml" -o -name "*.yml" | sort
echo ""

# Data structure
echo "📊 DATA DIRECTORY STRUCTURE:"
echo "============================"
if [ -d "data" ]; then
    find data -type d | head -20
    echo ""
    echo "Data files (first 20):"
    find data -type f | head -20
else
    echo "❌ No data directory found"
fi
echo ""

# External repositories
echo "🔗 EXTERNAL REPOSITORIES:"
echo "========================="
if [ -d "external" ]; then
    echo "External subdirs:"
    ls -la external/
    echo ""
    echo "External Python files:"
    find external -name "*.py" -type f | head -10
else
    echo "❌ No external directory found"
fi
echo ""

# Pretrained models
echo "🤖 PRETRAINED MODELS:"
echo "===================="
if [ -d "pretrained_models" ]; then
    find pretrained_models -type f | head -15
else
    echo "❌ No pretrained_models directory found"
fi
echo ""

# Pipeline results structure
echo "🏃 PIPELINE RESULTS ANALYSIS:"
echo "============================="
echo "Pipeline result directories:"
ls -td pipeline_results* 2>/dev/null | head -5

if ls pipeline_results* >/dev/null 2>&1; then
    latest_pipeline=$(ls -td pipeline_results*/run_* 2>/dev/null | head -1)
    if [ -n "$latest_pipeline" ]; then
        echo ""
        echo "LATEST PIPELINE RUN STRUCTURE:"
        echo "============================="
        echo "Directory: $latest_pipeline"
        echo ""
        echo "Complete file tree:"
        find "$latest_pipeline" -type f | sort
        echo ""
        echo "Directory structure:"
        find "$latest_pipeline" -type d | sort
    else
        echo "❌ No pipeline run directories found"
    fi
else
    echo "❌ No pipeline_results directories found"
fi
echo ""

# Evaluation results structure  
echo "📈 EVALUATION RESULTS ANALYSIS:"
echo "==============================="
echo "Evaluation result directories:"
ls -td evaluation_results* 2>/dev/null | head -5

if ls evaluation_results* >/dev/null 2>&1; then
    latest_eval=$(ls -td evaluation_results* 2>/dev/null | head -1)
    if [ -n "$latest_eval" ]; then
        echo ""
        echo "LATEST EVALUATION STRUCTURE:"
        echo "==========================="
        echo "Directory: $latest_eval"
        echo ""
        echo "Files in evaluation:"
        find "$latest_eval" -type f | head -20
        echo ""
        echo "Subdirectories:"
        find "$latest_eval" -type d | head -10
    fi
else
    echo "❌ No evaluation_results directories found"
fi
echo ""

# Quick start and service files
echo "🚀 SERVICE AND STARTUP FILES:"
echo "============================="
echo "Startup scripts:"
ls -la quick_start.sh service_manager.py main*.py 2>/dev/null
echo ""

# Python environment info
echo "🐍 PYTHON ENVIRONMENT:"
echo "====================="
echo "Python path: $(which python)"
echo "Python version: $(python --version 2>&1)"
echo "Current working directory: $(pwd)"
echo ""

# Git status (if available)
echo "📝 GIT STATUS:"
echo "============="
if [ -d ".git" ]; then
    echo "Branch: $(git branch --show-current 2>/dev/null)"
    echo "Recent commits:"
    git log --oneline -3 2>/dev/null
    echo ""
    echo "Modified files:"
    git status --porcelain 2>/dev/null | head -10
else
    echo "❌ Not a git repository"
fi
echo ""

# Search for specific file patterns
echo "🔍 KEY FILE PATTERNS:"
echo "===================="
echo "Parameter files:"
find . -name "*param*" -type f | grep -E "\.(json|pkl|pth|tar)$" | head -10

echo ""
echo "Config files:"
find . -name "*config*" -type f | head -10

echo ""
echo "Model files:"
find . -name "*.pth*" -o -name "*.pkl" -o -name "*.onnx" | head -10

echo ""
echo "JSON result files:"
find . -name "*.json" -type f | grep -v __pycache__ | head -15

echo ""
echo "Mesh files:"
find . -name "*.obj" -o -name "*.ply" -o -name "*.npy" | head -10
echo ""

# File counts by type
echo "📊 FILE TYPE SUMMARY:"
echo "===================="
echo "Python files: $(find . -name "*.py" -type f | wc -l)"
echo "JSON files: $(find . -name "*.json" -type f | wc -l)"
echo "Image files: $(find . \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) -type f | wc -l)"
echo "Model files: $(find . \( -name "*.pth" -o -name "*.pkl" -o -name "*.tar" \) -type f | wc -l)"
echo "Mesh files: $(find . \( -name "*.obj" -o -name "*.ply" -o -name "*.npy" \) -type f | wc -l)"
echo ""

# Disk usage
echo "💾 DISK USAGE:"
echo "============="
echo "Total project size: $(du -sh . 2>/dev/null | cut -f1)"
echo ""
echo "Largest directories:"
du -sh */ 2>/dev/null | sort -hr | head -10
echo ""

echo "✅ CODEBASE STRUCTURE ANALYSIS COMPLETE"
echo "======================================="
echo ""
echo "To save this output to a file, run:"
echo "bash this_script.sh > codebase_structure.txt 2>&1"