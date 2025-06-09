#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Starting Evaluation on GPU 0 ---"

# Ensure we are in the correct project directory
# This makes the script runnable from anywhere
cd "$(dirname "$0")"

# Activate Conda environment
# Use the full path to your conda.sh for robustness
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate unified_pose_env

# Run the evaluation for GPU 0
echo "Running evaluator for frames in frames_gpu0.txt..."
python evaluation/fast_ehf_fusion_evaluator.py \
    --workers 2 \
    --frame_list frames_gpu0.txt
    
echo "--- GPU 0 Evaluation Finished ---"