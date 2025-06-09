#!/bin/bash
set -e

echo "--- Starting Evaluation on GPU 1 ---"

cd "$(dirname "$0")"

# Activate Conda environment
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate unified_pose_env

# Set the GPU and run the command
# NOTE: The only difference is CUDA_VISIBLE_DEVICES and the frame_list file
export CUDA_VISIBLE_DEVICES=1

echo "Running evaluator for frames in frames_gpu1.txt..."
python evaluation/fast_ehf_fusion_evaluator.py \
    --workers 3 \
    --frame_list frames_gpu1.txt
    
echo "--- GPU 1 Evaluation Finished ---"