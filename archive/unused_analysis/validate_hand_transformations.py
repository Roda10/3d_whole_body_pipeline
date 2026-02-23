#!/usr/bin/env python3
"""
Quick validation to check if hand pose transformations are working
"""

import numpy as np
import json
from pathlib import Path

def validate_hand_transformations(results_dir: str):
    """Check if the hand transformations are producing reasonable results"""
    
    results_dir = Path(results_dir)
    
    print("🔍 Validating hand pose transformations...\n")
    
    # Load original and fused parameters
    fusion_dir = results_dir / 'fusion_results'
    
    # Load fused parameters
    fused_path = fusion_dir / 'fused_parameters.json'
    if not fused_path.exists():
        print("❌ No fused parameters found. Run fusion first!")
        return
    
    with open(fused_path, 'r') as f:
        fused = json.load(f)
    
    # Load original SMPLX parameters
    smplx_params = None
    for param_file in results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
        with open(param_file, 'r') as f:
            smplx_params = json.load(f)
            break
    
    if not smplx_params:
        print("❌ No SMPLest-X parameters found!")
        return
    
    # Analyze differences
    left_orig = np.array(smplx_params['left_hand_pose'])
    right_orig = np.array(smplx_params['right_hand_pose'])
    left_fused = np.array(fused['left_hand_pose'])
    right_fused = np.array(fused['right_hand_pose'])
    
    print("📊 HAND POSE ANALYSIS:")
    print("=" * 50)
    
    # Check if poses actually changed
    left_diff = np.abs(left_fused - left_orig).mean()
    right_diff = np.abs(right_fused - right_orig).mean()
    
    print(f"\n1. Change Detection:")
    print(f"   Left hand difference: {left_diff:.4f}")
    print(f"   Right hand difference: {right_diff:.4f}")
    
    if left_diff < 0.01 and right_diff < 0.01:
        print("   ❌ PROBLEM: Poses haven't changed! Fusion not working.")
    else:
        print("   ✅ Poses have changed significantly")
    
    # Check pose ranges
    print(f"\n2. Pose Ranges (should be roughly -π to π):")
    print(f"   Original left: [{left_orig.min():.2f}, {left_orig.max():.2f}]")
    print(f"   Fused left: [{left_fused.min():.2f}, {left_fused.max():.2f}]")
    print(f"   Original right: [{right_orig.min():.2f}, {right_orig.max():.2f}]")
    print(f"   Fused right: [{right_fused.min():.2f}, {right_fused.max():.2f}]")
    
    # Check for extreme values
    if np.abs(left_fused).max() > np.pi * 2 or np.abs(right_fused).max() > np.pi * 2:
        print("   ⚠️  WARNING: Some pose values are extreme!")
    else:
        print("   ✅ Pose values in reasonable range")
    
    # Visual pose pattern check
    print(f"\n3. Pose Patterns:")
    print(f"   Left hand first 5 values: {left_fused[:5].round(3)}")
    print(f"   Right hand first 5 values: {right_fused[:5].round(3)}")
    
    # Success indicators
    print("\n" + "=" * 50)
    print("EXPECTED RESULTS FOR SUCCESS:")
    print("- Significant difference from original (> 0.1)")
    print("- Pose values mostly between -3.14 and 3.14")
    print("- Right hand should look natural first")
    print("- Left hand may need fine-tuning")
    
    return {
        'left_changed': left_diff > 0.01,
        'right_changed': right_diff > 0.01,
        'reasonable_ranges': np.abs(left_fused).max() < np.pi * 2 and np.abs(right_fused).max() < np.pi * 2
    }

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    validate_hand_transformations(args.results_dir)