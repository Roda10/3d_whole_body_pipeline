#!/usr/bin/env python3
"""
Test different fusion approaches to find what works best
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def test_fusion_approaches(results_dir: str):
    """Test and compare different fusion approaches"""
    
    results_dir = Path(results_dir)
    print("🧪 TESTING FUSION APPROACHES")
    print("=" * 60)
    
    # Load current fused result
    with open(results_dir / 'fusion_results' / 'fused_parameters.json', 'r') as f:
        current_fused = json.load(f)
    
    # Load original SMPLX
    smplx_params = None
    for param_file in results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
        with open(param_file, 'r') as f:
            smplx_params = json.load(f)
            break
    
    # Analyze current approach
    print("\n1. CURRENT APPROACH ANALYSIS:")
    print("-" * 40)
    
    # Check if wrist is unchanged (hierarchical composition indicator)
    left_orig = np.array(smplx_params['left_hand_pose'])
    left_fused = np.array(current_fused['left_hand_pose'])
    
    wrist_unchanged = np.allclose(left_orig[:3], left_fused[:3], atol=1e-6)
    print(f"   Wrist unchanged (hierarchical): {wrist_unchanged}")
    
    if wrist_unchanged:
        print("   ⚠️  PROBLEM: Using hierarchical composition!")
        print("      This mixes SMPLest-X wrist with WiLoR fingers")
        print("      WiLoR fingers are relative to WiLoR wrist, not SMPLest-X!")
    
    # Analyze finger patterns
    left_joints = left_fused.reshape(15, 3)
    right_joints = np.array(current_fused['right_hand_pose']).reshape(15, 3)
    
    print("\n2. FINGER PATTERN ANALYSIS:")
    print("-" * 40)
    
    # Expected pattern: thumb should have moderate values, pinky smallest
    thumb_norm = np.linalg.norm(left_joints[0:3], axis=1).mean()
    index_norm = np.linalg.norm(left_joints[3:6], axis=1).mean()
    middle_norm = np.linalg.norm(left_joints[6:9], axis=1).mean()
    ring_norm = np.linalg.norm(left_joints[9:12], axis=1).mean()
    pinky_norm = np.linalg.norm(left_joints[12:15], axis=1).mean()
    
    print(f"   Left hand finger norms:")
    print(f"      Thumb:  {thumb_norm:.3f}")
    print(f"      Index:  {index_norm:.3f}")
    print(f"      Middle: {middle_norm:.3f}")
    print(f"      Ring:   {ring_norm:.3f}")
    print(f"      Pinky:  {pinky_norm:.3f}")
    
    # Check for issues
    if pinky_norm > thumb_norm:
        print("   ⚠️  ISSUE: Pinky has higher norm than thumb (unusual)")
    
    # Suggest fixes
    print("\n3. RECOMMENDED FIXES:")
    print("-" * 40)
    
    fixes = []
    
    if wrist_unchanged:
        fixes.append("Use FULL replacement instead of hierarchical composition")
        fixes.append("Include WiLoR's global_orient in the hand pose")
    
    if pinky_norm > thumb_norm * 1.2:
        fixes.append("Check joint ordering - might need reordering")
    
    if np.abs(left_fused).max() > 2.5:
        fixes.append("Apply scaling to keep values in reasonable range")
    
    for i, fix in enumerate(fixes, 1):
        print(f"   {i}. {fix}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Wrist comparison
    ax = axes[0, 0]
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width/2, left_orig[:3], width, label='Original', alpha=0.7)
    ax.bar(x + width/2, left_fused[:3], width, label='Fused', alpha=0.7)
    ax.set_xlabel('Axis')
    ax.set_ylabel('Value')
    ax.set_title('Left Wrist (First 3 params)')
    ax.set_xticks(x)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.legend()
    
    # Plot 2: Finger norms
    ax = axes[0, 1]
    fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    norms = [thumb_norm, index_norm, middle_norm, ring_norm, pinky_norm]
    ax.bar(fingers, norms)
    ax.set_ylabel('Average Joint Norm')
    ax.set_title('Left Hand Finger Patterns')
    ax.axhline(y=np.mean(norms), color='r', linestyle='--', label='Mean')
    
    # Plot 3: Full pose comparison
    ax = axes[1, 0]
    ax.plot(left_orig, 'b-', label='Original', alpha=0.7)
    ax.plot(left_fused, 'r-', label='Fused')
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Value')
    ax.set_title('Full Left Hand Pose')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Recommendations
    ax = axes[1, 1]
    ax.axis('off')
    rec_text = "QUICK FIX ATTEMPT:\n\n"
    rec_text += "# Replace these methods in your fusion code:\n\n"
    rec_text += "1. Change extract_wilor_hand_poses to use\n"
    rec_text += "   FULL replacement (include global_orient)\n\n"
    rec_text += "2. Remove hierarchical composition\n"
    rec_text += "   (don't mix SMPLX wrist + WiLoR fingers)\n\n"
    rec_text += "3. For left hand, try simpler transform:\n"
    rec_text += "   pose[1::3] *= -1  # Flip Y\n"
    rec_text += "   pose[2::3] *= -1  # Flip Z\n\n"
    rec_text += "4. Scale if values > 2.0"
    
    ax.text(0.05, 0.95, rec_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    output_path = results_dir / 'fusion_approach_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✅ Saved analysis to {output_path}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    test_fusion_approaches(args.results_dir)