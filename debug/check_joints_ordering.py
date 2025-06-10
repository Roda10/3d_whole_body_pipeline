#!/usr/bin/env python3
"""
Check if WiLoR and SMPLX use different joint ordering
"""

import numpy as np
import json
from pathlib import Path

def check_joint_ordering(results_dir: str):
    """
    Analyze joint ordering differences between WiLoR and SMPLX
    """
    results_dir = Path(results_dir)
    
    print("🔍 CHECKING JOINT ORDERING")
    print("=" * 60)
    
    # MANO/WiLoR joint order (15 joints, 3 DoF each = 45 params)
    # Based on MANO documentation
    mano_joint_order = [
        # Wrist is separate (global_orient)
        'thumb_base', 'thumb_mid', 'thumb_tip',      # 0-2  (params 0-8)
        'index_base', 'index_mid', 'index_tip',      # 3-5  (params 9-17)
        'middle_base', 'middle_mid', 'middle_tip',   # 6-8  (params 18-26)
        'ring_base', 'ring_mid', 'ring_tip',         # 9-11 (params 27-35)
        'pinky_base', 'pinky_mid', 'pinky_tip'       # 12-14 (params 36-44)
    ]
    
    # SMPLX might use a different order
    # This is a common alternative ordering
    smplx_potential_order = [
        # Some models order by joint level rather than by finger
        'thumb_base', 'index_base', 'middle_base', 'ring_base', 'pinky_base',  # All bases first
        'thumb_mid', 'index_mid', 'middle_mid', 'ring_mid', 'pinky_mid',       # All mids
        'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'        # All tips
    ]
    
    print("\n📋 Standard MANO joint ordering:")
    for i, joint in enumerate(mano_joint_order):
        param_idx = i * 3
        print(f"  Joint {i:2d}: {joint:15s} → params [{param_idx:2d}-{param_idx+2:2d}]")
    
    print("\n📋 Alternative SMPLX ordering (if different):")
    for i, joint in enumerate(smplx_potential_order):
        param_idx = i * 3
        print(f"  Joint {i:2d}: {joint:15s} → params [{param_idx:2d}-{param_idx+2:2d}]")
    
    # Load actual data to check patterns
    print("\n📊 Analyzing actual parameter patterns...")
    
    # Load fused parameters
    fusion_dir = results_dir / 'fusion_results'
    if (fusion_dir / 'fused_parameters.json').exists():
        with open(fusion_dir / 'fused_parameters.json', 'r') as f:
            fused = json.load(f)
        
        left_hand = np.array(fused['left_hand_pose']).reshape(15, 3)
        right_hand = np.array(fused['right_hand_pose']).reshape(15, 3)
        
        # Check for patterns that might indicate wrong ordering
        print("\n🔍 Checking for ordering issues:")
        
        # Test 1: Check if thumb has extreme values (often happens with wrong order)
        thumb_joints_mano = left_hand[0:3]  # First 3 joints in MANO order
        thumb_norm = np.linalg.norm(thumb_joints_mano, axis=1)
        print(f"\n  Thumb joints (MANO order) norms: {thumb_norm.round(2)}")
        
        # Test 2: Check finger progression
        print("\n  Finger progression analysis:")
        for finger_idx, finger_name in enumerate(['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']):
            base_idx = finger_idx * 3
            joints = left_hand[base_idx:base_idx+3]
            norms = np.linalg.norm(joints, axis=1)
            print(f"    {finger_name}: {norms.round(2)}")
        
        # Test 3: Create reordering map if needed
        print("\n💡 Creating reordering solution...")
        
        # If joints are ordered by level instead of by finger:
        mano_to_smplx_reorder = []
        for level in range(3):  # base, mid, tip
            for finger in range(5):  # 5 fingers
                original_idx = finger * 3 + level
                mano_to_smplx_reorder.append(original_idx)
        
        print(f"\n  Reordering map (if needed): {mano_to_smplx_reorder}")
        
        # Save reordering info
        reorder_info = {
            'mano_joint_names': mano_joint_order,
            'potential_smplx_order': smplx_potential_order,
            'reorder_indices': mano_to_smplx_reorder,
            'analysis': {
                'thumb_norms': thumb_norm.tolist(),
                'may_need_reordering': bool(thumb_norm[0] > 1.5)  # Heuristic
            }
        }
        
        output_path = results_dir / 'joint_ordering_analysis.json'
        with open(output_path, 'w') as f:
            json.dump(reorder_info, f, indent=2)
        
        print(f"\n✅ Saved analysis to: {output_path}")
        
        # Provide reordering function
        print("\n📝 If reordering is needed, use this function:")
        print("""
def reorder_mano_to_smplx(mano_pose):
    '''Reorder MANO joints to SMPLX format if needed'''
    # Reshape to per-joint
    joints = mano_pose.reshape(15, 3)
    
    # Reorder by level (all bases, all mids, all tips)
    reordered = []
    for level in range(3):  # base, mid, tip
        for finger in range(5):  # 5 fingers
            reordered.append(joints[finger * 3 + level])
    
    return np.array(reordered).flatten()
""")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    check_joint_ordering(args.results_dir)