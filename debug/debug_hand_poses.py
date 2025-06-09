# FILE: fusion/debug_hand_poses.py
# (Upgraded to analyze BOTH hands)

import numpy as np
import json
from pathlib import Path
from typing import Tuple
from scipy.spatial.transform import Rotation as R
import argparse

# Standard MANO/SMPL-X hand joint topology
JOINT_NAMES = {
    0: "WRIST",
    1: "Index_MCP", 2: "Index_PIP", 3: "Index_DIP",
    4: "Middle_MCP", 5: "Middle_PIP", 6: "Middle_DIP",
    7: "Pinky_MCP", 8: "Pinky_PIP", 9: "Pinky_DIP",
    10: "Ring_MCP", 11: "Ring_PIP", 12: "Ring_DIP",
    13: "Thumb_MCP", 14: "Thumb_PIP", 15: "Thumb_DIP"
}

def get_wilor_poses(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Loads WiLoR data and converts it to axis-angle."""
    left_hand_pose = np.zeros(45); right_hand_pose = np.zeros(45)
    wilor_file = next(run_dir.glob('wilor_results/*_parameters.json'))
    with open(wilor_file, 'r') as f: wilor_params = json.load(f)

    for hand in wilor_params.get('hands', []):
        if 'mano_parameters' in hand and 'parameters' in hand['mano_parameters']:
            if 'hand_pose' in hand['mano_parameters']['parameters']:
                matrices = np.array(hand['mano_parameters']['parameters']['hand_pose']['values']).reshape(15, 3, 3)
                axis_angles = R.from_matrix(matrices).as_rotvec().flatten()
                if hand.get('hand_type') == 'left':
                    left_hand_pose = axis_angles
                elif hand.get('hand_type') == 'right':
                    right_hand_pose = axis_angles
    return left_hand_pose, right_hand_pose

def get_smplestx_poses(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Loads SMPLest-X hand poses directly."""
    param_file = next(run_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'))
    with open(param_file, 'r') as f: smplx_params = json.load(f)
    return np.array(smplx_params['left_hand_pose']), np.array(smplx_params['right_hand_pose'])

def print_comparison_table(hand_side: str, smplx_pose: np.ndarray, wilor_pose: np.ndarray):
    """Prints a formatted comparison table for a single hand."""
    smplx_pose = smplx_pose.reshape(15, 3)
    wilor_pose = wilor_pose.reshape(15, 3)
    
    print("\n" + "="*80)
    print(f"HAND POSE COMPARISON: {hand_side.upper()} HAND")
    print("="*80)
    print(f"{'JOINT':<12} | {'SMPLest-X (X, Y, Z)':<30} | {'WiLoR (X, Y, Z)':<30} | {'L2 Norm Diff':<15}")
    print("-"*80)
    
    for i in range(15):
        joint_name = JOINT_NAMES.get(i, f"Joint_{i}")
        s_pose = smplx_pose[i]
        w_pose = wilor_pose[i]
        diff_norm = np.linalg.norm(s_pose - w_pose)
        
        s_str = f"{s_pose[0]:>8.4f}, {s_pose[1]:>8.4f}, {s_pose[2]:>8.4f}"
        w_str = f"{w_pose[0]:>8.4f}, {w_pose[1]:>8.4f}, {w_pose[2]:>8.4f}"
        
        print(f"{joint_name:<12} | {s_str:<30} | {w_str:<30} | {diff_norm:<15.4f}")

def main():
    parser = argparse.ArgumentParser(description="Debug and Compare Hand Poses")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to a single pipeline run directory (e.g., ./pipeline_results/run_...)')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: Directory not found at {run_dir}"); return

    # Load poses from both sources
    wilor_left, wilor_right = get_wilor_poses(run_dir)
    smplestx_left, smplestx_right = get_smplestx_poses(run_dir)
    
    # Print tables for both hands
    print_comparison_table("Left", smplestx_left, wilor_left)
    print_comparison_table("Right", smplestx_right, wilor_right)

if __name__ == '__main__':
    main()