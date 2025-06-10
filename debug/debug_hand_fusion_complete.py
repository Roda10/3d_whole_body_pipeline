#!/usr/bin/env python3
"""
Comprehensive debugging system for WiLoR to SMPLX hand fusion
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R
import os

class HandFusionDebugger:
    """Debug the entire hand fusion pipeline"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.debug_dir = self.results_dir / 'hand_fusion_debug'
        self.debug_dir.mkdir(exist_ok=True)
        
    def run_complete_debug(self):
        """Run all debugging checks"""
        print("🔍 COMPREHENSIVE HAND FUSION DEBUGGING")
        print("=" * 60)
        
        # 1. Check MANO mean poses
        self.debug_mano_means()
        
        # 2. Analyze WiLoR raw output
        self.debug_wilor_raw()
        
        # 3. Test transformations
        self.debug_transformations()
        
        # 4. Compare with ground truth
        self.debug_final_comparison()
        
        print("\n✅ Debug complete. Check", self.debug_dir)
    
    def debug_mano_means(self):
        """Debug MANO mean pose issues"""
        print("\n1. DEBUGGING MANO MEAN POSES")
        print("-" * 40)
        
        # Check multiple possible locations
        mean_paths = [
            'pretrained_models/mano_mean_params.npz',
            'pretrained_models/mano/MANO_RIGHT.pkl',
            'pretrained_models/mano/MANO_LEFT.pkl',
            'human_models/human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl',
            'human_models/human_model_files/smplx/SMPLX_NEUTRAL.pkl'
        ]
        
        means_found = {}
        
        for path in mean_paths:
            if os.path.exists(path):
                print(f"✓ Found: {path}")
                
                if path.endswith('.npz'):
                    data = np.load(path)
                    print(f"  Keys: {list(data.keys())}")
                    for key in data.keys():
                        if 'mean' in key or 'pose' in key:
                            print(f"  {key}: shape={data[key].shape}")
                            
                elif path.endswith('.pkl'):
                    try:
                        with open(path, 'rb') as f:
                            data = pickle.load(f, encoding='latin1')
                        
                        if isinstance(data, dict):
                            relevant_keys = [k for k in data.keys() if 'hand' in str(k).lower() or 'mean' in str(k).lower()]
                            if relevant_keys:
                                print(f"  Relevant keys: {relevant_keys}")
                                
                                # Check for SMPLX model with hand means
                                if 'hands_mean' in data:
                                    means_found['hands_mean'] = data['hands_mean']
                                    print(f"  hands_mean shape: {data['hands_mean'].shape}")
                                    
                                if 'hands_meanl' in data:  # Left hand mean
                                    means_found['left'] = data['hands_meanl']
                                    print(f"  Left hand mean shape: {data['hands_meanl'].shape}")
                                    
                                if 'hands_meanr' in data:  # Right hand mean
                                    means_found['right'] = data['hands_meanr']
                                    print(f"  Right hand mean shape: {data['hands_meanr'].shape}")
                    except Exception as e:
                        print(f"  Error loading: {e}")
        
        # Save what we found
        np.savez(self.debug_dir / 'found_means.npz', **means_found)
        
        # Create zero means as fallback
        print("\n📝 Creating zero mean poses as comparison baseline...")
        np.savez(self.debug_dir / 'zero_means.npz',
                 hands_mean_left=np.zeros(45),
                 hands_mean_right=np.zeros(45))
    
    def debug_wilor_raw(self):
        """Analyze raw WiLoR output"""
        print("\n2. DEBUGGING WILOR RAW OUTPUT")
        print("-" * 40)
        
        # Load WiLoR parameters
        wilor_params = None
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f:
                wilor_params = json.load(f)
                break
        
        if not wilor_params:
            print("❌ No WiLoR parameters found!")
            return
        
        debug_info = {}
        
        for i, hand in enumerate(wilor_params.get('hands', [])):
            hand_type = hand.get('hand_type', 'unknown')
            print(f"\n🖐️  {hand_type.upper()} HAND:")
            
            if 'mano_parameters' in hand and 'parameters' in hand['mano_parameters']:
                params = hand['mano_parameters']['parameters']
                
                # Check rotation matrices
                if 'hand_pose' in params:
                    pose_data = params['hand_pose']
                    values = np.array(pose_data['values'])
                    shape = pose_data['shape']
                    
                    print(f"  Raw shape: {shape}")
                    print(f"  Values shape: {values.shape}")
                    print(f"  Min/Max: [{values.min():.3f}, {values.max():.3f}]")
                    
                    # Check if valid rotation matrices
                    # print(f'values type: {len(values)}')
                    n_joints = len(values)
                    rot_mats = values.reshape(n_joints, 3, 3)
                    
                    print(f"  Number of joints: {n_joints}")
                    
                    # Check orthogonality of first matrix
                    first_mat = rot_mats[0]
                    orthogonality = np.abs(np.dot(first_mat, first_mat.T) - np.eye(3)).max()
                    print(f"  First matrix orthogonality error: {orthogonality:.6f}")
                    
                    # Convert to axis-angle
                    axis_angles = []
                    for j in range(n_joints):
                        aa = R.from_matrix(rot_mats[j]).as_rotvec()
                        axis_angles.append(aa)
                    
                    axis_angles = np.array(axis_angles).flatten()
                    print(f"  Axis-angle shape: {axis_angles.shape}")
                    print(f"  Axis-angle range: [{axis_angles.min():.3f}, {axis_angles.max():.3f}]")
                    
                    debug_info[f'{hand_type}_raw'] = {
                        'rotation_matrices': rot_mats,
                        'axis_angles': axis_angles,
                        'orthogonality_error': orthogonality
                    }
        
        # Save debug info
        np.save(self.debug_dir / 'wilor_raw_debug.npy', debug_info)
    
    def debug_transformations(self):
        """Test different transformation approaches"""
        print("\n3. TESTING DIFFERENT TRANSFORMATIONS")
        print("-" * 40)
        
        # Load WiLoR data
        wilor_params = None
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f:
                wilor_params = json.load(f)
                break
        
        # Test different mean poses
        test_means = {
            'zero': np.zeros(45),
            'small': np.ones(45) * 0.1,
            'smplx_typical': self._get_typical_smplx_mean()
        }
        
        results = {}
        
        for hand in wilor_params.get('hands', []):
            if 'mano_parameters' not in hand:
                continue
                
            hand_type = hand.get('hand_type', '')
            params = hand['mano_parameters']['parameters']
            
            if 'hand_pose' in params:
                # Get raw rotation matrices
                rot_matrices = np.array(params['hand_pose']['values']).flatten()
                n_joints = rot_matrices.shape[0] // 9
                rot_mats = rot_matrices.reshape(n_joints, 3, 3)
                
                # Convert to axis-angle
                axis_angles = []
                for i in range(n_joints):
                    aa = R.from_matrix(rot_mats[i]).as_rotvec()
                    axis_angles.append(aa)
                axis_angles = np.concatenate(axis_angles)[:45]
                
                print(f"\n🖐️  Testing {hand_type} hand transformations:")
                
                # Test different approaches
                transforms = {}
                
                # Original (no transform)
                transforms['original'] = axis_angles.copy()
                
                # GitHub approach
                if hand_type == 'left':
                    github_style = axis_angles.copy()
                    github_style *= -1
                    github_style[::3] *= -1
                    transforms['github_flip'] = github_style
                else:
                    transforms['github_flip'] = axis_angles.copy()
                
                # Alternative: flip only Y
                if hand_type == 'left':
                    flip_y = axis_angles.copy()
                    flip_y[1::3] *= -1
                    transforms['flip_y_only'] = flip_y
                
                # Alternative: flip Y and Z
                if hand_type == 'left':
                    flip_yz = axis_angles.copy()
                    flip_yz[1::3] *= -1
                    flip_yz[2::3] *= -1
                    transforms['flip_yz'] = flip_yz
                
                # Test with different means
                for mean_name, mean_pose in test_means.items():
                    for transform_name, transform in transforms.items():
                        final = transform - mean_pose
                        key = f"{hand_type}_{transform_name}_{mean_name}"
                        results[key] = {
                            'values': final,
                            'range': [final.min(), final.max()],
                            'std': final.std()
                        }
                        print(f"  {key}: range=[{final.min():.2f}, {final.max():.2f}], std={final.std():.2f}")
        
        # Save all results
        np.save(self.debug_dir / 'transformation_tests.npy', results)
        
        # Create visualization
        self._visualize_transforms(results)
    
    def debug_final_comparison(self):
        """Compare final results with expected"""
        print("\n4. FINAL COMPARISON")
        print("-" * 40)
        
        # Load all relevant data
        fusion_dir = self.results_dir / 'fusion_results'
        
        # Load fused parameters
        with open(fusion_dir / 'fused_parameters.json', 'r') as f:
            fused = json.load(f)
        
        # Load original SMPLX
        smplx_params = None
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f:
                smplx_params = json.load(f)
                break
        
        print("\n📊 Parameter Statistics:")
        print(f"  Original left hand std: {np.std(smplx_params['left_hand_pose']):.3f}")
        print(f"  Fused left hand std: {np.std(fused['left_hand_pose']):.3f}")
        print(f"  Original right hand std: {np.std(smplx_params['right_hand_pose']):.3f}")
        print(f"  Fused right hand std: {np.std(fused['right_hand_pose']):.3f}")
        
        # Check finger articulation
        left_orig = np.array(smplx_params['left_hand_pose']).reshape(15, 3)
        left_fused = np.array(fused['left_hand_pose']).reshape(15, 3)
        right_orig = np.array(smplx_params['right_hand_pose']).reshape(15, 3)
        right_fused = np.array(fused['right_hand_pose']).reshape(15, 3)
        
        print("\n🔍 Per-joint analysis:")
        joint_names = ['Thumb1', 'Thumb2', 'Thumb3', 'Index1', 'Index2', 'Index3',
                      'Middle1', 'Middle2', 'Middle3', 'Ring1', 'Ring2', 'Ring3',
                      'Pinky1', 'Pinky2', 'Pinky3']
        
        for i, name in enumerate(joint_names[:5]):  # First 5 joints
            print(f"\n  {name}:")
            print(f"    Left orig: {left_orig[i].round(2)}")
            print(f"    Left fused: {left_fused[i].round(2)}")
            print(f"    Right orig: {right_orig[i].round(2)}")
            print(f"    Right fused: {right_fused[i].round(2)}")
    
    def _get_typical_smplx_mean(self):
        """Get a typical SMPLX hand mean pose"""
        # These are typical values for a slightly curled hand
        mean = np.zeros(45)
        # Add slight curl to fingers (not thumb)
        for i in range(4, 15):  # Skip thumb joints
            if i % 3 == 1:  # Middle joints of each finger
                mean[i*3:(i+1)*3] = [0.0, 0.2, 0.0]  # Slight flex
        return mean
    
    def _visualize_transforms(self, results):
        """Create visualization of different transformations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Hand Pose Transformation Analysis', fontsize=16)
        
        # Get representative transformations
        plot_configs = [
            ('left_original_zero', 'Left Original'),
            ('left_github_flip_zero', 'Left GitHub Flip'),
            ('left_flip_yz_zero', 'Left Flip YZ'),
            ('right_original_zero', 'Right Original'),
            ('right_github_flip_zero', 'Right GitHub'),
            ('right_original_smplx_typical', 'Right w/ Mean')
        ]
        
        for idx, (key, title) in enumerate(plot_configs):
            ax = axes[idx // 3, idx % 3]
            
            if key in results:
                values = results[key]['values']
                ax.plot(values, 'b-', linewidth=2)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.3)
                ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.3)
                ax.set_title(title)
                ax.set_xlabel('Parameter Index')
                ax.set_ylabel('Angle (radians)')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-4, 4)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.debug_dir / 'transformation_analysis.png', dpi=150)
        print(f"\n📊 Saved visualization to {self.debug_dir / 'transformation_analysis.png'}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    debugger = HandFusionDebugger(args.results_dir)
    debugger.run_complete_debug()