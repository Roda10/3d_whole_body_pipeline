#!/usr/bin/env python3
"""
Deep diagnostic tool to understand WiLoR hand pose format and fix conversion
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import torch

class WiLoRDeepDiagnostic:
    """Comprehensive diagnostic for WiLoR to SMPL-X conversion"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
    def load_and_inspect_wilor(self):
        """Load WiLoR data and inspect its exact structure"""
        print("="*60)
        print("🔍 DEEP WILOR DATA INSPECTION")
        print("="*60)
        
        # Find WiLoR parameter file
        wilor_files = list(self.results_dir.glob('wilor_results/*_parameters.json'))
        if not wilor_files:
            print("❌ No WiLoR parameter files found!")
            return None
            
        wilor_file = wilor_files[0]
        print(f"\n📄 Loading: {wilor_file}")
        
        with open(wilor_file, 'r') as f:
            data = json.load(f)
        
        # Deep inspection
        for i, hand in enumerate(data.get('hands', [])):
            print(f"\n🖐️  Hand {i} ({hand.get('hand_type', 'unknown')} hand):")
            
            if 'mano_parameters' in hand:
                mano = hand['mano_parameters']
                print(f"   Source: {mano.get('source', 'unknown')}")
                
                if 'parameters' in mano:
                    params = mano['parameters']
                    print("\n   Available parameters:")
                    for key, value in params.items():
                        if isinstance(value, dict):
                            print(f"   - {key}: shape={value.get('shape', 'unknown')}, type={value.get('type', 'unknown')}")
                            
                            # Inspect actual values
                            if 'values' in value and key == 'hand_pose':
                                vals = np.array(value['values'])
                                print(f"     Raw shape: {vals.shape}")
                                print(f"     Min: {vals.min():.3f}, Max: {vals.max():.3f}")
                                print(f"     First few values: {vals.flatten()[:6]}")
                                
                                # Check if it's rotation matrices
                                if vals.size == 135:  # 15*3*3
                                    mats = vals.reshape(15, 3, 3)
                                    print(f"\n     Checking rotation matrix properties:")
                                    for j in range(min(3, 15)):  # Check first 3 joints
                                        det = np.linalg.det(mats[j])
                                        print(f"     Joint {j} det: {det:.3f}")
                                        print(f"     Joint {j} matrix:\n{mats[j]}")
        
        return data
    
    def test_conversion_methods(self, wilor_data):
        """Test different conversion methods to find the correct one"""
        print("\n" + "="*60)
        print("🧪 TESTING CONVERSION METHODS")
        print("="*60)
        
        results = {}
        
        for hand in wilor_data.get('hands', []):
            hand_type = hand.get('hand_type', 'unknown')
            print(f"\n🖐️  Testing {hand_type} hand conversions:")
            
            if 'mano_parameters' not in hand:
                continue
                
            params = hand['mano_parameters'].get('parameters', {})
            if 'hand_pose' not in params:
                continue
                
            # Get the raw values
            raw_values = np.array(params['hand_pose']['values'])
            print(f"   Raw values shape: {raw_values.shape}")
            
            # Method 1: Direct interpretation as rotation matrices
            print("\n   Method 1: Direct rotation matrices → axis-angle")
            try:
                if raw_values.size == 135:
                    rot_mats = raw_values.reshape(15, 3, 3)
                    axis_angles_m1 = []
                    
                    for i in range(15):
                        # Direct conversion
                        rot = R.from_matrix(rot_mats[i])
                        aa = rot.as_rotvec()
                        axis_angles_m1.append(aa)
                        if i < 3:  # Show first 3
                            print(f"     Joint {i}: {aa}")
                    
                    results[f'{hand_type}_method1'] = np.array(axis_angles_m1).flatten()
            except Exception as e:
                print(f"     ❌ Failed: {e}")
            
            # Method 2: Check if they're already axis-angle
            print("\n   Method 2: Interpret as axis-angle (3x3 reshaping)")
            try:
                if raw_values.size == 135:
                    # Maybe it's 15 * 9 values that represent 3x3 for visualization?
                    # Or maybe it's 45 * 3?
                    if raw_values.size % 45 == 0:
                        # Could be 45 axis-angle values stored differently
                        aa_reshaped = raw_values.reshape(-1, 3)[:15]  # Take first 15
                        print(f"     Reshaped to: {aa_reshaped.shape}")
                        for i in range(min(3, len(aa_reshaped))):
                            print(f"     Joint {i}: {aa_reshaped[i]}")
                        results[f'{hand_type}_method2'] = aa_reshaped.flatten()
            except Exception as e:
                print(f"     ❌ Failed: {e}")
            
            # Method 3: Check rot6d format (6 values per joint)
            print("\n   Method 3: Check if rot6d format")
            try:
                if raw_values.size == 90:  # 15 * 6
                    print("     Detected rot6d format!")
                    # This would need rot6d_to_rotmat conversion
                    results[f'{hand_type}_method3'] = "rot6d_detected"
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                
            # Method 4: Transpose matrices before conversion
            print("\n   Method 4: Transposed rotation matrices")
            try:
                if raw_values.size == 135:
                    rot_mats = raw_values.reshape(15, 3, 3)
                    axis_angles_m4 = []
                    
                    for i in range(15):
                        # Transpose before conversion
                        rot = R.from_matrix(rot_mats[i].T)
                        aa = rot.as_rotvec()
                        axis_angles_m4.append(aa)
                        if i < 3:
                            print(f"     Joint {i}: {aa}")
                    
                    results[f'{hand_type}_method4'] = np.array(axis_angles_m4).flatten()
            except Exception as e:
                print(f"     ❌ Failed: {e}")
        
        return results
    
    def compare_with_smplx(self):
        """Compare with SMPL-X hand pose format"""
        print("\n" + "="*60)
        print("📊 SMPL-X HAND POSE ANALYSIS")
        print("="*60)
        
        # Load SMPL-X params
        smplx_files = list(self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'))
        if not smplx_files:
            print("❌ No SMPL-X files found!")
            return
            
        with open(smplx_files[0], 'r') as f:
            smplx_data = json.load(f)
        
        print("\n🤚 SMPL-X Hand Parameters:")
        for hand in ['left_hand_pose', 'right_hand_pose']:
            pose = np.array(smplx_data[hand])
            print(f"\n   {hand}:")
            print(f"   - Shape: {pose.shape}")
            print(f"   - Range: [{pose.min():.3f}, {pose.max():.3f}]")
            print(f"   - First 6 values: {pose[:6]}")
            
            # Analyze the values
            pose_3d = pose.reshape(-1, 3)
            norms = np.linalg.norm(pose_3d, axis=1)
            print(f"   - Rotation magnitudes: min={norms.min():.3f}, max={norms.max():.3f}")
            print(f"   - Zero rotations: {np.sum(norms < 0.01)}/15")
    
    def visualize_hand_poses(self, conversions):
        """Visualize different conversion results"""
        print("\n" + "="*60)
        print("📈 VISUALIZING CONVERSION RESULTS")
        print("="*60)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        method_names = ['method1', 'method2', 'method4']  # Skip method3 if rot6d
        
        for idx, (key, values) in enumerate(conversions.items()):
            if isinstance(values, str) or idx >= 4:
                continue
                
            ax = axes[idx]
            
            # Plot rotation magnitudes
            if values.size >= 45:
                pose_3d = values[:45].reshape(-1, 3)
                magnitudes = np.linalg.norm(pose_3d, axis=1)
                
                ax.bar(range(len(magnitudes)), magnitudes)
                ax.set_title(f'{key} - Rotation Magnitudes')
                ax.set_xlabel('Joint Index')
                ax.set_ylabel('Rotation Magnitude (rad)')
                ax.axhline(y=np.pi/2, color='r', linestyle='--', alpha=0.5, label='π/2')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'hand_conversion_comparison.png', dpi=150)
        plt.close()
        print(f"   ✅ Saved visualization to hand_conversion_comparison.png")
    
    def suggest_fix(self, wilor_data, test_results):
        """Suggest the best conversion approach"""
        print("\n" + "="*60)
        print("💡 RECOMMENDED FIX")
        print("="*60)
        
        print("\nBased on the analysis, here's what I found:")
        print("\n1. WiLoR outputs rotation matrices in (15,3,3) format")
        print("2. These need to be converted to axis-angle for SMPL-X")
        print("3. The main issue is likely:")
        print("   - Joint ordering mismatch")
        print("   - Need to check if matrices are transposed")
        print("   - May need to apply per-joint corrections")
        
        print("\n🔧 Recommended approach:")
        print("1. Use Method 1 (direct rotation matrix conversion)")
        print("2. Check matrix validity with determinant")
        print("3. Apply joint-specific corrections if needed")
        print("4. Verify with visualization")
    
    def run_diagnostic(self):
        """Run complete diagnostic"""
        # Load data
        wilor_data = self.load_and_inspect_wilor()
        if not wilor_data:
            return
        
        # Test conversions
        test_results = self.test_conversion_methods(wilor_data)
        
        # Compare with SMPL-X
        self.compare_with_smplx()
        
        # Visualize
        if test_results:
            self.visualize_hand_poses(test_results)
        
        # Suggest fix
        self.suggest_fix(wilor_data, test_results)
        
        return wilor_data, test_results


# Quick test function
def diagnose_hand_issue(results_dir: str):
    """Run diagnostic on hand pose conversion issue"""
    diagnostic = WiLoRDeepDiagnostic(results_dir)
    wilor_data, test_results = diagnostic.run_diagnostic()
    
    # Return the most promising conversion
    if test_results:
        # Check which method produces reasonable values
        for method_key, values in test_results.items():
            if isinstance(values, np.ndarray) and values.size >= 45:
                pose_3d = values[:45].reshape(-1, 3)
                magnitudes = np.linalg.norm(pose_3d, axis=1)
                
                # Check if magnitudes are reasonable (mostly < π)
                if np.sum(magnitudes < np.pi) > 10:  # Most joints < 180 degrees
                    print(f"\n✅ {method_key} looks promising!")
                    print(f"   Average rotation: {magnitudes.mean():.3f} rad")
                    print(f"   Max rotation: {magnitudes.max():.3f} rad")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_hand_issue(sys.argv[1])
    else:
        print("Usage: python wilor_deep_diagnostic.py /path/to/results_dir")