#!/usr/bin/env python3
"""
Diagnostic tool to analyze hand parameter fusion issues
This will help us understand why the hands aren't working
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HandFusionDiagnostics:
    """Diagnose issues with hand parameter fusion"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.fusion_dir = self.results_dir / 'fusion_results'
        self.diagnostics_dir = self.fusion_dir / 'diagnostics'
        self.diagnostics_dir.mkdir(exist_ok=True)
        
    def load_all_parameters(self):
        """Load original and fused parameters"""
        print("📥 Loading parameters for diagnosis...")
        
        # Load fused parameters
        fused_path = self.fusion_dir / 'fused_parameters.json'
        if fused_path.exists():
            with open(fused_path, 'r') as f:
                self.fused_params = json.load(f)
            print("   ✅ Loaded fused parameters")
        else:
            print("   ❌ No fused parameters found!")
            self.fused_params = None
            
        # Load original SMPLest-X parameters
        self.smplx_params = None
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f:
                self.smplx_params = json.load(f)
                print("   ✅ Loaded SMPLest-X parameters")
                break
                
        # Load WiLoR parameters
        self.wilor_params = {'left': None, 'right': None}
        for hand in ['left', 'right']:
            param_file = self.results_dir / f'wilor_results/{hand}_hand_parameters.json'
            if param_file.exists():
                with open(param_file, 'r') as f:
                    self.wilor_params[hand] = json.load(f)
                print(f"   ✅ Loaded WiLoR {hand} hand")
                
    def analyze_parameter_differences(self):
        """Analyze differences between original and fused parameters"""
        print("\n📊 PARAMETER DIFFERENCE ANALYSIS")
        print("="*60)
        
        if not self.fused_params or not self.smplx_params:
            print("❌ Missing parameters for comparison")
            return
            
        report = []
        
        # Analyze hand poses
        for hand in ['left_hand_pose', 'right_hand_pose']:
            original = np.array(self.smplx_params.get(hand, []))
            fused = np.array(self.fused_params.get(hand, []))
            
            if len(original) > 0 and len(fused) > 0:
                diff = np.abs(fused - original)
                report.append(f"\n{hand.upper()}:")
                report.append(f"  Original shape: {original.shape}")
                report.append(f"  Fused shape: {fused.shape}")
                report.append(f"  Mean difference: {diff.mean():.6f}")
                report.append(f"  Max difference: {diff.max():.6f}")
                report.append(f"  Min difference: {diff.min():.6f}")
                report.append(f"  Zero differences: {np.sum(diff < 1e-6)} / {len(diff)}")
                
                # Check if parameters actually changed
                if diff.mean() < 0.001:
                    report.append("  ⚠️  WARNING: Parameters barely changed!")
                else:
                    report.append("  ✅ Parameters show significant change")
                    
                # Check parameter ranges
                report.append(f"  Original range: [{original.min():.3f}, {original.max():.3f}]")
                report.append(f"  Fused range: [{fused.min():.3f}, {fused.max():.3f}]")
                
                # Check for extreme values
                if np.abs(fused).max() > np.pi * 2:
                    report.append("  ⚠️  WARNING: Extreme rotation values detected!")
                    
        # Print report
        for line in report:
            print(line)
            
        # Save report
        report_path = self.diagnostics_dir / 'parameter_analysis.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        print(f"\n💾 Report saved to: {report_path}")
        
    def visualize_pose_parameters(self):
        """Visualize hand pose parameters as graphs"""
        print("\n📈 Creating parameter visualizations...")
        
        if not self.fused_params or not self.smplx_params:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Hand Pose Parameter Analysis', fontsize=16)
        
        for idx, hand in enumerate(['left_hand_pose', 'right_hand_pose']):
            original = np.array(self.smplx_params.get(hand, []))
            fused = np.array(self.fused_params.get(hand, []))
            
            if len(original) == 0 or len(fused) == 0:
                continue
                
            # Plot original vs fused
            ax1 = axes[idx, 0]
            x = np.arange(len(original))
            ax1.plot(x, original, 'b-', label='Original', alpha=0.7)
            ax1.plot(x, fused, 'r-', label='Fused', alpha=0.7)
            ax1.set_title(f'{hand.replace("_", " ").title()}')
            ax1.set_xlabel('Parameter Index')
            ax1.set_ylabel('Value (radians)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot differences
            ax2 = axes[idx, 1]
            diff = fused - original
            ax2.bar(x, diff, color='green' if diff.mean() > 0.01 else 'red', alpha=0.7)
            ax2.set_title(f'{hand.replace("_", " ").title()} - Differences')
            ax2.set_xlabel('Parameter Index')
            ax2.set_ylabel('Difference (radians)')
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        viz_path = self.diagnostics_dir / 'parameter_visualization.png'
        plt.savefig(viz_path, dpi=150)
        plt.close()
        print(f"   ✅ Visualization saved to: {viz_path}")
        
    def check_wilor_data_integrity(self):
        """Check if WiLoR data is being loaded correctly"""
        print("\n🔍 Checking WiLoR data integrity...")
        
        report = []
        
        for hand in ['left', 'right']:
            report.append(f"\n{hand.upper()} HAND WILOR DATA:")
            
            if self.wilor_params[hand]:
                # Check MANO parameters
                mano_params = self.wilor_params[hand].get('mano_parameters', {})
                
                if 'hand_pose' in mano_params:
                    pose = np.array(mano_params['hand_pose'])
                    report.append(f"  Hand pose shape: {pose.shape}")
                    report.append(f"  Hand pose range: [{pose.min():.3f}, {pose.max():.3f}]")
                    report.append(f"  Hand pose mean: {pose.mean():.3f}")
                    
                    # Check if it's the flattened 45-dim or full 48-dim
                    if pose.shape[-1] == 48:
                        report.append("  ⚠️  Full 48-dim pose (includes global rotation)")
                    elif pose.shape[-1] == 45:
                        report.append("  ✅ Correct 45-dim pose (15 joints × 3)")
                    else:
                        report.append(f"  ❌ Unexpected pose dimension: {pose.shape[-1]}")
                else:
                    report.append("  ❌ No hand_pose found in MANO parameters!")
                    
                # Check 3D joints
                if 'cam_3d_joints' in self.wilor_params[hand]:
                    joints = np.array(self.wilor_params[hand]['cam_3d_joints'])
                    report.append(f"  3D joints shape: {joints.shape}")
                    report.append(f"  3D joints range: [{joints.min():.3f}, {joints.max():.3f}]")
            else:
                report.append("  ❌ No WiLoR data found!")
                
        # Print and save report
        for line in report:
            print(line)
            
        report_path = self.diagnostics_dir / 'wilor_data_check.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
    def analyze_coordinate_transformation(self):
        """Analyze if coordinate transformations are being applied"""
        print("\n🔄 Analyzing coordinate transformations...")
        
        # Load coordinate analysis if available
        coord_file = self.results_dir / 'coordinate_analysis_summary.json'
        if coord_file.exists():
            with open(coord_file, 'r') as f:
                coord_data = json.load(f)
                
            scale = coord_data['transformation_parameters']['scale_factor']
            trans = coord_data['transformation_parameters']['translation_vector']
            
            print(f"  Scale factor: {scale:.4f}")
            print(f"  Translation: {trans}")
            
            # Check if scale is being applied
            print("\n  Checking if transformations are applied in fusion...")
            
            # This would need access to the actual fusion code to verify
            # For now, we can check the mesh vertices if available
            
    def run_full_diagnostics(self):
        """Run all diagnostic checks"""
        print("🏥 HAND FUSION DIAGNOSTICS")
        print("="*60)
        
        self.load_all_parameters()
        self.analyze_parameter_differences()
        self.visualize_pose_parameters()
        self.check_wilor_data_integrity()
        self.analyze_coordinate_transformation()
        
        print("\n✅ Diagnostics complete!")
        print(f"📁 Results saved to: {self.diagnostics_dir}")
        

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnose hand fusion issues')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing pipeline results')
    args = parser.parse_args()
    
    diagnostics = HandFusionDiagnostics(args.results_dir)
    diagnostics.run_full_diagnostics()
    

if __name__ == '__main__':
    main()