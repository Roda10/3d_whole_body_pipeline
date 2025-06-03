#!/usr/bin/env python3
"""
MANO Parameter Extraction Verifier
Validates that WiLoR MANO parameters are properly extracted and formatted
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class MANOExtractionVerifier:
    """Verifies MANO parameter extraction from WiLoR outputs"""
    
    def __init__(self, wilor_results_path: str):
        self.results_path = Path(wilor_results_path)
        
    def load_wilor_parameters(self) -> Dict:
        """Load WiLoR parameter file"""
        param_files = list(self.results_path.glob('*_parameters.json'))
        
        if not param_files:
            raise FileNotFoundError(f"No WiLoR parameter files found in {self.results_path}")
        
        param_file = param_files[0]  # Use first file found
        print(f"📄 Loading: {param_file}")
        
        with open(param_file, 'r') as f:
            return json.load(f)
    
    def verify_mano_structure(self, params: Dict) -> Dict:
        """Verify MANO parameter structure"""
        verification = {
            'extraction_successful': False,
            'issues_found': [],
            'parameter_summary': {},
            'validation_tests': {}
        }
        
        hands = params.get('hands', [])
        if not hands:
            verification['issues_found'].append("No hands found in parameters")
            return verification
        
        print(f"🖐️  Found {len(hands)} hands")
        
        for i, hand in enumerate(hands):
            hand_type = hand.get('hand_type', f'hand_{i}')
            print(f"\n--- Verifying {hand_type} hand ---")
            
            mano_params = hand.get('mano_parameters', {})
            if not mano_params:
                verification['issues_found'].append(f"{hand_type}: No mano_parameters found")
                continue
            
            # Check for extraction errors
            if 'error' in mano_params:
                verification['issues_found'].append(f"{hand_type}: {mano_params['error']}")
                print(f"❌ Error: {mano_params['error']}")
                continue
            
            # Verify parameter extraction success
            success_indicators = [
                'global_orient_axis_angle' in mano_params,
                'hand_pose_axis_angle' in mano_params,
                'betas' in mano_params,
                'source' in mano_params and mano_params['source'] == 'pred_mano_params_extracted'
            ]
            
            if all(success_indicators):
                print(f"✅ MANO parameters successfully extracted")
                verification['extraction_successful'] = True
                
                # Detailed parameter verification
                self._verify_parameter_details(hand_type, mano_params, verification)
            else:
                missing = []
                if 'global_orient_axis_angle' not in mano_params:
                    missing.append('global_orient_axis_angle')
                if 'hand_pose_axis_angle' not in mano_params:
                    missing.append('hand_pose_axis_angle')
                if 'betas' not in mano_params:
                    missing.append('betas')
                
                verification['issues_found'].append(f"{hand_type}: Missing parameters: {missing}")
                print(f"❌ Missing parameters: {missing}")
        
        return verification
    
    def _verify_parameter_details(self, hand_type: str, mano_params: Dict, verification: Dict):
        """Verify detailed parameter structure and values"""
        
        # 1. Check parameter dimensions
        print(f"📊 Parameter dimensions:")
        
        global_orient = np.array(mano_params['global_orient_axis_angle'])
        hand_pose = np.array(mano_params['hand_pose_axis_angle'])
        betas = np.array(mano_params['betas'])
        
        expected_shapes = {
            'global_orient': (1, 3),
            'hand_pose': (15, 3),
            'betas': (10,)
        }
        
        actual_shapes = {
            'global_orient': global_orient.shape,
            'hand_pose': hand_pose.shape,
            'betas': betas.shape
        }
        
        for param_name, expected_shape in expected_shapes.items():
            actual_shape = actual_shapes[param_name]
            if actual_shape == expected_shape:
                print(f"  ✅ {param_name}: {actual_shape} (correct)")
            else:
                print(f"  ❌ {param_name}: {actual_shape} (expected {expected_shape})")
                verification['issues_found'].append(f"{hand_type}: {param_name} wrong shape")
        
        # 2. Check parameter value ranges
        print(f"📈 Parameter value ranges:")
        
        value_checks = {
            'global_orient': global_orient,
            'hand_pose': hand_pose.flatten(),
            'betas': betas
        }
        
        for param_name, values in value_checks.items():
            val_min, val_max = float(values.min()), float(values.max())
            val_mean, val_std = float(values.mean()), float(values.std())
            
            print(f"  📊 {param_name}:")
            print(f"     Range: [{val_min:.4f}, {val_max:.4f}]")
            print(f"     Mean: {val_mean:.4f}, Std: {val_std:.4f}")
            
            # Sanity checks for reasonable ranges
            if param_name in ['global_orient', 'hand_pose']:
                # Axis-angle values should typically be in [-π, π]
                if abs(val_max) > 10 or abs(val_min) > 10:
                    verification['issues_found'].append(f"{hand_type}: {param_name} values seem too large (rotation)")
                    print(f"     ⚠️  Values seem large for rotation parameters")
            
            elif param_name == 'betas':
                # Shape parameters should typically be in [-3, 3]
                if abs(val_max) > 5 or abs(val_min) > 5:
                    verification['issues_found'].append(f"{hand_type}: {param_name} values seem too large (shape)")
                    print(f"     ⚠️  Values seem large for shape parameters")
        
        # 3. Check rotation matrix conversion
        if 'global_orient_rotmat' in mano_params and 'hand_pose_rotmat' in mano_params:
            print(f"🔄 Verifying rotation matrix to axis-angle conversion:")
            
            # Check if rotation matrices are valid (det = 1, orthogonal)
            global_rotmat = np.array(mano_params['global_orient_rotmat'])
            hand_rotmat = np.array(mano_params['hand_pose_rotmat'])
            
            # Test first rotation matrix
            if global_rotmat.shape == (1, 3, 3):
                test_matrix = global_rotmat[0]
                det = np.linalg.det(test_matrix)
                is_orthogonal = np.allclose(test_matrix @ test_matrix.T, np.eye(3), atol=1e-3)
                
                print(f"  📐 Global orient rotation matrix:")
                print(f"     Determinant: {det:.6f} (should be ~1.0)")
                print(f"     Is orthogonal: {is_orthogonal}")
                
                if abs(det - 1.0) > 0.1:
                    verification['issues_found'].append(f"{hand_type}: Invalid rotation matrix determinant")
                if not is_orthogonal:
                    verification['issues_found'].append(f"{hand_type}: Rotation matrix not orthogonal")
        
        # Store summary
        verification['parameter_summary'][hand_type] = {
            'shapes': actual_shapes,
            'value_ranges': {name: [float(vals.min()), float(vals.max())] 
                           for name, vals in value_checks.items()},
            'total_pose_params': int(1 * 3 + 15 * 3),
            'total_shape_params': int(len(betas))
        }
    
    def test_smplx_compatibility(self, params: Dict) -> Dict:
        """Test if parameters are compatible with SMPL-X format"""
        compatibility = {
            'smplx_ready': True,
            'format_issues': [],
            'conversion_needed': []
        }
        
        print(f"\n🔗 Testing SMPL-X compatibility:")
        
        hands = params.get('hands', [])
        for hand in hands:
            hand_type = hand.get('hand_type', 'unknown')
            mano_params = hand.get('mano_parameters', {})
            
            if 'error' in mano_params:
                compatibility['smplx_ready'] = False
                compatibility['format_issues'].append(f"{hand_type}: MANO extraction failed")
                continue
            
            # Check required fields for SMPL-X
            required_fields = ['global_orient_axis_angle', 'hand_pose_axis_angle', 'betas']
            missing_fields = [field for field in required_fields if field not in mano_params]
            
            if missing_fields:
                compatibility['smplx_ready'] = False
                compatibility['format_issues'].append(f"{hand_type}: Missing {missing_fields}")
            
            # Check if axis-angle format is ready
            if 'hand_pose_axis_angle' in mano_params:
                hand_pose_aa = np.array(mano_params['hand_pose_axis_angle'])
                expected_total_params = 15 * 3  # 15 joints × 3 axis-angle components
                actual_total_params = hand_pose_aa.size
                
                if actual_total_params == expected_total_params:
                    print(f"  ✅ {hand_type}: {actual_total_params} pose parameters (correct)")
                else:
                    print(f"  ❌ {hand_type}: {actual_total_params} pose parameters (expected {expected_total_params})")
                    compatibility['smplx_ready'] = False
        
        return compatibility
    
    def create_verification_report(self, verification: Dict, compatibility: Dict):
        """Create visual verification report"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MANO Parameter Extraction Verification Report', fontsize=16)
        
        # Plot 1: Extraction success overview
        ax1 = axes[0, 0]
        
        success_count = 1 if verification['extraction_successful'] else 0
        failure_count = 1 - success_count
        
        ax1.pie([success_count, failure_count], 
                labels=['Success', 'Failed'], 
                colors=['green', 'red'],
                autopct='%1.1f%%')
        ax1.set_title('Extraction Success Rate')
        
        # Plot 2: Parameter dimensions verification
        ax2 = axes[0, 1]
        
        if verification['parameter_summary']:
            hand_names = list(verification['parameter_summary'].keys())
            pose_params = [verification['parameter_summary'][hand]['total_pose_params'] 
                          for hand in hand_names]
            shape_params = [verification['parameter_summary'][hand]['total_shape_params']
                           for hand in hand_names]
            
            x = np.arange(len(hand_names))
            width = 0.35
            
            ax2.bar(x - width/2, pose_params, width, label='Pose Params', alpha=0.8)
            ax2.bar(x + width/2, shape_params, width, label='Shape Params', alpha=0.8)
            
            ax2.set_xlabel('Hands')
            ax2.set_ylabel('Parameter Count')
            ax2.set_title('Parameter Counts per Hand')
            ax2.set_xticks(x)
            ax2.set_xticklabels(hand_names)
            ax2.legend()
        
        # Plot 3: Parameter value ranges
        ax3 = axes[1, 0]
        
        if verification['parameter_summary']:
            param_types = ['global_orient', 'hand_pose', 'betas']
            colors = ['red', 'blue', 'green']
            
            for i, param_type in enumerate(param_types):
                ranges = []
                for hand_data in verification['parameter_summary'].values():
                    if param_type in hand_data['value_ranges']:
                        val_range = hand_data['value_ranges'][param_type]
                        ranges.append(val_range[1] - val_range[0])  # Range span
                
                if ranges:
                    ax3.bar(i, np.mean(ranges), color=colors[i], alpha=0.7, 
                           label=f'{param_type} (avg range)')
            
            ax3.set_xlabel('Parameter Type')
            ax3.set_ylabel('Average Value Range')
            ax3.set_title('Parameter Value Ranges')
            ax3.set_xticks(range(len(param_types)))
            ax3.set_xticklabels(param_types, rotation=45)
            ax3.legend()
        
        # Plot 4: Issues summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create text summary
        summary_text = "VERIFICATION SUMMARY\n" + "="*30 + "\n\n"
        
        if verification['extraction_successful']:
            summary_text += "✅ MANO extraction: SUCCESS\n"
        else:
            summary_text += "❌ MANO extraction: FAILED\n"
        
        if compatibility['smplx_ready']:
            summary_text += "✅ SMPL-X compatibility: READY\n"
        else:
            summary_text += "❌ SMPL-X compatibility: NEEDS WORK\n"
        
        summary_text += f"\nIssues found: {len(verification['issues_found'])}\n"
        
        if verification['issues_found']:
            summary_text += "\nISSUES:\n"
            for issue in verification['issues_found'][:5]:  # Show first 5 issues
                summary_text += f"• {issue}\n"
            if len(verification['issues_found']) > 5:
                summary_text += f"... and {len(verification['issues_found']) - 5} more\n"
        
        if compatibility['format_issues']:
            summary_text += "\nFORMAT ISSUES:\n"
            for issue in compatibility['format_issues']:
                summary_text += f"• {issue}\n"
        
        ax4.text(0.05, 0.95, summary_text, fontsize=10, verticalalignment='top',
                fontfamily='monospace', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save report
        report_path = self.results_path / 'mano_verification_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Verification report saved: {report_path}")
        plt.show()
    
    def run_verification(self):
        """Run complete MANO parameter verification"""
        print("="*60)
        print("MANO PARAMETER EXTRACTION VERIFICATION")
        print("="*60)
        
        try:
            # Load parameters
            params = self.load_wilor_parameters()
            
            # Verify structure
            verification = self.verify_mano_structure(params)
            
            # Test SMPL-X compatibility
            compatibility = self.test_smplx_compatibility(params)
            
            # Create visual report
            self.create_verification_report(verification, compatibility)
            
            # Final verdict
            print(f"\n🎯 FINAL VERDICT:")
            print("="*30)
            
            if verification['extraction_successful'] and compatibility['smplx_ready']:
                print("🎉 SUCCESS: MANO parameters properly extracted and SMPL-X ready!")
                print("✅ You can proceed with parameter fusion")
            elif verification['extraction_successful']:
                print("⚠️  PARTIAL SUCCESS: MANO extracted but needs format fixes")
                print("🔧 Check compatibility issues above")
            else:
                print("❌ FAILURE: MANO parameter extraction failed")
                print("🔧 Check the WiLoR adapter and extraction code")
            
            # Save detailed verification results
            results = {
                'verification': verification,
                'compatibility': compatibility,
                'summary': {
                    'extraction_successful': verification['extraction_successful'],
                    'smplx_ready': compatibility['smplx_ready'],
                    'total_issues': len(verification['issues_found']) + len(compatibility['format_issues'])
                }
            }
            
            results_path = self.results_path / 'mano_verification_results.json'
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n💾 Detailed results saved: {results_path}")
            
        except Exception as e:
            print(f"❌ Verification failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify MANO parameter extraction')
    parser.add_argument('--wilor_results', type=str, required=True,
                       help='Path to WiLoR results directory')
    
    args = parser.parse_args()
    
    verifier = MANOExtractionVerifier(args.wilor_results)
    verifier.run_verification()

if __name__ == '__main__':
    main()