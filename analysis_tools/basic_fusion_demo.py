#!/usr/bin/env python3
"""
Basic Fusion Demo - Simple Coordinate Combination
Demonstrates concept by combining SMPL-X + WiLoR + EMOCA using coordinate transforms
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle

class BasicFusionDemo:
    """Basic fusion demo showing coordinate combination concept"""
    
    def __init__(self, results_dir: str, fusion_analysis_file: str = "fusion_analysis.json"):
        self.results_dir = Path(results_dir)
        self.fusion_analysis = self._load_fusion_analysis(fusion_analysis_file)
        
        # Extract transformation parameters from analysis
        self.wilor_transform = self.fusion_analysis['fusion_strategy']['transformations_needed']['wilor_to_smplx']
        self.translation = np.array(self.wilor_transform['translation'])
        self.scale = self.wilor_transform['scale']
        
    def _load_fusion_analysis(self, filename: str) -> Dict:
        """Load the fusion analysis results"""
        analysis_file = self.results_dir / filename
        with open(analysis_file, 'r') as f:
            return json.load(f)
    
    def load_smplx_data(self) -> Dict:
        """Load SMPL-X foundation data"""
        # Find SMPL-X parameter file
        param_file = None
        for person_dir in self.results_dir.glob('smplestx_results/*/person_*'):
            person_id = person_dir.name
            candidate = person_dir / f'smplx_params_{person_id}.json'
            if candidate.exists():
                param_file = candidate
                break
        
        if not param_file:
            raise FileNotFoundError("No SMPL-X parameter file found")
            
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        return {
            'joints_3d': np.array(params['joints_3d']),
            'translation': np.array(params['translation']),
            'body_pose': np.array(params['body_pose']),
            'left_hand_pose': np.array(params['left_hand_pose']),
            'right_hand_pose': np.array(params['right_hand_pose']),
            'betas': np.array(params['betas']),
            'expression': np.array(params['expression'])
        }
    
    def load_wilor_data(self) -> Dict:
        """Load WiLoR hand data"""
        # Find WiLoR parameter file
        param_file = None
        for candidate in self.results_dir.glob('wilor_results/*_parameters.json'):
            param_file = candidate
            break
            
        if not param_file:
            raise FileNotFoundError("No WiLoR parameter file found")
            
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        # Extract hand data
        hands = []
        for hand_data in params['hands']:
            hands.append({
                'hand_type': hand_data['hand_type'],
                'vertices_3d': np.array(hand_data['vertices_3d']),
                'keypoints_3d': np.array(hand_data['keypoints_3d']),
                'camera_translation': np.array(hand_data['camera_translation'])
            })
            
        return {'hands': hands}
    
    def load_emoca_data(self) -> Dict:
        """Load EMOCA expression data"""
        # Look for EMOCA codes file (try both formats)
        codes_file = None
        
        # First try combined codes.json in various locations
        search_patterns = [
            'emoca_results/*/codes.json',
            'emoca_results/*/*/codes.json',
            'emoca_results/EMOCA_*/test*/codes.json'
        ]
        
        for pattern in search_patterns:
            for candidate in self.results_dir.glob(pattern):
                codes_file = candidate
                break
            if codes_file:
                break
        
        if codes_file and codes_file.exists():
            print(f"   📄 Found codes.json at: {codes_file}")
            with open(codes_file, 'r') as f:
                codes = json.load(f)
            return {
                'shapecode': np.array(codes['shapecode']),
                'expcode': np.array(codes['expcode']),
                'posecode': np.array(codes['posecode'])
            }
        
        # Try individual code files (alternative format)
        individual_patterns = [
            'emoca_results/*/*/',
            'emoca_results/EMOCA_*/test*/',
            'emoca_results/*/'
        ]
        
        for pattern in individual_patterns:
            for emoca_subdir in self.results_dir.glob(pattern):
                print(f"   🔍 Checking directory: {emoca_subdir}")
                
                code_files = {
                    'shapecode': emoca_subdir / 'shape.json',
                    'expcode': emoca_subdir / 'exp.json', 
                    'posecode': emoca_subdir / 'pose.json'
                }
                
                codes = {}
                files_found = 0
                
                for code_type, file_path in code_files.items():
                    if file_path.exists():
                        try:
                            print(f"   📄 Loading {file_path}")
                            with open(file_path, 'r') as f:
                                codes[code_type] = np.array(json.load(f))
                                files_found += 1
                        except Exception as e:
                            print(f"   ⚠️  Error loading {file_path}: {e}")
                
                if files_found >= 2:  # Need at least shape and expression
                    print(f"   ✅ Successfully loaded {files_found} EMOCA code files")
                    return codes
        
        # Debug: List what's actually in the emoca_results directory
        print(f"   🔍 Debug: Contents of emoca_results:")
        for item in (self.results_dir / 'emoca_results').rglob('*'):
            if item.is_file():
                print(f"      📄 {item.relative_to(self.results_dir)}")
        
        raise FileNotFoundError("No EMOCA codes files found. Check debug output above for available files.")
    
    def transform_wilor_to_smplx(self, wilor_coords: np.ndarray) -> np.ndarray:
        """Transform WiLoR coordinates to SMPL-X coordinate system"""
        # Apply scale and translation transformation
        transformed = wilor_coords * self.scale + self.translation
        return transformed
    
    def map_emoca_to_smplx_expression(self, emoca_expcode: np.ndarray) -> np.ndarray:
        """Map EMOCA 50D expression to SMPL-X 10D expression"""
        # Simple dimensionality reduction: take first 10 components and scale
        if len(emoca_expcode) >= 10:
            # Use first 10 components
            smplx_exp = emoca_expcode[:10]
        else:
            # Pad with zeros if less than 10
            smplx_exp = np.zeros(10)
            smplx_exp[:len(emoca_expcode)] = emoca_expcode
        
        # Scale to reasonable SMPL-X expression range (typically [-2, 2])
        emoca_range = np.max(emoca_expcode) - np.min(emoca_expcode)
        if emoca_range > 0:
            smplx_exp = (smplx_exp - np.mean(emoca_expcode)) / emoca_range * 2.0
        
        return smplx_exp
    
    def create_fused_representation(self) -> Dict:
        """Create fused representation combining all three models"""
        print("🔄 Loading model data...")
        
        # Load all model data
        smplx_data = self.load_smplx_data()
        wilor_data = self.load_wilor_data()
        emoca_data = self.load_emoca_data()
        
        print(f"✅ SMPL-X: {smplx_data['joints_3d'].shape[0]} joints")
        print(f"✅ WiLoR: {len(wilor_data['hands'])} hands")
        print(f"✅ EMOCA: {len(emoca_data['expcode'])}D expression")
        
        print("\n🔄 Applying coordinate transformations...")
        
        # Transform WiLoR hands to SMPL-X coordinate system
        transformed_hands = []
        for hand in wilor_data['hands']:
            transformed_vertices = self.transform_wilor_to_smplx(hand['vertices_3d'])
            transformed_keypoints = self.transform_wilor_to_smplx(hand['keypoints_3d'])
            
            transformed_hands.append({
                'hand_type': hand['hand_type'],
                'vertices_3d_original': hand['vertices_3d'],
                'vertices_3d_transformed': transformed_vertices,
                'keypoints_3d_original': hand['keypoints_3d'],
                'keypoints_3d_transformed': transformed_keypoints
            })
        
        print(f"   🖐️  Transformed {len(transformed_hands)} hands to SMPL-X space")
        
        # Map EMOCA expression to SMPL-X expression space
        original_expression = smplx_data['expression']
        enhanced_expression = self.map_emoca_to_smplx_expression(emoca_data['expcode'])
        
        print(f"   😊 Mapped EMOCA {len(emoca_data['expcode'])}D → SMPL-X 10D expression")
        
        # Create fused representation
        fused = {
            'base_model': 'smplx',
            'smplx_foundation': {
                'joints_3d': smplx_data['joints_3d'],
                'translation': smplx_data['translation'],
                'body_pose': smplx_data['body_pose'],
                'betas': smplx_data['betas'],
                'original_expression': original_expression,
                'enhanced_expression': enhanced_expression
            },
            'enhanced_hands': transformed_hands,
            'transformation_applied': {
                'wilor_scale': self.scale,
                'wilor_translation': self.translation.tolist(),
                'emoca_expression_mapping': 'first_10_components_scaled'
            },
            'fusion_stats': {
                'smplx_joints': smplx_data['joints_3d'].shape[0],
                'total_hand_vertices': sum(len(h['vertices_3d_transformed']) for h in transformed_hands),
                'expression_enhancement': f"EMOCA {len(emoca_data['expcode'])}D → SMPL-X 10D"
            }
        }
        
        return fused
    
    def visualize_fusion(self, fused_data: Dict):
        """Visualize the fusion results"""
        fig = plt.figure(figsize=(20, 10))
        
        # Plot 1: Original coordinates comparison
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        smplx_joints = fused_data['smplx_foundation']['joints_3d']
        ax1.scatter(smplx_joints[:, 0], smplx_joints[:, 1], smplx_joints[:, 2], 
                   c='blue', alpha=0.6, s=20, label='SMPL-X Joints')
        ax1.set_title('SMPL-X Foundation\n(Body Joints)')
        ax1.legend()
        
        # Plot 2: Original WiLoR hands
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        colors = ['red', 'orange']
        for i, hand in enumerate(fused_data['enhanced_hands']):
            vertices = hand['vertices_3d_original']
            ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       c=colors[i % 2], alpha=0.6, s=1, 
                       label=f"{hand['hand_type']} hand")
        ax2.set_title('WiLoR Hands\n(Original Coordinates)')
        ax2.legend()
        
        # Plot 3: Transformed WiLoR hands
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        for i, hand in enumerate(fused_data['enhanced_hands']):
            vertices = hand['vertices_3d_transformed']
            ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       c=colors[i % 2], alpha=0.6, s=1, 
                       label=f"{hand['hand_type']} hand")
        ax3.set_title('WiLoR Hands\n(Transformed to SMPL-X Space)')
        ax3.legend()
        
        # Plot 4: Combined view
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        # SMPL-X joints
        ax4.scatter(smplx_joints[:, 0], smplx_joints[:, 1], smplx_joints[:, 2], 
                   c='blue', alpha=0.8, s=30, label='SMPL-X Body')
        # Transformed hands
        for i, hand in enumerate(fused_data['enhanced_hands']):
            vertices = hand['vertices_3d_transformed']
            ax4.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                       c=colors[i % 2], alpha=0.6, s=2, 
                       label=f"{hand['hand_type']} hand")
        ax4.set_title('Fused Representation\n(SMPL-X + Transformed WiLoR)')
        ax4.legend()
        
        # Plot 5: Expression comparison
        ax5 = fig.add_subplot(2, 3, 5)
        original_exp = fused_data['smplx_foundation']['original_expression']
        enhanced_exp = fused_data['smplx_foundation']['enhanced_expression']
        
        x_pos = np.arange(len(original_exp))
        width = 0.35
        
        ax5.bar(x_pos - width/2, original_exp, width, label='Original SMPL-X', alpha=0.7)
        ax5.bar(x_pos + width/2, enhanced_exp, width, label='EMOCA Enhanced', alpha=0.7)
        ax5.set_xlabel('Expression Component')
        ax5.set_ylabel('Value')
        ax5.set_title('Expression Enhancement\n(SMPL-X vs EMOCA-Enhanced)')
        ax5.legend()
        
        # Plot 6: Fusion statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = f"""
        FUSION STATISTICS
        
        Foundation: {fused_data['base_model'].upper()}
        
        Body Joints: {fused_data['fusion_stats']['smplx_joints']}
        Hand Vertices: {fused_data['fusion_stats']['total_hand_vertices']}
        
        Coordinate Transform:
        • Scale: {fused_data['transformation_applied']['wilor_scale']:.2f}x
        • Translation: [{fused_data['transformation_applied']['wilor_translation'][0]:.3f}, 
                      {fused_data['transformation_applied']['wilor_translation'][1]:.3f}, 
                      {fused_data['transformation_applied']['wilor_translation'][2]:.3f}]
        
        Expression: {fused_data['fusion_stats']['expression_enhancement']}
        """
        ax6.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top', 
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'basic_fusion_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_fused_output(self, fused_data: Dict):
        """Save the fused representation"""
        output_file = self.results_dir / 'basic_fusion_output.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = self._make_serializable(fused_data)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"💾 Fused representation saved to: {output_file}")
        
        # Save summary
        summary = {
            'fusion_type': 'basic_coordinate_combination',
            'models_used': ['SMPLest-X', 'WiLoR', 'EMOCA'],
            'coordinate_system': 'SMPL-X',
            'transformations': fused_data['transformation_applied'],
            'statistics': fused_data['fusion_stats'],
            'next_steps': [
                'Optimize hand attachment points',
                'Refine expression mapping',
                'Add anatomical constraints',
                'Implement mesh blending'
            ]
        }
        
        summary_file = self.results_dir / 'fusion_demo_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Fusion summary saved to: {summary_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def run_basic_fusion_demo(self):
        """Run the complete basic fusion demonstration"""
        print("="*80)
        print("BASIC FUSION DEMO - COORDINATE COMBINATION")
        print("="*80)
        print("Concept: Combine SMPL-X + WiLoR + EMOCA using coordinate transforms")
        print()
        
        try:
            # Create fused representation
            fused_data = self.create_fused_representation()
            
            print("\n🎯 FUSION SUCCESSFUL!")
            print("-" * 40)
            print(f"Combined {fused_data['fusion_stats']['smplx_joints']} body joints")
            print(f"Added {fused_data['fusion_stats']['total_hand_vertices']} hand vertices")
            print(f"Enhanced expression: {fused_data['fusion_stats']['expression_enhancement']}")
            
            # Visualize results
            print("\n📊 Creating visualizations...")
            self.visualize_fusion(fused_data)
            
            # Save outputs
            print("\n💾 Saving results...")
            self.save_fused_output(fused_data)
            
            print("\n✅ BASIC FUSION DEMO COMPLETE!")
            print("="*80)
            print("Ready for next steps:")
            print("• Optimize hand-body attachment points")
            print("• Refine expression parameter mapping")
            print("• Add anatomical constraints")
            print("• Implement mesh-level blending")
            
        except Exception as e:
            print(f"❌ Fusion failed: {str(e)}")
            print("Check that all model outputs are available")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Basic Fusion Demo')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing pipeline results and fusion analysis')
    
    args = parser.parse_args()
    
    demo = BasicFusionDemo(args.results_dir)
    demo.run_basic_fusion_demo()

if __name__ == '__main__':
    main()