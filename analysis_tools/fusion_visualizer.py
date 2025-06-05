#!/usr/bin/env python3
"""
Fusion Results Visualizer
Creates comprehensive visualizations comparing original vs fused parameters and meshes
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import cv2
import argparse
from typing import Dict, Tuple, Optional

class FusionVisualizer:
    """Visualizes fusion results with side-by-side comparisons"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.fusion_dir = self.results_dir / 'fusion_results'
        
    def load_fusion_data(self) -> Tuple[Dict, Dict, Optional[np.ndarray], Optional[np.ndarray]]:
        """Load original and fused data for comparison"""
        print("📥 Loading fusion data for visualization...")
        
        # Load original SMPLest-X parameters
        original_params = None
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f:
                original_params = json.load(f)
            break
        
        # Load fused parameters
        fused_params = None
        fused_file = self.fusion_dir / 'fused_parameters.json'
        if fused_file.exists():
            with open(fused_file, 'r') as f:
                fused_params = json.load(f)
        
        # Load meshes
        original_mesh = None
        if original_params and 'mesh' in original_params:
            original_mesh = np.array(original_params['mesh'])
        
        enhanced_mesh = None
        mesh_file = self.fusion_dir / 'enhanced_mesh.npy'
        if mesh_file.exists():
            enhanced_mesh = np.load(mesh_file)
        
        if original_params is None or fused_params is None:
            raise FileNotFoundError("Required fusion data not found")
        
        print(f"   ✅ Original parameters: {len(original_params)} types")
        print(f"   ✅ Fused parameters: {len(fused_params)} types")
        print(f"   ✅ Original mesh: {original_mesh.shape if original_mesh is not None else 'None'}")
        print(f"   ✅ Enhanced mesh: {enhanced_mesh.shape if enhanced_mesh is not None else 'None'}")
        
        return original_params, fused_params, original_mesh, enhanced_mesh
    
    def create_parameter_comparison_plot(self, original_params: Dict, fused_params: Dict):
        """Create detailed parameter comparison visualizations"""
        print("📊 Creating parameter comparison plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Parameter Fusion Comparison: Original vs Enhanced', fontsize=16, fontweight='bold')
        
        # Parameters to compare
        param_configs = [
            ('left_hand_pose', 'Left Hand Pose (45D)', 0, 0),
            ('right_hand_pose', 'Right Hand Pose (45D)', 0, 1),
            ('expression', 'Facial Expression (10D)', 0, 2),
            ('body_pose', 'Body Pose (63D)', 1, 0),
            ('betas', 'Body Shape (10D)', 1, 1),
            ('root_pose', 'Root Pose (3D)', 1, 2)
        ]
        
        for param_name, title, row, col in param_configs:
            ax = axes[row, col]
            
            # Get parameter arrays
            orig_param = np.array(original_params[param_name])
            fused_param = np.array(fused_params[param_name])
            
            # Create parameter index
            param_indices = np.arange(len(orig_param))
            
            # Plot comparison
            ax.plot(param_indices, orig_param, 'b-', label='Original (SMPLest-X)', linewidth=2, alpha=0.7)
            ax.plot(param_indices, fused_param, 'r-', label='Fused (Enhanced)', linewidth=2, alpha=0.7)
            
            # Highlight differences
            diff_mask = np.abs(orig_param - fused_param) > 0.01
            if np.any(diff_mask):
                ax.scatter(param_indices[diff_mask], fused_param[diff_mask], 
                          c='red', s=30, marker='o', alpha=0.8, label='Significant Changes')
            
            # Calculate metrics
            diff_norm = np.linalg.norm(orig_param - fused_param)
            max_change = np.abs(orig_param - fused_param).max()
            
            ax.set_title(f'{title}\nDiff Norm: {diff_norm:.4f}, Max Change: {max_change:.4f}')
            ax.set_xlabel('Parameter Index')
            ax.set_ylabel('Parameter Value')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Color background based on change magnitude
            if diff_norm > 0.1:
                ax.set_facecolor('#ffe6e6')  # Light red for significant changes
            elif diff_norm > 0.01:
                ax.set_facecolor('#fff2e6')  # Light orange for moderate changes
            else:
                ax.set_facecolor('#f0f8ff')  # Light blue for minimal changes
        
        plt.tight_layout()
        plt.savefig(self.fusion_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Parameter comparison plot saved")
    
    def create_mesh_comparison_3d(self, original_mesh: Optional[np.ndarray], 
                                 enhanced_mesh: Optional[np.ndarray]):
        """Create 3D mesh comparison visualization"""
        if original_mesh is None or enhanced_mesh is None:
            print("⚠️  Cannot create mesh comparison - missing mesh data")
            return
        
        print("🎯 Creating 3D mesh comparison...")
        
        fig = plt.figure(figsize=(20, 10))
        
        # Subsample meshes for visualization (too many points slow down plotting)
        sample_size = min(2000, original_mesh.shape[0])
        sample_indices = np.random.choice(original_mesh.shape[0], sample_size, replace=False)
        
        orig_sample = original_mesh[sample_indices]
        enhanced_sample = enhanced_mesh[sample_indices]
        
        # Original mesh
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2], 
                   c='blue', s=1, alpha=0.6)
        ax1.set_title('Original Mesh (SMPLest-X)')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        
        # Enhanced mesh
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        ax2.scatter(enhanced_sample[:, 0], enhanced_sample[:, 1], enhanced_sample[:, 2], 
                   c='red', s=1, alpha=0.6)
        ax2.set_title('Enhanced Mesh (Fused)')
        ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
        
        # Difference visualization
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        mesh_diff = enhanced_mesh - original_mesh
        diff_sample = mesh_diff[sample_indices]
        diff_magnitude = np.linalg.norm(diff_sample, axis=1)
        
        scatter = ax3.scatter(orig_sample[:, 0], orig_sample[:, 1], orig_sample[:, 2], 
                             c=diff_magnitude, s=2, alpha=0.7, cmap='hot')
        ax3.set_title('Difference Magnitude')
        ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
        plt.colorbar(scatter, ax=ax3, shrink=0.5)
        
        # Coordinate statistics
        ax4 = fig.add_subplot(2, 3, 4)
        coord_names = ['X', 'Y', 'Z']
        orig_stats = [original_mesh[:, i].std() for i in range(3)]
        enhanced_stats = [enhanced_mesh[:, i].std() for i in range(3)]
        
        x = np.arange(len(coord_names))
        width = 0.35
        
        ax4.bar(x - width/2, orig_stats, width, label='Original', alpha=0.7)
        ax4.bar(x + width/2, enhanced_stats, width, label='Enhanced', alpha=0.7)
        ax4.set_xlabel('Coordinate')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('Coordinate Spread Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(coord_names)
        ax4.legend()
        
        # Centroid comparison
        ax5 = fig.add_subplot(2, 3, 5)
        orig_centroid = original_mesh.mean(axis=0)
        enhanced_centroid = enhanced_mesh.mean(axis=0)
        
        x = np.arange(len(coord_names))
        ax5.bar(x - width/2, orig_centroid, width, label='Original', alpha=0.7)
        ax5.bar(x + width/2, enhanced_centroid, width, label='Enhanced', alpha=0.7)
        ax5.set_xlabel('Coordinate')
        ax5.set_ylabel('Centroid Position')
        ax5.set_title('Mesh Centroid Comparison')
        ax5.set_xticks(x)
        ax5.set_xticklabels(coord_names)
        ax5.legend()
        
        # Overall mesh statistics
        ax6 = fig.add_subplot(2, 3, 6)
        metrics = ['Vertices', 'Volume Est.', 'Surface Area Est.', 'Max Distance']
        
        # Calculate basic metrics
        orig_metrics = [
            original_mesh.shape[0],
            np.abs(np.linalg.det(np.cov(original_mesh.T))),  # Volume estimate
            np.sum(np.linalg.norm(np.diff(original_mesh, axis=0), axis=1)),  # Surface estimate
            np.linalg.norm(original_mesh.max(axis=0) - original_mesh.min(axis=0))  # Bounding box diagonal
        ]
        
        enhanced_metrics = [
            enhanced_mesh.shape[0],
            np.abs(np.linalg.det(np.cov(enhanced_mesh.T))),
            np.sum(np.linalg.norm(np.diff(enhanced_mesh, axis=0), axis=1)),
            np.linalg.norm(enhanced_mesh.max(axis=0) - enhanced_mesh.min(axis=0))
        ]
        
        # Normalize metrics for comparison
        max_vals = [max(orig_metrics[i], enhanced_metrics[i]) for i in range(len(metrics))]
        norm_orig = [orig_metrics[i] / max_vals[i] for i in range(len(metrics))]
        norm_enhanced = [enhanced_metrics[i] / max_vals[i] for i in range(len(metrics))]
        
        x = np.arange(len(metrics))
        ax6.bar(x - width/2, norm_orig, width, label='Original', alpha=0.7)
        ax6.bar(x + width/2, norm_enhanced, width, label='Enhanced', alpha=0.7)
        ax6.set_xlabel('Metric')
        ax6.set_ylabel('Normalized Value')
        ax6.set_title('Mesh Characteristics')
        ax6.set_xticks(x)
        ax6.set_xticklabels(metrics, rotation=45)
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(self.fusion_dir / 'mesh_comparison_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ 3D mesh comparison saved")
    
    def create_coordinate_transformation_visualization(self):
        """Visualize the coordinate transformation applied to WiLoR data"""
        print("🔄 Creating coordinate transformation visualization...")
        
        # Load coordinate analysis
        coord_file = self.results_dir / 'coordinate_analysis_summary.json'
        if not coord_file.exists():
            print("   ⚠️  Coordinate analysis not found, skipping transformation viz")
            return
        
        with open(coord_file, 'r') as f:
            coord_analysis = json.load(f)
        
        # Load WiLoR data
        wilor_data = None
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f:
                wilor_data = json.load(f)
            break
        
        if wilor_data is None:
            print("   ⚠️  WiLoR data not found, skipping transformation viz")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Coordinate Transformation Visualization', fontsize=16, fontweight='bold')
        
        # Extract transformation parameters
        scale_factor = coord_analysis['transformation_parameters']['scale_factor']
        translation = np.array(coord_analysis['transformation_parameters']['translation_vector'])
        
        # Get WiLoR hand data
        for hand in wilor_data.get('hands', []):
            if 'vertices_3d' in hand:
                original_vertices = np.array(hand['vertices_3d'])
                transformed_vertices = original_vertices * scale_factor + translation
                
                hand_type = hand.get('hand_type', 'unknown')
                color = 'blue' if hand_type == 'left' else 'red'
                
                # 2D projections of transformation
                # XY plane
                ax = axes[0, 0]
                ax.scatter(original_vertices[:, 0], original_vertices[:, 1], 
                          c=color, alpha=0.5, s=1, label=f'Original {hand_type}')
                ax.scatter(transformed_vertices[:, 0], transformed_vertices[:, 1], 
                          c=color, alpha=0.8, s=1, marker='x', label=f'Transformed {hand_type}')
                ax.set_title('XY Projection')
                ax.set_xlabel('X'); ax.set_ylabel('Y')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # XZ plane
                ax = axes[0, 1]
                ax.scatter(original_vertices[:, 0], original_vertices[:, 2], 
                          c=color, alpha=0.5, s=1, label=f'Original {hand_type}')
                ax.scatter(transformed_vertices[:, 0], transformed_vertices[:, 2], 
                          c=color, alpha=0.8, s=1, marker='x', label=f'Transformed {hand_type}')
                ax.set_title('XZ Projection')
                ax.set_xlabel('X'); ax.set_ylabel('Z')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # YZ plane
                ax = axes[1, 0]
                ax.scatter(original_vertices[:, 1], original_vertices[:, 2], 
                          c=color, alpha=0.5, s=1, label=f'Original {hand_type}')
                ax.scatter(transformed_vertices[:, 1], transformed_vertices[:, 2], 
                          c=color, alpha=0.8, s=1, marker='x', label=f'Transformed {hand_type}')
                ax.set_title('YZ Projection')
                ax.set_xlabel('Y'); ax.set_ylabel('Z')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Transformation parameters summary
        ax = axes[1, 1]
        ax.text(0.1, 0.8, f'Transformation Parameters:', fontsize=14, fontweight='bold')
        ax.text(0.1, 0.7, f'Scale Factor: {scale_factor:.6f}', fontsize=12)
        ax.text(0.1, 0.6, f'Translation X: {translation[0]:.6f}', fontsize=12)
        ax.text(0.1, 0.5, f'Translation Y: {translation[1]:.6f}', fontsize=12)
        ax.text(0.1, 0.4, f'Translation Z: {translation[2]:.6f}', fontsize=12)
        ax.text(0.1, 0.3, f'Translation Magnitude: {np.linalg.norm(translation):.6f}', fontsize=12)
        ax.text(0.1, 0.1, f'Coordinate System: WiLoR → SMPLest-X', fontsize=12, style='italic')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.fusion_dir / 'coordinate_transformation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ Coordinate transformation visualization saved")
    
    def create_fusion_summary_report(self, original_params: Dict, fused_params: Dict):
        """Create a comprehensive text summary of the fusion results"""
        print("📄 Creating fusion summary report...")
        
        report_path = self.fusion_dir / 'fusion_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("PARAMETER FUSION RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Fusion overview
            f.write("FUSION OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write("✓ Body foundation: SMPLest-X parameters\n")
            f.write("✓ Hand enhancement: WiLoR parameters (coordinate transformed)\n")
            f.write("✓ Expression enhancement: EMOCA parameters (dimensionality mapped)\n")
            f.write("✓ Coordinate transformation applied: 7.854x scale + translation\n\n")
            
            # Parameter changes analysis
            f.write("PARAMETER CHANGES ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            
            key_params = ['left_hand_pose', 'right_hand_pose', 'expression', 'body_pose', 'betas']
            
            for param_name in key_params:
                orig = np.array(original_params[param_name])
                fused = np.array(fused_params[param_name])
                
                diff_norm = np.linalg.norm(orig - fused)
                max_change = np.abs(orig - fused).max()
                mean_change = np.abs(orig - fused).mean()
                significant_changes = np.sum(np.abs(orig - fused) > 0.01)
                
                f.write(f"{param_name.replace('_', ' ').title()}:\n")
                f.write(f"  Difference norm: {diff_norm:.6f}\n")
                f.write(f"  Max change: {max_change:.6f}\n")
                f.write(f"  Mean change: {mean_change:.6f}\n")
                f.write(f"  Significant changes (>0.01): {significant_changes}/{len(orig)}\n")
                
                if param_name in ['left_hand_pose', 'right_hand_pose']:
                    f.write(f"  Source: WiLoR (transformed coordinates)\n")
                elif param_name == 'expression':
                    f.write(f"  Source: EMOCA (50D→10D mapped)\n")
                else:
                    f.write(f"  Source: SMPLest-X (unchanged)\n")
                f.write("\n")
            
            # Coordinate transformation details
            if self.results_dir / 'coordinate_analysis_summary.json' in Path(self.results_dir).glob('coordinate_analysis_summary.json'):
                coord_file = self.results_dir / 'coordinate_analysis_summary.json'
                with open(coord_file, 'r') as coord_f:
                    coord_data = json.load(coord_f)
                
                f.write("COORDINATE TRANSFORMATION DETAILS:\n")
                f.write("-" * 35 + "\n")
                f.write(f"Scale factor: {coord_data['transformation_parameters']['scale_factor']:.6f}\n")
                f.write(f"Translation vector: {coord_data['transformation_parameters']['translation_vector']}\n")
                f.write(f"Transformation type: Similarity transform (scale + translation)\n")
                f.write(f"Source coordinate system: WiLoR hand-centric\n")
                f.write(f"Target coordinate system: SMPLest-X body-centric\n\n")
            
            # Quality assessment
            f.write("FUSION QUALITY ASSESSMENT:\n")
            f.write("-" * 30 + "\n")
            
            # Check if meaningful changes occurred
            hand_changes = (np.linalg.norm(np.array(original_params['left_hand_pose']) - np.array(fused_params['left_hand_pose'])) + 
                          np.linalg.norm(np.array(original_params['right_hand_pose']) - np.array(fused_params['right_hand_pose'])))
            
            expr_changes = np.linalg.norm(np.array(original_params['expression']) - np.array(fused_params['expression']))
            
            f.write(f"Hand pose enhancement: {'✓ Significant' if hand_changes > 0.1 else '⚠ Minimal'} (change: {hand_changes:.4f})\n")
            f.write(f"Expression enhancement: {'✓ Significant' if expr_changes > 0.1 else '⚠ Minimal'} (change: {expr_changes:.4f})\n")
            f.write(f"Body structure preservation: {'✓ Maintained' if np.linalg.norm(np.array(original_params['betas']) - np.array(fused_params['betas'])) < 0.001 else '⚠ Modified'}\n")
            
            f.write(f"\nOverall fusion success: {'✓ Successful' if hand_changes > 0.05 or expr_changes > 0.05 else '⚠ Limited enhancement'}\n")
        
        print(f"   ✅ Summary report saved to: {report_path}")
    
    def run_visualization(self):
        """Execute complete visualization pipeline"""
        print("\n" + "="*60)
        print("📊 FUSION RESULTS VISUALIZATION")
        print("="*60 + "\n")
        
        # Load data
        original_params, fused_params, original_mesh, enhanced_mesh = self.load_fusion_data()
        
        # Create visualizations
        self.create_parameter_comparison_plot(original_params, fused_params)
        self.create_mesh_comparison_3d(original_mesh, enhanced_mesh)
        self.create_coordinate_transformation_visualization()
        self.create_fusion_summary_report(original_params, fused_params)
        
        print("\n" + "="*60)
        print("✅ VISUALIZATION COMPLETE!")
        print("="*60)
        print("\n📊 Generated visualizations:")
        print("   - parameter_comparison.png (parameter changes)")
        print("   - mesh_comparison_3d.png (3D mesh analysis)")
        print("   - coordinate_transformation.png (transformation visualization)")
        print("   - fusion_summary_report.txt (comprehensive report)")
        print(f"\n📁 Visualization directory: {self.fusion_dir}")

def main():
    parser = argparse.ArgumentParser(description='Fusion Results Visualizer')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing fusion results')
    
    args = parser.parse_args()
    
    # Validate input
    results_path = Path(args.results_dir)
    fusion_path = results_path / 'fusion_results'
    
    if not results_path.exists():
        print(f"❌ Error: Results directory not found: {results_path}")
        return
    
    if not fusion_path.exists():
        print(f"❌ Error: Fusion results not found. Run parameter fusion first.")
        return
    
    # Run visualization
    visualizer = FusionVisualizer(args.results_dir)
    visualizer.run_visualization()

if __name__ == '__main__':
    main()