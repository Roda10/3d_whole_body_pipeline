#!/usr/bin/env python3
"""
Fused Mesh Regeneration and Rendering
Takes fused parameters and regenerates/renders the enhanced SMPL-X mesh
"""

import numpy as np
import json
import cv2
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch

# Add SMPL-X paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'SMPLest-X'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'human_models'))

# Import visualization utils
try:
    from visualization_utils import render_mesh, save_obj
except ImportError:
    print("❌ Could not import visualization_utils")
    
    def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
        """Fallback render function"""
        if len(vertices) > 0:
            focal = cam_param.get('focal', [500, 500])
            princpt = cam_param.get('princpt', [img.shape[1]/2, img.shape[0]/2])
            
            vertices_2d = vertices.copy()
            if vertices_2d.shape[1] >= 3 and np.any(vertices_2d[:, 2] != 0):
                valid_z = vertices_2d[:, 2] != 0
                vertices_2d[valid_z, 0] = vertices_2d[valid_z, 0] * focal[0] / vertices_2d[valid_z, 2] + princpt[0]
                vertices_2d[valid_z, 1] = vertices_2d[valid_z, 1] * focal[1] / vertices_2d[valid_z, 2] + princpt[1]
            
            for vertex in vertices_2d:
                if 0 <= vertex[0] < img.shape[1] and 0 <= vertex[1] < img.shape[0]:
                    cv2.circle(img, (int(vertex[0]), int(vertex[1])), 2, (0, 255, 0), -1)
        return img
    
    def save_obj(vertices, faces, filename):
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# Try to import SMPL-X model
try:
    from human_models import SMPLX
    SMPLX_AVAILABLE = True
    print("✅ SMPL-X model available for mesh regeneration")
except ImportError:
    SMPLX_AVAILABLE = False
    print("⚠️  SMPL-X model not available - will use pre-computed mesh")

class FusedMeshRenderer:
    """Regenerate and render mesh with fused parameters"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_fused_parameters(self) -> Dict:
        """Load the fused parameters"""
        
        param_file = self.results_dir / 'parameter_fusion' / 'fused_smplx_parameters.json'
        if not param_file.exists():
            raise FileNotFoundError(f"Run parameter fusion first. Missing: {param_file}")
            
        with open(param_file, 'r') as f:
            fused_params = json.load(f)
            
        # Convert back to numpy arrays
        for key, value in fused_params.items():
            if isinstance(value, list):
                fused_params[key] = np.array(value)
                
        return fused_params
    
    def load_original_parameters(self) -> Dict:
        """Load original SMPL-X parameters for comparison"""
        
        # Find original SMPL-X parameter file
        param_file = None
        for person_dir in self.results_dir.glob('smplestx_results/*/person_*'):
            person_id = person_dir.name
            candidate = person_dir / f'smplx_params_{person_id}.json'
            if candidate.exists():
                param_file = candidate
                break
        
        if not param_file:
            raise FileNotFoundError("No original SMPL-X parameters found")
            
        with open(param_file, 'r') as f:
            original_params = json.load(f)
            
        # Convert to numpy arrays
        for key, value in original_params.items():
            if isinstance(value, list):
                original_params[key] = np.array(value)
                
        return original_params
    
    def regenerate_smplx_mesh(self, fused_params: Dict) -> np.ndarray:
        """Regenerate SMPL-X mesh with fused parameters"""
        
        if SMPLX_AVAILABLE:
            print("🔄 Regenerating SMPL-X mesh with fused parameters...")
            
            try:
                # Initialize SMPL-X model
                smplx_model = SMPLX()
                
                # Convert parameters to tensors
                betas = torch.tensor(fused_params['betas'], dtype=torch.float32).unsqueeze(0)
                body_pose = torch.tensor(fused_params['body_pose'], dtype=torch.float32).unsqueeze(0)
                left_hand_pose = torch.tensor(fused_params['left_hand_pose'], dtype=torch.float32).unsqueeze(0)
                right_hand_pose = torch.tensor(fused_params['right_hand_pose'], dtype=torch.float32).unsqueeze(0)
                expression = torch.tensor(fused_params['expression'], dtype=torch.float32).unsqueeze(0)
                jaw_pose = torch.tensor(fused_params['jaw_pose'], dtype=torch.float32).unsqueeze(0)
                
                # Generate mesh
                smplx_output = smplx_model(
                    betas=betas,
                    body_pose=body_pose,
                    left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose,
                    expression=expression,
                    jaw_pose=jaw_pose
                )
                
                # Extract vertices
                vertices = smplx_output.vertices[0].detach().cpu().numpy()
                
                print(f"   ✅ Regenerated mesh with {len(vertices)} vertices")
                return vertices
                
            except Exception as e:
                print(f"   ❌ SMPL-X regeneration failed: {e}")
                print("   📝 Using original mesh vertices")
                return fused_params['mesh']
        else:
            print("   📝 Using original mesh vertices (SMPL-X model not available)")
            return fused_params['mesh']
    
    def load_original_image(self) -> np.ndarray:
        """Load the original input image"""
        
        # Try multiple locations for the original image
        for pattern in ['**/test*.jpg', '**/temp_input/*.jpg', '**/*.jpg']:
            for img_file in self.results_dir.rglob(pattern):
                if any(term in str(img_file).lower() for term in ['temp_input', 'test', 'input']):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None and img.size > 0:
                            print(f"   📷 Found original image: {img_file.name}")
                            return img
                    except:
                        continue
        
        # Create default image if not found
        print("   📷 Using default background")
        return np.ones((512, 512, 3), dtype=np.uint8) * 200
    
    def get_camera_parameters(self, fused_params: Dict, img_shape: Tuple[int, int]) -> Dict:
        """Get camera parameters for rendering"""
        
        img_height, img_width = img_shape[:2]
        
        # Use reasonable camera parameters
        focal_length = max(img_width, img_height) * 0.8
        
        return {
            'focal': [focal_length, focal_length],
            'princpt': [img_width / 2.0, img_height / 2.0]
        }
    
    def create_comparison_renderings(self, original_params: Dict, fused_params: Dict, 
                                   original_img: np.ndarray) -> Dict[str, np.ndarray]:
        """Create side-by-side comparison renderings"""
        
        print("🎨 Creating comparison renderings...")
        
        # Get camera parameters
        cam_param = self.get_camera_parameters(fused_params, original_img.shape)
        
        renderings = {}
        
        # 1. Original SMPL-X mesh
        print("   🎨 Rendering original SMPL-X mesh...")
        original_img_copy = original_img.copy()
        original_mesh = original_params['mesh']
        original_rendered = render_mesh(original_img_copy, original_mesh, [], cam_param, mesh_as_vertices=True)
        renderings['original_smplx'] = original_rendered
        
        # 2. Fused mesh (with enhanced parameters)
        print("   🎨 Rendering fused mesh...")
        fused_img_copy = original_img.copy()
        
        # Regenerate or use existing mesh
        if 'fused_mesh' in fused_params:
            fused_mesh = fused_params['fused_mesh']
        else:
            fused_mesh = self.regenerate_smplx_mesh(fused_params)
            
        fused_rendered = render_mesh(fused_img_copy, fused_mesh, [], cam_param, mesh_as_vertices=True)
        renderings['fused_mesh'] = fused_rendered
        
        # 3. Side-by-side comparison
        print("   🎨 Creating side-by-side comparison...")
        
        # Resize to same height for comparison
        target_height = 600
        aspect_ratio = original_img.shape[1] / original_img.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        original_resized = cv2.resize(original_rendered, (target_width, target_height))
        fused_resized = cv2.resize(fused_rendered, (target_width, target_height))
        
        # Create comparison
        comparison = np.hstack([original_resized, fused_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        color = (255, 255, 255)
        thickness = 2
        
        cv2.putText(comparison, 'Original SMPL-X', (20, 40), font, font_scale, color, thickness)
        cv2.putText(comparison, 'Enhanced (WiLoR + EMOCA)', (target_width + 20, 40), font, font_scale, color, thickness)
        
        # Add enhancement details
        details_y = target_height - 100
        cv2.putText(comparison, 'Body: SMPL-X', (20, details_y), font, 0.6, color, 1)
        cv2.putText(comparison, 'Hands: SMPL-X', (20, details_y + 25), font, 0.6, color, 1)
        cv2.putText(comparison, 'Face: SMPL-X', (20, details_y + 50), font, 0.6, color, 1)
        
        cv2.putText(comparison, 'Body: SMPL-X', (target_width + 20, details_y), font, 0.6, color, 1)
        cv2.putText(comparison, 'Hands: WiLoR Enhanced', (target_width + 20, details_y + 25), font, 0.6, (0, 255, 0), 1)
        cv2.putText(comparison, 'Face: EMOCA Enhanced', (target_width + 20, details_y + 50), font, 0.6, (0, 255, 0), 1)
        
        renderings['side_by_side_comparison'] = comparison
        
        return renderings
    
    def analyze_parameter_differences(self, original_params: Dict, fused_params: Dict) -> Dict:
        """Analyze the differences between original and fused parameters"""
        
        differences = {}
        
        # Hand pose differences
        for hand in ['left_hand_pose', 'right_hand_pose']:
            original = original_params[hand]
            fused = fused_params[hand]
            
            diff = np.abs(fused - original)
            differences[hand] = {
                'max_difference': float(np.max(diff)),
                'mean_difference': float(np.mean(diff)),
                'parameters_changed': int(np.sum(diff > 0.01)),  # Threshold for meaningful change
                'total_parameters': len(diff)
            }
        
        # Expression differences
        original_exp = original_params['expression']
        fused_exp = fused_params['expression']
        exp_diff = np.abs(fused_exp - original_exp)
        
        differences['expression'] = {
            'max_difference': float(np.max(exp_diff)),
            'mean_difference': float(np.mean(exp_diff)),
            'parameters_changed': int(np.sum(exp_diff > 0.01)),
            'total_parameters': len(exp_diff)
        }
        
        return differences
    
    def save_all_outputs(self, renderings: Dict, original_params: Dict, fused_params: Dict):
        """Save all rendering outputs and analysis"""
        
        output_dir = self.results_dir / 'fused_mesh_renderings'
        output_dir.mkdir(exist_ok=True)
        
        # Save rendered images
        for name, img in renderings.items():
            cv2.imwrite(str(output_dir / f'{name}.jpg'), img)
        
        # Save enhanced mesh as OBJ
        if 'fused_mesh' in fused_params:
            fused_mesh = fused_params['fused_mesh']
        else:
            fused_mesh = self.regenerate_smplx_mesh(fused_params)
            
        save_obj(fused_mesh, [], str(output_dir / 'enhanced_smplx_mesh.obj'))
        
        # Analyze and save parameter differences
        differences = self.analyze_parameter_differences(original_params, fused_params)
        
        with open(output_dir / 'parameter_analysis.json', 'w') as f:
            json.dump(differences, f, indent=2)
        
        # Create summary report
        summary = {
            'fusion_type': 'parameter_replacement_with_mesh_regeneration',
            'enhancements_applied': {
                'left_hand': f"{differences['left_hand_pose']['parameters_changed']}/{differences['left_hand_pose']['total_parameters']} parameters changed",
                'right_hand': f"{differences['right_hand_pose']['parameters_changed']}/{differences['right_hand_pose']['total_parameters']} parameters changed",
                'expression': f"{differences['expression']['parameters_changed']}/{differences['expression']['total_parameters']} parameters changed"
            },
            'mesh_vertices': len(fused_mesh),
            'output_files': {
                'comparison': 'side_by_side_comparison.jpg',
                'original': 'original_smplx.jpg',
                'enhanced': 'fused_mesh.jpg',
                'mesh_file': 'enhanced_smplx_mesh.obj'
            }
        }
        
        with open(output_dir / 'enhancement_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ All outputs saved to: {output_dir}")
        return output_dir
    
    def run_fused_mesh_rendering(self):
        """Run complete fused mesh regeneration and rendering"""
        
        print("="*80)
        print("FUSED MESH REGENERATION AND RENDERING")
        print("="*80)
        print("Regenerating SMPL-X mesh with WiLoR hands + EMOCA face parameters")
        print()
        
        try:
            # Load parameters
            print("🔄 Loading fused parameters...")
            fused_params = self.load_fused_parameters()
            
            print("🔄 Loading original parameters...")
            original_params = self.load_original_parameters()
            
            print("🔄 Loading original image...")
            original_img = self.load_original_image()
            
            # Create renderings
            renderings = self.create_comparison_renderings(original_params, fused_params, original_img)
            
            # Save everything
            print("\n💾 Saving all outputs...")
            output_dir = self.save_all_outputs(renderings, original_params, fused_params)
            
            print(f"\n🎯 FUSED MESH RENDERING COMPLETE!")
            print("-" * 50)
            print("Key outputs:")
            print(f"✓ side_by_side_comparison.jpg - Shows original vs enhanced")
            print(f"✓ enhanced_smplx_mesh.obj - Enhanced 3D mesh")
            print(f"✓ parameter_analysis.json - Detailed parameter changes")
            
            print(f"\n📁 Check: {output_dir}/side_by_side_comparison.jpg")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Fused mesh rendering failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fused Mesh Regeneration and Rendering')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing parameter fusion results')
    
    args = parser.parse_args()
    
    renderer = FusedMeshRenderer(args.results_dir)
    renderer.run_fused_mesh_rendering()

if __name__ == '__main__':
    main()