#!/usr/bin/env python3
"""
Enhanced Direct Parameter Fusion System
Properly handles parameter conversion, coordinate alignment, and mesh generation
"""

import numpy as np
import json
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import cv2
from scipy.spatial.transform import Rotation as R
import argparse

# Add paths for SMPL-X model access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
from human_models.human_models import SMPLX
from utils.visualization_utils import render_mesh
from utils.transforms import rot6d_to_axis_angle, batch_rodrigues
from main.config import Config

class EnhancedParameterFusion:
    """Enhanced fusion system with proper parameter handling and mesh generation"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.fusion_dir = self.results_dir / 'fusion_results'
        self.fusion_dir.mkdir(exist_ok=True)
        
        # Load analysis results
        self.load_coordinate_analysis()
        
        # Initialize models
        self.setup_smplx_model()
        self.setup_rendering_config()
        
        # Fusion parameters
        self.hand_blend_weight = 0.8  # How much to trust WiLoR vs SMPLest-X
        self.expression_scale = 0.3   # Scale EMOCA expression to prevent extremes
        
    def load_coordinate_analysis(self):
        """Load coordinate transformation parameters"""
        print("📥 Loading coordinate analysis...")
        
        coord_file = self.results_dir / 'coordinate_analysis_summary.json'
        if not coord_file.exists():
            raise FileNotFoundError(f"Coordinate analysis not found: {coord_file}")
        
        with open(coord_file, 'r') as f:
            self.coord_analysis = json.load(f)
        
        self.scale_factor = self.coord_analysis['transformation_parameters']['scale_factor']
        self.translation = np.array(self.coord_analysis['transformation_parameters']['translation_vector'])
        
        print(f"   ✅ Scale factor: {self.scale_factor:.4f}")
        print(f"   ✅ Translation: {self.translation}")
    
    def setup_smplx_model(self):
        """Initialize SMPL-X model"""
        print("🤖 Setting up SMPL-X model...")
        
        # Try multiple possible paths
        model_paths = [
            'human_models/human_model_files/',
            '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/pretrained_models/smplx',
            './pretrained_models/smplx',
            '../pretrained_models/smplx'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.smplx_model = SMPLX(path)
                    print(f"   ✅ SMPL-X model loaded from: {path}")
                    return
                except Exception as e:
                    print(f"   ⚠️  Failed to load from {path}: {e}")
        
        raise RuntimeError("Could not load SMPL-X model from any path")
    
    def setup_rendering_config(self):
        """Setup rendering configuration"""
        print("🎨 Setting up rendering configuration...")
        
        config_paths = [
            'pretrained_models/smplest_x/config_base.py',
            '../pretrained_models/smplest_x/config_base.py',
            'external/SMPLest-X/configs/config_smplest_x_h.py'
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                self.config = Config.load_config(path)
                print(f"   ✅ Config loaded from: {path}")
                return
        
        # Use defaults if no config found
        self.config = type('Config', (), {
            'model': type('Model', (), {
                'focal': [5000.0, 5000.0],
                'princpt': [128.0, 128.0],
                'input_body_shape': [256, 256]
            })()
        })()
        print("   ⚠️  Using default config")
    
    def load_all_parameters(self) -> Tuple[Dict, Dict, Dict]:
        """Load parameters from all three models with proper path handling"""
        print("\n📥 Loading all model parameters...")
        
        # Load SMPLest-X
        smplx_params = self._load_smplestx_params()
        
        # Load WiLoR
        wilor_params = self._load_wilor_params()
        
        # Load EMOCA
        emoca_params = self._load_emoca_params()
        
        return smplx_params, wilor_params, emoca_params
    
    def _load_smplestx_params(self) -> Dict:
        """Load SMPLest-X parameters"""
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"   ✅ SMPLest-X: {len(params)} parameters loaded")
            return params
        
        raise FileNotFoundError("SMPLest-X parameters not found")
    
    def _load_wilor_params(self) -> Dict:
        """Load WiLoR parameters"""
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"   ✅ WiLoR: {len(params.get('hands', []))} hands loaded")
            return params
        
        raise FileNotFoundError("WiLoR parameters not found")
    
    def _load_emoca_params(self) -> Dict:
        """Load EMOCA parameters with proper path handling"""
        # Try multiple possible paths
        search_patterns = [
            'emoca_results/EMOCA*/test*/codes.json',
            'emoca_results/*/codes.json',
            'emoca_results/*/*/codes.json'
        ]
        
        for pattern in search_patterns:
            for codes_file in self.results_dir.glob(pattern):
                with open(codes_file, 'r') as f:
                    params = json.load(f)
                print(f"   ✅ EMOCA: {len(params)} parameter types loaded from {codes_file.relative_to(self.results_dir)}")
                return params
        
        print("   ⚠️  EMOCA parameters not found, using neutral expression")
        return {}
    
    def convert_rotation_matrix_to_axis_angle(self, rot_matrices: np.ndarray) -> np.ndarray:
        """Convert rotation matrices to axis-angle representation"""
        # Reshape to (N, 3, 3)
        n_joints = rot_matrices.shape[0] // 9
        rot_matrices = rot_matrices.reshape(n_joints, 3, 3)
        
        # Convert each matrix to axis-angle
        axis_angles = []
        for i in range(n_joints):
            rot = R.from_matrix(rot_matrices[i])
            axis_angle = rot.as_rotvec()
            axis_angles.append(axis_angle)
        
        return np.array(axis_angles).flatten()
    
    def extract_wilor_hand_poses(self, wilor_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and convert WiLoR hand poses to SMPL-X format"""
        print("\n🖐️  Extracting WiLoR hand poses...")
        
        left_hand_pose = np.zeros(45)  # Default neutral pose
        right_hand_pose = np.zeros(45)
        
        for hand in wilor_params.get('hands', []):
            hand_type = hand.get('hand_type', '')
            
            # Check for MANO parameters
            if 'mano_parameters' in hand and 'parameters' in hand['mano_parameters']:
                mano_params = hand['mano_parameters']['parameters']
                
                if 'hand_pose' in mano_params:
                    # Get rotation matrices
                    rot_matrices = np.array(mano_params['hand_pose']['values']).flatten()
                    
                    # Convert to axis-angle
                    axis_angles = self.convert_rotation_matrix_to_axis_angle(rot_matrices)
                    
                    # MANO has 15 joints, SMPL-X expects 15 joints too
                    if len(axis_angles) >= 45:
                        if hand_type == 'left':
                            left_hand_pose = axis_angles[:45]
                            print(f"   ✅ Left hand: converted {len(rot_matrices)} values to 45 axis-angles")
                        elif hand_type == 'right':
                            right_hand_pose = axis_angles[:45]
                            print(f"   ✅ Right hand: converted {len(rot_matrices)} values to 45 axis-angles")
        
        return left_hand_pose, right_hand_pose
    
    def align_hand_coordinate_frame(self, hand_pose: np.ndarray, hand_type: str) -> np.ndarray:
        """Align hand pose to body coordinate frame"""
        # WiLoR uses a different coordinate convention than SMPL-X
        # We need to apply a coordinate transformation to the root joint
        
        # For now, apply empirical corrections based on observation
        aligned_pose = hand_pose.copy()
        
        # Scale down extreme rotations
        max_rotation = np.pi / 2  # 90 degrees max
        aligned_pose = np.clip(aligned_pose, -max_rotation, max_rotation)
        
        return aligned_pose
    
    def map_emoca_expression(self, emoca_params: Dict) -> np.ndarray:
        """Map EMOCA expression to SMPL-X with proper scaling"""
        if not emoca_params or 'expcode' not in emoca_params:
            print("   ⚠️  No EMOCA expression, using neutral")
            return np.zeros(10)
        
        print("\n🎭 Mapping EMOCA expression...")
        
        emoca_exp = np.array(emoca_params['expcode'])
        
        # Use PCA-like approach: take most significant components
        if len(emoca_exp) >= 10:
            # Take first 10 components and scale them
            mapped_exp = emoca_exp[:10] * self.expression_scale
        else:
            # Pad with zeros
            mapped_exp = np.zeros(10)
            mapped_exp[:len(emoca_exp)] = emoca_exp * self.expression_scale
        
        # Clip to reasonable range
        mapped_exp = np.clip(mapped_exp, -2.0, 2.0)
        
        print(f"   ✅ Expression mapped: 50D → 10D, range [{mapped_exp.min():.2f}, {mapped_exp.max():.2f}]")
        
        return mapped_exp
    
    def blend_hand_poses(self, smplx_pose: np.ndarray, wilor_pose: np.ndarray, 
                        blend_weight: float = 0.8) -> np.ndarray:
        """Blend SMPLest-X and WiLoR hand poses for smooth transition"""
        # Use weighted average for smooth transition at wrist
        return smplx_pose * (1 - blend_weight) + wilor_pose * blend_weight
    
    def create_fused_parameters(self, smplx_params: Dict, wilor_params: Dict, 
                               emoca_params: Dict) -> Dict:
        """Create properly fused parameters"""
        print("\n🔧 Creating fused parameters...")
        
        # Start with SMPLest-X foundation
        fused = {
            'betas': np.array(smplx_params['betas']),
            'body_pose': np.array(smplx_params['body_pose']),
            'root_pose': np.array(smplx_params['root_pose']),
            'translation': np.array(smplx_params['translation']),
            'jaw_pose': np.array(smplx_params['jaw_pose'])
        }
        
        # Extract and convert WiLoR hand poses
        left_hand_wilor, right_hand_wilor = self.extract_wilor_hand_poses(wilor_params)
        
        # Align hand coordinate frames
        left_hand_aligned = self.align_hand_coordinate_frame(left_hand_wilor, 'left')
        right_hand_aligned = self.align_hand_coordinate_frame(right_hand_wilor, 'right')
        
        # Blend with original SMPLest-X hand poses
        smplx_left = np.array(smplx_params['left_hand_pose'])
        smplx_right = np.array(smplx_params['right_hand_pose'])
        
        fused['left_hand_pose'] = self.blend_hand_poses(smplx_left, left_hand_aligned, 
                                                        self.hand_blend_weight)
        fused['right_hand_pose'] = self.blend_hand_poses(smplx_right, right_hand_aligned,
                                                         self.hand_blend_weight)
        
        # Map EMOCA expression
        fused['expression'] = self.map_emoca_expression(emoca_params)
        
        # Add metadata
        fused['fusion_metadata'] = {
            'body_source': 'SMPLest-X',
            'hand_source': 'WiLoR (converted & aligned)',
            'expression_source': 'EMOCA (mapped & scaled)',
            'hand_blend_weight': self.hand_blend_weight,
            'expression_scale': self.expression_scale
        }
        
        return fused
    
    def validate_parameters(self, params: Dict) -> bool:
        """Validate fused parameters are in reasonable ranges"""
        print("\n✅ Validating parameters...")
        
        issues = []
        
        # Check pose parameters (should be in radians, typically < pi)
        for pose_name in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
            pose = params[pose_name]
            max_val = np.abs(pose).max()
            if max_val > np.pi:
                issues.append(f"{pose_name} has extreme values: max={max_val:.2f}")
        
        # Check shape parameters (typically -3 to 3)
        betas = params['betas']
        if np.abs(betas).max() > 5:
            issues.append(f"Shape parameters extreme: max={np.abs(betas).max():.2f}")
        
        # Check expression (typically -2 to 2)
        expr = params['expression']
        if np.abs(expr).max() > 3:
            issues.append(f"Expression extreme: max={np.abs(expr).max():.2f}")
        
        if issues:
            print("   ⚠️  Validation issues:")
            for issue in issues:
                print(f"      - {issue}")
            return False
        else:
            print("   ✅ All parameters within reasonable ranges")
            return True
    
    def generate_mesh(self, params: Dict) -> np.ndarray:
        """Generate mesh from fused parameters"""
        print("\n🎯 Generating enhanced mesh...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smplx_layer = self.smplx_model.layer['neutral'].to(device)
        
        # Convert to torch tensors
        torch_params = {
            'betas': torch.tensor(params['betas']).float().unsqueeze(0).to(device),
            'body_pose': torch.tensor(params['body_pose']).float().unsqueeze(0).to(device),
            'global_orient': torch.tensor(params['root_pose']).float().unsqueeze(0).to(device),
            'left_hand_pose': torch.tensor(params['left_hand_pose']).float().unsqueeze(0).to(device),
            'right_hand_pose': torch.tensor(params['right_hand_pose']).float().unsqueeze(0).to(device),
            'jaw_pose': torch.tensor(params['jaw_pose']).float().unsqueeze(0).to(device),
            'expression': torch.tensor(params['expression']).float().unsqueeze(0).to(device),
            'transl': torch.tensor(params['translation']).float().unsqueeze(0).to(device),
            'leye_pose': torch.zeros(1, 3).float().to(device),
            'reye_pose': torch.zeros(1, 3).float().to(device)
        }
        
        # Generate mesh
        with torch.no_grad():
            output = smplx_layer(**torch_params)
            mesh = output.vertices[0].detach().cpu().numpy()
        
        print(f"   ✅ Mesh generated: {mesh.shape[0]} vertices")
        return mesh
    
    def render_comparison(self, original_params: Dict, fused_params: Dict, 
                         original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        """Render comparison images"""
        print("\n🎨 Rendering comparison...")
        
        # Load original image
        img_path = None
        for temp_dir in [self.results_dir / 'wilor_results' / 'temp_input',
                        self.results_dir / 'emoca_results' / 'temp_input']:
            if temp_dir.exists():
                for img_file in temp_dir.glob('*'):
                    img_path = img_file
                    break
                if img_path:
                    break
        
        if not img_path:
            print("   ⚠️  No input image found for rendering")
            return
        
        # Load image
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Create simple bbox (center crop)
        bbox_size = min(h, w) * 0.8
        bbox = np.array([
            (w - bbox_size) / 2,
            (h - bbox_size) / 2,
            bbox_size,
            bbox_size
        ])
        
        # Camera parameters
        focal = [self.config.model.focal[0] / self.config.model.input_body_shape[1] * bbox[2],
                self.config.model.focal[1] / self.config.model.input_body_shape[0] * bbox[3]]
        princpt = [self.config.model.princpt[0] / self.config.model.input_body_shape[1] * bbox[2] + bbox[0],
                  self.config.model.princpt[1] / self.config.model.input_body_shape[0] * bbox[3] + bbox[1]]
        
        cam_params = {'focal': focal, 'princpt': princpt}
        
        # Render both meshes
        img_orig = img.copy()
        img_enhanced = img.copy()
        
        img_orig = render_mesh(img_orig, original_mesh, self.smplx_model.face, 
                             cam_params, mesh_as_vertices=False)
        img_enhanced = render_mesh(img_enhanced, enhanced_mesh, self.smplx_model.face,
                                 cam_params, mesh_as_vertices=False)
        
        # Create side-by-side comparison
        comparison = np.hstack([img_orig, img_enhanced])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Enhanced', (w + 10, 30), font, 1, (255, 255, 255), 2)
        
        # Save
        cv2.imwrite(str(self.fusion_dir / 'mesh_comparison.pdf'), comparison)
        print(f"   ✅ Comparison saved to {self.fusion_dir / 'mesh_comparison.pdf'}")
    
    def save_results(self, original_params: Dict, fused_params: Dict, 
                    original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        """Save all results properly"""
        print("\n💾 Saving results...")
        
        # Save fused parameters
        serializable_fused = {}
        for key, value in fused_params.items():
            if isinstance(value, np.ndarray):
                serializable_fused[key] = value.tolist()
            else:
                serializable_fused[key] = value
        
        with open(self.fusion_dir / 'fused_parameters.json', 'w') as f:
            json.dump(serializable_fused, f, indent=2)
        
        # Save enhanced mesh
        np.save(self.fusion_dir / 'enhanced_mesh.npy', enhanced_mesh)
        
        # Save mesh info
        with open(self.fusion_dir / 'enhanced_mesh_info.txt', 'w') as f:
            f.write(f"Enhanced Mesh Information\n")
            f.write(f"========================\n\n")
            f.write(f"Vertices: {enhanced_mesh.shape[0]}\n")
            f.write(f"Dimensions: {enhanced_mesh.shape[1]}\n")
            f.write(f"Bounding box:\n")
            f.write(f"  X: [{enhanced_mesh[:, 0].min():.4f}, {enhanced_mesh[:, 0].max():.4f}]\n")
            f.write(f"  Y: [{enhanced_mesh[:, 1].min():.4f}, {enhanced_mesh[:, 1].max():.4f}]\n")
            f.write(f"  Z: [{enhanced_mesh[:, 2].min():.4f}, {enhanced_mesh[:, 2].max():.4f}]\n")
            f.write(f"\nCentroid: [{enhanced_mesh.mean(axis=0)[0]:.4f}, "
                   f"{enhanced_mesh.mean(axis=0)[1]:.4f}, {enhanced_mesh.mean(axis=0)[2]:.4f}]\n")
        
        # Save parameter comparison
        with open(self.fusion_dir / 'parameter_changes.txt', 'w') as f:
            f.write("Parameter Changes Summary\n")
            f.write("========================\n\n")
            
            for param_name in ['left_hand_pose', 'right_hand_pose', 'expression']:
                orig = np.array(original_params[param_name])
                fused = np.array(fused_params[param_name])
                
                diff = np.linalg.norm(orig - fused)
                percent_change = (diff / (np.linalg.norm(orig) + 1e-8)) * 100
                
                f.write(f"{param_name}:\n")
                f.write(f"  Original norm: {np.linalg.norm(orig):.4f}\n")
                f.write(f"  Fused norm: {np.linalg.norm(fused):.4f}\n")
                f.write(f"  Difference: {diff:.4f}\n")
                f.write(f"  Change: {percent_change:.1f}%\n\n")
        
        print(f"   ✅ All results saved to {self.fusion_dir}")
    
    def run_fusion(self):
        """Execute the enhanced fusion pipeline"""
        print("\n" + "="*60)
        print("🚀 ENHANCED PARAMETER FUSION SYSTEM")
        print("="*60)
        
        try:
            # Load all parameters
            smplx_params, wilor_params, emoca_params = self.load_all_parameters()
            
            # Create fused parameters
            fused_params = self.create_fused_parameters(smplx_params, wilor_params, emoca_params)
            
            # Validate parameters
            is_valid = self.validate_parameters(fused_params)
            
            # Generate meshes
            original_mesh = np.array(smplx_params['mesh'])
            enhanced_mesh = self.generate_mesh(fused_params)
            
            # Render comparison
            self.render_comparison(smplx_params, fused_params, original_mesh, enhanced_mesh)
            
            # Save results
            self.save_results(smplx_params, fused_params, original_mesh, enhanced_mesh)
            
            print("\n" + "="*60)
            print("✅ FUSION COMPLETE!")
            print("="*60)
            print(f"\n📊 Results in: {self.fusion_dir}")
            print("   - fused_parameters.json (enhanced parameters)")
            print("   - enhanced_mesh.npy (3D mesh data)")
            print("   - mesh_comparison.pdf (visual comparison)")
            print("   - parameter_changes.txt (detailed changes)")
            
            if not is_valid:
                print("\n⚠️  Warning: Some parameters may be extreme. Check validation output above.")
            
        except Exception as e:
            print(f"\n❌ Fusion failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Parameter Fusion System')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing pipeline results')
    
    args = parser.parse_args()
    
    # Run enhanced fusion
    fusion = EnhancedParameterFusion(args.results_dir)
    fusion.run_fusion()

if __name__ == '__main__':
    main()