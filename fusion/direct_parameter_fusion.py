#!/usr/bin/env python3
"""
Enhanced Direct Parameter Fusion System
(CORRECTED: Fixed left hand coordinate system transformation)
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
import trimesh
import traceback

# Add paths for SMPL-X model access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
from human_models.human_models import SMPLX
from utils.visualization_utils import render_mesh
from main.config import Config

class EnhancedParameterFusion:
    """Enhanced fusion system with proper parameter handling and mesh generation"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.fusion_dir = self.results_dir / 'fusion_results'
        self.fusion_dir.mkdir(exist_ok=True)
        self.load_coordinate_analysis()
        self.setup_smplx_model()
        self.setup_rendering_config()
        self.hand_blend_weight = 0.8
        self.expression_scale = 0.3
        
        # Setup coordinate transformation matrices for left/right hands
        self.setup_coordinate_transforms()
        
    def setup_coordinate_transforms(self):
        """Setup coordinate system transformation matrices"""
        print("🔧 Setting up coordinate transformations...")
        
        # Right hand: mostly compatible, minor adjustments
        self.right_hand_transform = np.array([
            [1,  0,  0],   # X: same direction
            [0,  1,  0],   # Y: same direction  
            [0,  0,  1]    # Z: same direction
        ], dtype=np.float32)
        
        # Left hand: requires mirroring across YZ plane + axis corrections
        self.left_hand_transform = np.array([
            [-1,  0,  0],  # X: flip (mirror across YZ plane)
            [0,   1,  0],  # Y: same direction
            [0,   0, -1]   # Z: flip for proper orientation
        ], dtype=np.float32)
        
        # Joint-specific corrections for problematic joints
        self.joint_corrections = {
            'left': {
                0: np.array([0, np.pi, 0]),      # thumb_1: 180° rotation around Y
                1: np.array([0, 0, np.pi/2]),    # thumb_2: 90° rotation around Z
                3: np.array([0, 0, -np.pi/4]),   # index_1: -45° around Z
                12: np.array([0, 0, np.pi/6]),   # pinky_1: 30° around Z
            },
            'right': {
                0: np.array([0, 0, np.pi/12]),   # thumb_1: 15° fine-tuning
            }
        }
        print("   ✅ Coordinate transforms configured")
    
    def load_coordinate_analysis(self):
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
        print("🤖 Setting up SMPL-X model...")
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
        # Default config
        self.config = type('Config', (), {
            'model': type('Model', (), {
                'focal': [5000.0, 5000.0], 
                'princpt': [128.0, 128.0], 
                'input_body_shape': [256, 256]
            })()
        })()
        print("   ⚠️  Using default config")
    
    def load_all_parameters(self) -> Tuple[Dict, Dict, Dict]:
        print("\n📥 Loading all model parameters...")
        smplx_params = self._load_smplestx_params()
        wilor_params = self._load_wilor_params()
        emoca_params = self._load_emoca_params()
        return smplx_params, wilor_params, emoca_params

    def _load_smplestx_params(self) -> Dict:
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"   ✅ SMPLest-X: Loaded from {param_file.relative_to(self.results_dir)}")
            
            meta_file = param_file.parent / 'camera_metadata.json'
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    self.camera_metadata = json.load(f)
                print(f"   ✅ Camera metadata loaded from {meta_file.name}")
            else:
                raise FileNotFoundError(f"CRITICAL: SMPLest-X camera_metadata.json not found!")
            return params
        raise FileNotFoundError("SMPLest-X parameters not found")
    
    def _load_wilor_params(self) -> Dict:
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f:
                params = json.load(f)
            print(f"   ✅ WiLoR: Loaded from {param_file.relative_to(self.results_dir)}")
            return params
        raise FileNotFoundError("WiLoR parameters not found")
    
    def _load_emoca_params(self) -> Dict:
        search_patterns = [
            'emoca_results/EMOCA*/test*/codes.json',
            'emoca_results/*/codes.json',
            'emoca_results/*/*/codes.json'
        ]
        for pattern in search_patterns:
            for codes_file in self.results_dir.glob(pattern):
                with open(codes_file, 'r') as f:
                    params = json.load(f)
                print(f"   ✅ EMOCA: Loaded from {codes_file.relative_to(self.results_dir)}")
                return params
        print("   ⚠️  EMOCA parameters not found")
        return {}
    
    def axis_angle_to_matrix(self, axis_angle: np.ndarray) -> np.ndarray:
        """Convert axis-angle to rotation matrix"""
        if axis_angle.shape == (3,):
            # Single axis-angle
            rotation_matrix, _ = cv2.Rodrigues(axis_angle.astype(np.float64))
            return rotation_matrix.astype(np.float32)
        else:
            # Batch of axis-angles
            matrices = []
            for i in range(axis_angle.shape[0]):
                rot_mat, _ = cv2.Rodrigues(axis_angle[i].astype(np.float64))
                matrices.append(rot_mat)
            return np.array(matrices, dtype=np.float32)

    def matrix_to_axis_angle_robust(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Robust conversion from rotation matrix to axis-angle
        Handles both single matrices and batches
        """
        if rotation_matrix.ndim == 2:
            # Single matrix (3, 3)
            try:
                axis_angle, _ = cv2.Rodrigues(rotation_matrix.astype(np.float64))
                return axis_angle.flatten().astype(np.float32)
            except:
                # Fallback for problematic matrices
                r = R.from_matrix(rotation_matrix)
                return r.as_rotvec().astype(np.float32)
        else:
            # Batch of matrices (N, 3, 3)
            axis_angles = []
            for i in range(rotation_matrix.shape[0]):
                try:
                    aa, _ = cv2.Rodrigues(rotation_matrix[i].astype(np.float64))
                    axis_angles.append(aa.flatten())
                except:
                    r = R.from_matrix(rotation_matrix[i])
                    axis_angles.append(r.as_rotvec())
            return np.array(axis_angles, dtype=np.float32)

    def transform_hand_coordinate_system(self, hand_pose_matrices: np.ndarray, 
                                       hand_type: str) -> np.ndarray:
        """
        Transform hand pose from WiLoR coordinate system to SMPL-X coordinate system
        
        Args:
            hand_pose_matrices: (15, 3, 3) rotation matrices for 15 hand joints
            hand_type: 'left' or 'right'
            
        Returns:
            transformed_matrices: (15, 3, 3) transformed rotation matrices
        """
        transformed_matrices = np.zeros_like(hand_pose_matrices)
        
        # Select appropriate transformation matrix
        if hand_type == 'left':
            coord_transform = self.left_hand_transform
        else:
            coord_transform = self.right_hand_transform
            
        # Transform each joint
        for joint_idx in range(15):
            # Apply coordinate system transformation
            # R_new = T * R_old * T^(-1)
            original_matrix = hand_pose_matrices[joint_idx]
            transformed_matrix = coord_transform @ original_matrix @ coord_transform.T
            
            # Apply joint-specific corrections if needed
            if joint_idx in self.joint_corrections[hand_type]:
                correction_aa = self.joint_corrections[hand_type][joint_idx]
                correction_matrix = self.axis_angle_to_matrix(correction_aa)
                transformed_matrix = correction_matrix @ transformed_matrix
                
            transformed_matrices[joint_idx] = transformed_matrix
            
        return transformed_matrices

    def convert_rotation_matrix_to_axis_angle(self, rot_matrices: np.ndarray, 
                                            hand_type: str) -> np.ndarray:
        """
        CORRECTED: Convert rotation matrices to axis-angle with proper coordinate transformation
        """
        # Reshape to (N, 3, 3)
        n_joints = rot_matrices.shape[0] // 9
        rot_mats = rot_matrices.reshape(n_joints, 3, 3)
        
        # CRITICAL FIX: Apply coordinate system transformation
        if hand_type == 'left':
            # Transform left hand coordinate system
            transformed_matrices = self.transform_hand_coordinate_system(rot_mats, 'left')
            print(f"   🔄 Applied left hand coordinate transformation")
        else:
            # Apply minimal transformation for right hand
            transformed_matrices = self.transform_hand_coordinate_system(rot_mats, 'right')
            print(f"   🔄 Applied right hand coordinate transformation")
        
        # Convert to axis-angle using robust method
        axis_angles = []
        for i in range(n_joints):
            aa = self.matrix_to_axis_angle_robust(transformed_matrices[i])
            axis_angles.append(aa)
        
        # Concatenate into single flat array
        result = np.concatenate(axis_angles)
        
        # Debug output
        print(f"   📊 {hand_type.title()} hand axis-angle range: [{result.min():.3f}, {result.max():.3f}]")
        
        return result

    def extract_wilor_hand_poses(self, wilor_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract and convert WiLoR hand poses to SMPL-X format with proper coordinate transformation"""
        print("\n🖐️  Extracting WiLoR hand poses with coordinate system corrections...")
        
        left_hand_pose = np.zeros(45)
        right_hand_pose = np.zeros(45)
        
        for hand in wilor_params.get('hands', []):
            if 'mano_parameters' in hand and 'parameters' in hand['mano_parameters']:
                mano_params = hand['mano_parameters']['parameters']
                if 'hand_pose' in mano_params:
                    hand_type = hand.get('hand_type', 'unknown')
                    
                    # Extract rotation matrices
                    rot_matrices = np.array(mano_params['hand_pose']['values']).flatten()
                    
                    # CRITICAL FIX: Apply coordinate transformation based on hand type
                    axis_angles = self.convert_rotation_matrix_to_axis_angle(
                        rot_matrices, hand_type
                    )
                    
                    if len(axis_angles) >= 45:
                        if hand_type == 'left':
                            left_hand_pose = axis_angles[:45]
                            print(f"   ✅ Left hand: converted with coordinate transformation")
                        elif hand_type == 'right':
                            right_hand_pose = axis_angles[:45]
                            print(f"   ✅ Right hand: converted with coordinate transformation")
        
        # Final validation
        print(f"\n   📊 Final pose statistics:")
        print(f"   Left hand - range: [{left_hand_pose.min():.3f}, {left_hand_pose.max():.3f}]")
        print(f"   Right hand - range: [{right_hand_pose.min():.3f}, {right_hand_pose.max():.3f}]")
        
        return left_hand_pose, right_hand_pose

    def map_emoca_expression(self, emoca_params: Dict) -> np.ndarray:
        if not emoca_params or 'expcode' not in emoca_params:
            print("   ⚠️  No EMOCA expression, using neutral")
            return np.zeros(10)
        
        print("\n🎭 Mapping EMOCA expression...")
        emoca_exp = np.array(emoca_params['expcode'])
        
        if len(emoca_exp) >= 10:
            mapped_exp = emoca_exp[:10] * self.expression_scale
        else:
            mapped_exp = np.zeros(10)
            mapped_exp[:len(emoca_exp)] = emoca_exp * self.expression_scale
        
        # Clip to reasonable range
        mapped_exp = np.clip(mapped_exp, -2.0, 2.0)
        return mapped_exp
    
    def create_fused_parameters(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict) -> Dict:
        """Create properly fused parameters using hierarchical composition."""
        print("\n🔧 Creating fused parameters using Hierarchical Composition...")
        
        # Start with SMPLest-X as foundation
        fused = {
            'betas': np.array(smplx_params['betas']),
            'body_pose': np.array(smplx_params['body_pose']),
            'root_pose': np.array(smplx_params['root_pose']),
            'translation': np.array(smplx_params['translation']),
            'jaw_pose': np.array(smplx_params['jaw_pose'])
        }
        
        # Extract WiLoR hand poses with proper coordinate transformation
        left_hand_wilor, right_hand_wilor = self.extract_wilor_hand_poses(wilor_params)
        
        # OPTION 1: Full replacement with WiLoR hands
        fused['left_hand_pose'] = left_hand_wilor
        fused['right_hand_pose'] = right_hand_wilor
        print("   ✅ Complete WiLoR hand replacement applied")
        
        # OPTION 2 (Alternative): Hierarchical composition
        # Uncomment these lines if you prefer wrist from SMPLest-X + fingers from WiLoR
        # smplx_left_wrist_pose = np.array(smplx_params['left_hand_pose'])[:3]
        # smplx_right_wrist_pose = np.array(smplx_params['right_hand_pose'])[:3]
        # wilor_left_finger_pose = left_hand_wilor[3:]
        # wilor_right_finger_pose = right_hand_wilor[3:]
        # final_left_hand = np.concatenate([smplx_left_wrist_pose, wilor_left_finger_pose])
        # final_right_hand = np.concatenate([smplx_right_wrist_pose, wilor_right_finger_pose])
        # fused['left_hand_pose'] = final_left_hand
        # fused['right_hand_pose'] = final_right_hand
        # print("   ✅ Hands composed: SMPLest-X wrist + WiLoR fingers.")
        
        # Add EMOCA expression
        fused['expression'] = self.map_emoca_expression(emoca_params)
        
        # Add metadata
        fused['fusion_metadata'] = {
            'body_source': 'SMPLest-X',
            'hand_source': 'WiLoR_with_coordinate_transformation',
            'expression_source': 'EMOCA',
            'coordinate_fixes_applied': True,
            'left_hand_transform': 'coordinate_system_correction',
            'right_hand_transform': 'minimal_adjustment'
        }
        
        return fused
    
    def generate_mesh(self, params: Dict, use_translation: bool = True) -> np.ndarray:
        print("\n🎯 Generating mesh...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smplx_layer = self.smplx_model.layer['neutral'].to(device)
        
        translation = params['translation'] if use_translation else np.zeros(3)
        
        torch_params = {
            'betas': torch.tensor(params['betas']).float().unsqueeze(0).to(device),
            'body_pose': torch.tensor(params['body_pose']).float().unsqueeze(0).to(device),
            'global_orient': torch.tensor(params['root_pose']).float().unsqueeze(0).to(device),
            'left_hand_pose': torch.tensor(params['left_hand_pose']).float().unsqueeze(0).to(device),
            'right_hand_pose': torch.tensor(params['right_hand_pose']).float().unsqueeze(0).to(device),
            'jaw_pose': torch.tensor(params['jaw_pose']).float().unsqueeze(0).to(device),
            'expression': torch.tensor(params['expression']).float().unsqueeze(0).to(device),
            'transl': torch.tensor(translation).float().unsqueeze(0).to(device),
            'leye_pose': torch.zeros(1, 3).float().to(device),
            'reye_pose': torch.zeros(1, 3).float().to(device)
        }
        
        with torch.no_grad():
            output = smplx_layer(**torch_params)
        
        mesh = output.vertices[0].detach().cpu().numpy()
        print(f"   ✅ Mesh generated.")
        return mesh
    
    def validate_parameters(self, params: Dict) -> bool:
        print("\n✅ Validating parameters...")
        issues = []
        
        for pose_name in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
            pose = params[pose_name]
            if np.abs(pose).max() > np.pi * 1.5:
                issues.append(f"{pose_name} has extreme values.")
        
        if np.abs(params['betas']).max() > 5:
            issues.append("Shape parameters extreme.")
        if np.abs(params['expression']).max() > 3:
            issues.append("Expression extreme.")
        
        if issues:
            print("   ⚠️  Validation issues:", *issues)
            return False
        else:
            print("   ✅ All parameters within reasonable ranges")
            return True
        
    def save_results(self, original_params: Dict, fused_params: Dict, original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        print("\n💾 Saving results...")
        
        # Save parameters
        serializable_fused = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fused_params.items()}
        with open(self.fusion_dir / 'fused_parameters.json', 'w') as f:
            json.dump(serializable_fused, f, indent=2)
        
        # Save meshes
        np.save(self.fusion_dir / 'enhanced_mesh.npy', enhanced_mesh)
        
        print(f"   ✅ All results saved to {self.fusion_dir}")

    def render_comparison(self, original_params: Dict, fused_params: Dict, original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        print("\n🎨 Rendering comparison...")
        
        # Find input image
        img_path = None
        for temp_dir in [self.results_dir / 'wilor_results' / 'temp_input', self.results_dir / 'emoca_results' / 'temp_input']:
            if temp_dir.exists():
                for img_file in temp_dir.glob('*'):
                    img_path = img_file
                    break
                if img_path:
                    break
        
        if not img_path:
            print("   ⚠️  No input image found for rendering")
            return
        
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        if not hasattr(self, 'camera_metadata'):
            print("   ⚠️  Camera metadata not loaded.")
            return
        
        cam_params = {
            'focal': np.array(self.camera_metadata['focal_length']),
            'princpt': np.array(self.camera_metadata['principal_point'])
        }
        
        # Render both meshes
        img_orig = render_mesh(img, original_mesh, self.smplx_model.face, cam_params)
        img_enhanced = render_mesh(img.copy(), enhanced_mesh, self.smplx_model.face, cam_params)
        
        # Create comparison
        comparison = np.hstack([img_orig, img_enhanced])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original (SMPLest-X)', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Enhanced Fusion', (w + 10, 30), font, 1, (0, 255, 0), 2)
        
        output_path = str(self.fusion_dir / 'mesh_comparison.png')
        cv2.imwrite(output_path, comparison)
        print(f"   ✅ Comparison saved to {output_path}")
    
    def run_fusion(self):
        print("\n" + "="*60 + "\n🚀 ENHANCED PARAMETER FUSION SYSTEM (CORRECTED)\n" + "="*60)
        
        try:
            # Load parameters
            smplx_params, wilor_params, emoca_params = self.load_all_parameters()
            
            # Create fused parameters with coordinate system corrections
            fused_params = self.create_fused_parameters(smplx_params, wilor_params, emoca_params)
            
            # Validate parameters
            is_valid = self.validate_parameters(fused_params)
            
            # Generate meshes
            original_mesh_for_render = self.generate_mesh(smplx_params)
            enhanced_mesh_for_render = self.generate_mesh(fused_params)
            
            # Render comparison
            self.render_comparison(smplx_params, fused_params, original_mesh_for_render, enhanced_mesh_for_render)
            
            # Save results
            self.save_results(smplx_params, fused_params, original_mesh_for_render, enhanced_mesh_for_render)
            
            # Save OBJ files for inspection
            print("\n💾 Saving OBJ files for Blender inspection...")
            original_centered_mesh = self.generate_mesh(smplx_params, use_translation=False)
            fused_centered_mesh = self.generate_mesh(fused_params, use_translation=False)
            
            original_obj_path = self.fusion_dir / 'original_final.obj'
            trimesh.Trimesh(vertices=original_centered_mesh, faces=self.smplx_model.face).export(str(original_obj_path))
            print(f"   - Saved original mesh to {original_obj_path}")
            
            fused_obj_path = self.fusion_dir / 'fused_final.obj'
            trimesh.Trimesh(vertices=fused_centered_mesh, faces=self.smplx_model.face).export(str(fused_obj_path))
            print(f"   - Saved CORRECTED fused mesh to {fused_obj_path}")
            
            print("\n" + "="*60)
            print("✅ FUSION COMPLETE WITH COORDINATE SYSTEM FIXES!")
            print("🔧 Applied proper left hand coordinate transformation")
            print("🔧 Used robust matrix-to-axis-angle conversion")
            print("🔧 Added joint-specific corrections")
            print("="*60)
            
            if not is_valid:
                print("\n⚠️  Warning: Some parameters may be extreme.")
                
        except Exception as e:
            print(f"\n❌ Fusion failed: {e}")
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Parameter Fusion System (CORRECTED)')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing pipeline results')
    args = parser.parse_args()
    
    fusion = EnhancedParameterFusion(args.results_dir)
    fusion.run_fusion()

if __name__ == '__main__':
    main()