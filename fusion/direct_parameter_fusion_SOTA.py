#!/usr/bin/env python3
"""
Enhanced Direct Parameter Fusion System
(Using your proposed list comprehension for conversion)
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
import pickle

# Add paths for SMPL-X model access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
from human_models.human_models import SMPLX
from utils.visualization_utils import render_mesh
from main.config import Config

class EnhancedParameterFusion:
    """Enhanced fusion system with proper parameter handling and mesh generation"""
    
    # --- NO CHANGES TO ANY OF THESE METHODS ---
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir); self.fusion_dir = self.results_dir / 'fusion_results'; self.fusion_dir.mkdir(exist_ok=True)
        self.load_coordinate_analysis(); self.setup_smplx_model(); self.setup_rendering_config()
        self.hand_blend_weight = 0.8; self.expression_scale = 0.3
        
    def load_coordinate_analysis(self):
        print("📥 Loading coordinate analysis..."); coord_file = self.results_dir / 'coordinate_analysis_summary.json'
        if not coord_file.exists(): raise FileNotFoundError(f"Coordinate analysis not found: {coord_file}")
        with open(coord_file, 'r') as f: self.coord_analysis = json.load(f)
        self.scale_factor = self.coord_analysis['transformation_parameters']['scale_factor']
        self.translation = np.array(self.coord_analysis['transformation_parameters']['translation_vector'])
        print(f"   ✅ Scale factor: {self.scale_factor:.4f}")
        print(f"   ✅ Translation: {self.translation}")
    
    def setup_smplx_model(self):
        print("🤖 Setting up SMPL-X model..."); model_paths = ['human_models/human_model_files/', '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/pretrained_models/smplx', './pretrained_models/smplx', '../pretrained_models/smplx']
        for path in model_paths:
            if os.path.exists(path):
                try: self.smplx_model = SMPLX(path); print(f"   ✅ SMPL-X model loaded from: {path}"); return
                except Exception as e: print(f"   ⚠️  Failed to load from {path}: {e}")
        raise RuntimeError("Could not load SMPL-X model from any path")
    
    def setup_rendering_config(self):
        print("🎨 Setting up rendering configuration..."); config_paths = ['pretrained_models/smplest_x/config_base.py', '../pretrained_models/smplest_x/config_base.py', 'external/SMPLest-X/configs/config_smplest_x_h.py']
        for path in config_paths:
            if os.path.exists(path): self.config = Config.load_config(path); print(f"   ✅ Config loaded from: {path}"); return
        self.config = type('Config', (), {'model': type('Model', (), {'focal': [5000.0, 5000.0], 'princpt': [128.0, 128.0], 'input_body_shape': [256, 256]})()})(); print("   ⚠️  Using default config")
    
    def load_all_parameters(self) -> Tuple[Dict, Dict, Dict]:
        print("\n📥 Loading all model parameters..."); smplx_params = self._load_smplestx_params(); wilor_params = self._load_wilor_params(); emoca_params = self._load_emoca_params(); return smplx_params, wilor_params, emoca_params

    def _load_smplestx_params(self) -> Dict:
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f: params = json.load(f)
            print(f"   ✅ SMPLest-X: Loaded from {param_file.relative_to(self.results_dir)}")
            meta_file = param_file.parent / 'camera_metadata.json'
            if meta_file.exists():
                with open(meta_file, 'r') as f: self.camera_metadata = json.load(f)
                print(f"   ✅ Camera metadata loaded from {meta_file.name}")
            else: raise FileNotFoundError(f"CRITICAL: SMPLest-X camera_metadata.json not found!")
            return params
        raise FileNotFoundError("SMPLest-X parameters not found")
    
    def _load_wilor_params(self) -> Dict:
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f: params = json.load(f)
            print(f"   ✅ WiLoR: Loaded from {param_file.relative_to(self.results_dir)}"); return params
        raise FileNotFoundError("WiLoR parameters not found")
    
    def _load_emoca_params(self) -> Dict:
        search_patterns = ['emoca_results/EMOCA*/test*/codes.json', 'emoca_results/*/codes.json', 'emoca_results/*/*/codes.json']
        for pattern in search_patterns:
            for codes_file in self.results_dir.glob(pattern):
                with open(codes_file, 'r') as f: params = json.load(f)
                print(f"   ✅ EMOCA: Loaded from {codes_file.relative_to(self.results_dir)}"); return params
        print("   ⚠️  EMOCA parameters not found"); return {}
    
    # --- THIS IS YOUR PROPOSED FUNCTION ---
    def convert_rotation_matrix_to_axis_angle(self, rot_matrices: np.ndarray) -> np.ndarray:
        """Convert rotation matrices to axis-angle representation using a list comprehension."""
        # Reshape to (N, 3, 3)
        n_joints = rot_matrices.shape[0] // 9
        rot_mats = rot_matrices.reshape(n_joints, 3, 3)
        
        # Use a list comprehension for conversion
        axis_angles = [R.from_matrix(rot_mats[i]).as_rotvec() for i in range(n_joints)]
        
        # Concatenate the list of arrays into a single flat array
        return np.concatenate(axis_angles)

# ==============================================================================
#  Replacing  old load_mano_mean_poses with this one
# ==============================================================================

    def load_mano_mean_poses(self):
        """
        CORRECTED: Loads MANO mean poses directly from the official MANO .pkl files
        as specified by the WiloR author. This ensures we are using the exact
        mean pose values that the fusion logic expects.
        """
        print("loading mano mean poses")
        # Define common directories where MANO models might be stored
        mano_model_dirs = [
            '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/external/WiLoR/mano_data',  # Common in some projects
            'pretrained_models/mano',
            'mano/models'
        ]
        
        for directory in mano_model_dirs:
            left_path = Path(directory) / 'MANO_LEFT.pkl'
            right_path = Path(directory) / 'MANO_RIGHT.pkl'
            
            if left_path.exists() and right_path.exists():
                print(f"   ✅ Found MANO models in: {directory}")
                try:
                    with open(left_path, 'rb') as f:
                        mano_left_data = pickle.load(f, encoding='latin1')
                    with open(right_path, 'rb') as f:
                        mano_right_data = pickle.load(f, encoding='latin1')
                    
                    # The key is 'hands_mean' in BOTH files
                    mean_left = mano_left_data['hands_mean']
                    mean_right = mano_right_data['hands_mean']
                    
                    print("   ✅ Successfully loaded 'hands_mean' from both MANO_LEFT and MANO_RIGHT pkl files.")
                    return mean_left, mean_right
                except Exception as e:
                    print(f"   ⚠️  Found files, but failed to load from {directory}: {e}")
                    
        # If the loop completes without finding the files
        raise FileNotFoundError(
            "Could not find MANO_LEFT.pkl and MANO_RIGHT.pkl. "
            "Please download the MANO models from https://mano.is.tue.mpg.de/ "
            "and place them in one of the searched directories (e.g., 'pretrained_models/mano/')."
        )

    # ==============================================================================
    #  The "Supervisor-Approved" Final Function
    # ==============================================================================

    def extract_wilor_hand_poses(self, wilor_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts WiLoR hand poses using the exact transformation logic provided
        by the author of WiloR for left-hand compatibility with SMPL-X/MANO.
        """
        print("\n🖐️  Extracting WiLoR hand poses (using SUPERVISOR'S a.k.a WiloR author's logic)...")
        
        # IMPORTANT: Ensure this function loads the mean poses from the .pkl files
        # as specified by your supervisor, not a different .npz file.
        hands_mean_left, hands_mean_right = self.load_mano_mean_poses()
        
        left_hand_pose_param = np.zeros(45)
        right_hand_pose_param = np.zeros(45)
        
        for hand in wilor_params.get('hands', []):
            if 'mano_parameters' not in hand: continue
                
            mano_params = hand['mano_parameters']['parameters']
            hand_type = hand.get('hand_type', '')
            
            if 'hand_pose' in mano_params:
                # 1. Convert WiloR's rotation matrices to a flat (45,) axis-angle vector
                rot_matrices = np.array(mano_params['hand_pose']['values']).flatten()
                pose_parameter = self.convert_rotation_matrix_to_axis_angle(rot_matrices)

                if hand_type == 'left':
                    print("   Applying supervisor's special transform for LEFT hand...")
                    final_param = pose_parameter.copy()
                    
                    # 2. Apply the two-step transformation for the left hand
                    # Step A: Invert the entire rotation vector
                    final_param *= -1.0
                    # Step B: Flip the X-component of each joint back
                    final_param[::3] *= -1.0
                    
                    # 3. Subtract the mean pose at the end
                    left_hand_pose_param = final_param - hands_mean_left
                    print(f"   ✅ Processed LEFT hand with supervisor's logic.")

                elif hand_type == 'right':
                    # The right hand needs no special transformation, just mean subtraction.
                    right_hand_pose_param = pose_parameter - hands_mean_right
                    print(f"   ✅ Processed RIGHT hand.")

        return left_hand_pose_param, right_hand_pose_param


    def create_fused_parameters(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict) -> Dict:
        """
        CORRECTED: Use full replacement, not hierarchical composition
        """
        print("\n🔧 Creating fused parameters (CORRECTED)...")
        
        # Start with SMPLest-X base
        fused = {
            'betas': np.array(smplx_params['betas']),
            'body_pose': np.array(smplx_params['body_pose']),
            'root_pose': np.array(smplx_params['root_pose']),
            'translation': np.array(smplx_params['translation']),
            'jaw_pose': np.array(smplx_params['jaw_pose'])
        }
        
        # FULL REPLACEMENT - Don't mix wrist/fingers!
        left_hand, right_hand = self.extract_wilor_hand_poses(wilor_params)
        fused['left_hand_pose'] = left_hand
        fused['right_hand_pose'] = right_hand
        
        print("\n   ✅ Hands: FULL WiLoR replacement (no mixing)")
        
        # Expression
        fused['expression'] = self.map_emoca_expression(emoca_params)
        
        # Check differences
        left_diff = np.linalg.norm(left_hand - np.array(smplx_params['left_hand_pose']))
        right_diff = np.linalg.norm(right_hand - np.array(smplx_params['right_hand_pose']))
        print(f"\n   📊 Changes from original:")
        print(f"      Left hand: {left_diff:.3f}")
        print(f"      Right hand: {right_diff:.3f}")
        
        return fused

    def map_emoca_expression(self, emoca_params: Dict) -> np.ndarray:
        if not emoca_params or 'expcode' not in emoca_params: print("   ⚠️  No EMOCA expression, using neutral"); return np.zeros(10)
        print("\n🎭 Mapping EMOCA expression..."); emoca_exp = np.array(emoca_params['expcode'])
        if len(emoca_exp) >= 10: mapped_exp = emoca_exp[:10] * self.expression_scale
        else: mapped_exp = np.zeros(10); mapped_exp[:len(emoca_exp)] = emoca_exp * self.expression_scale
        mapped_exp = np.clip(mapped_exp, -2.0, 2.0); return mapped_exp
   
    def generate_mesh(self, params: Dict, use_translation: bool = True) -> np.ndarray:
        print("\n🎯 Generating mesh..."); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        smplx_layer = self.smplx_model.layer['neutral'].to(device); translation = params['translation'] if use_translation else np.zeros(3)
        torch_params = {
            'betas': torch.tensor(params['betas']).float().unsqueeze(0).to(device), 'body_pose': torch.tensor(params['body_pose']).float().unsqueeze(0).to(device),
            'global_orient': torch.tensor(params['root_pose']).float().unsqueeze(0).to(device), 'left_hand_pose': torch.tensor(params['left_hand_pose']).float().unsqueeze(0).to(device),
            'right_hand_pose': torch.tensor(params['right_hand_pose']).float().unsqueeze(0).to(device), 'jaw_pose': torch.tensor(params['jaw_pose']).float().unsqueeze(0).to(device),
            'expression': torch.tensor(params['expression']).float().unsqueeze(0).to(device), 'transl': torch.tensor(translation).float().unsqueeze(0).to(device),
            'leye_pose': torch.zeros(1, 3).float().to(device), 'reye_pose': torch.zeros(1, 3).float().to(device)
        }
        with torch.no_grad(): output = smplx_layer(**torch_params)
        mesh = output.vertices[0].detach().cpu().numpy(); print(f"   ✅ Mesh generated."); return mesh
    
    # --- The rest of your code is FAITHFULLY restored ---
    def validate_parameters(self, params: Dict) -> bool:
        print("\n✅ Validating parameters..."); issues = [];
        for pose_name in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
            pose = params[pose_name]
            if np.abs(pose).max() > np.pi * 1.5: issues.append(f"{pose_name} has extreme values.")
        if np.abs(params['betas']).max() > 5: issues.append("Shape parameters extreme.")
        if np.abs(params['expression']).max() > 3: issues.append("Expression extreme.")
        if issues: print("   ⚠️  Validation issues:", *issues); return False
        else: print("   ✅ All parameters within reasonable ranges"); return True
        
    def save_results(self, original_params: Dict, fused_params: Dict, original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        print("\n💾 Saving results..."); serializable_fused = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fused_params.items()}
        with open(self.fusion_dir / 'fused_parameters.json', 'w') as f: json.dump(serializable_fused, f, indent=2)
        np.save(self.fusion_dir / 'enhanced_mesh.npy', enhanced_mesh); print(f"   ✅ All results saved to {self.fusion_dir}")

    def render_comparison(self, original_params: Dict, fused_params: Dict, original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        print("\n🎨 Rendering comparison..."); img_path = None
        for temp_dir in [self.results_dir / 'wilor_results' / 'temp_input', self.results_dir / 'emoca_results' / 'temp_input']:
            if temp_dir.exists():
                for img_file in temp_dir.glob('*'): img_path = img_file; break
                if img_path: break
        if not img_path: print("   ⚠️  No input image found for rendering"); return
        img = cv2.imread(str(img_path)); h, w = img.shape[:2]
        if not hasattr(self, 'camera_metadata'): print("   ⚠️  Camera metadata not loaded."); return
        cam_params = {'focal': np.array(self.camera_metadata['focal_length']), 'princpt': np.array(self.camera_metadata['principal_point'])}
        img_orig = render_mesh(img, original_mesh, self.smplx_model.face, cam_params)
        img_enhanced = render_mesh(img.copy(), enhanced_mesh, self.smplx_model.face, cam_params)
        comparison = np.hstack([img_orig, img_enhanced]); font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, 'Original (SMPLest-X)', (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Enhanced Fusion', (w + 10, 30), font, 1, (0, 255, 0), 2)
        output_path = str(self.fusion_dir / 'mesh_comparison.png')
        cv2.imwrite(output_path, comparison); print(f"   ✅ Comparison saved to {output_path}")
    
    def run_fusion(self):
        print("\n" + "="*60 + "\n🚀 ENHANCED PARAMETER FUSION SYSTEM\n" + "="*60)
        try:
            smplx_params, wilor_params, emoca_params = self.load_all_parameters()
            fused_params = self.create_fused_parameters(smplx_params, wilor_params, emoca_params)
            is_valid = self.validate_parameters(fused_params)
            original_mesh_for_render = self.generate_mesh(smplx_params)
            enhanced_mesh_for_render = self.generate_mesh(fused_params)
            self.render_comparison(smplx_params, fused_params, original_mesh_for_render, enhanced_mesh_for_render)
            self.save_results(smplx_params, fused_params, original_mesh_for_render, enhanced_mesh_for_render)
            
            print("\n💾 Saving OBJ files for Blender inspection...")
            original_centered_mesh = self.generate_mesh(smplx_params, use_translation=False)
            fused_centered_mesh = self.generate_mesh(fused_params, use_translation=False)
            original_obj_path = self.fusion_dir / 'original_final.obj'
            trimesh.Trimesh(vertices=original_centered_mesh, faces=self.smplx_model.face).export(str(original_obj_path))
            print(f"   - Saved original mesh to {original_obj_path}")
            fused_obj_path = self.fusion_dir / 'fused_final.obj'
            trimesh.Trimesh(vertices=fused_centered_mesh, faces=self.smplx_model.face).export(str(fused_obj_path))
            print(f"   - Saved final corrected fused mesh to {fused_obj_path}")
            
            print("\n" + "="*60 + "\n✅ FUSION COMPLETE!\n" + "="*60)
            if not is_valid: print("\n⚠️  Warning: Some parameters may be extreme.")
        except Exception as e:
            print(f"\n❌ Fusion failed: {e}"); traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Parameter Fusion System')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing pipeline results')
    args = parser.parse_args()
    fusion = EnhancedParameterFusion(args.results_dir)
    fusion.run_fusion()

if __name__ == '__main__':
    main()