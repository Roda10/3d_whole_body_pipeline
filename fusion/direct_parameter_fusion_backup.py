#!/usr/bin/env python3
"""
Enhanced Direct Parameter Fusion System with Central Gallery Support
(Fully corrected for WiloR hand and EMOCA face fusion)
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
    """Enhanced fusion system with central gallery support and corrected face fusion"""

    def __init__(self, results_dir: str, gallery_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.fusion_dir = self.results_dir / 'fusion_results'
        self.fusion_dir.mkdir(exist_ok=True)
        
        if gallery_dir:
            self.gallery_dir = Path(gallery_dir)
            self.gallery_dir.mkdir(exist_ok=True)
            print(f"📁 Using external gallery: {self.gallery_dir}")
        else:
            self.gallery_dir = self.fusion_dir / 'render_gallery'
            self.gallery_dir.mkdir(exist_ok=True)
            print(f"📁 Using local gallery: {self.gallery_dir}")

        self.load_coordinate_analysis()
        self.setup_smplx_model()
        self.setup_rendering_config()

        self.expression_scale = 0.4 # Slightly increased scale might be good for full expression vector

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
        # Multiple patterns to handle different persistence output structures
        patterns = [
            "smplestx_results/inference_output_*/person_*/smplx_params_*.json",  # Persistence structure
            "smplestx_results/*/person_*/smplx_params_*.json",                  # Original structure  
            "smplestx_results/person_*/smplx_params_*.json"                     # Fallback
        ]
        
        for pattern in patterns:
            param_files = list(self.results_dir.glob(pattern))
            if param_files:
                param_file = param_files[0]  # Use first found
                print(f"   ✅ SMPLest-X: Found with pattern {pattern}")
                with open(param_file, 'r') as f:
                    params = json.load(f)
                print(f"   ✅ SMPLest-X: Loaded from {param_file.relative_to(self.results_dir)}")
                
                meta_file = param_file.parent / 'camera_metadata.json'
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        self.camera_metadata = json.load(f)
                    print(f"   ✅ Camera metadata loaded from {meta_file.name}")
                else:
                    raise FileNotFoundError("CRITICAL: SMPLest-X camera_metadata.json not found!")
                return params
        
        raise FileNotFoundError("SMPLest-X parameters not found")

    def _load_wilor_params(self) -> Dict:
        # Multiple patterns for WiLoR parameter files
        patterns = [
            'wilor_results/*_parameters.json',  # Standard naming: test2_parameters.json
            'wilor_results/parameters.json',    # Alternative naming
            'wilor_results/*.json'              # Fallback for any JSON in wilor_results
        ]
        
        for pattern in patterns:
            param_files = list(self.results_dir.glob(pattern))
            # Filter out non-parameter files if using wildcard
            if pattern == 'wilor_results/*.json':
                param_files = [f for f in param_files if 'param' in f.name.lower() or f.name.endswith('_parameters.json')]
            
            if param_files:
                param_file = param_files[0]  # Use first found
                print(f"   ✅ WiLoR: Found with pattern {pattern}")
                with open(param_file, 'r') as f:
                    params = json.load(f)
                print(f"   ✅ WiLoR: Loaded from {param_file.relative_to(self.results_dir)}")
                return params
        
        raise FileNotFoundError("WiLoR parameters not found")

    def _load_emoca_params(self) -> Dict:
        # Multiple search patterns for EMOCA codes.json - covers persistence and original structures
        search_patterns = [
            'emoca_results/test*/codes.json',        # Persistence: test200/codes.json
            'emoca_results/*/codes.json',            # Direct: any_dir/codes.json  
            'emoca_results/EMOCA*/test*/codes.json', # Original nested: EMOCA_v2_lr_mse_20/test200/codes.json
            'emoca_results/*/*/codes.json'          # Deep nested fallback
        ]
        
        for pattern in search_patterns:
            codes_files = list(self.results_dir.glob(pattern))
            if codes_files:
                codes_file = codes_files[0]  # Use first found
                print(f"   ✅ EMOCA: Found with pattern {pattern}")
                with open(codes_file, 'r') as f:
                    params = json.load(f)
                print(f"   ✅ EMOCA: Loaded from {codes_file.relative_to(self.results_dir)}")
                
                # EMOCA sometimes saves single-element lists, so we extract the dict inside
                if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
                    return params[0]
                return params
        
        print("   ⚠️  EMOCA parameters not found")
        return {}

    def convert_rotation_matrix_to_axis_angle(self, rot_matrices: np.ndarray) -> np.ndarray:
        n_joints = rot_matrices.shape[0] // 9
        rot_mats = rot_matrices.reshape(n_joints, 3, 3)
        axis_angles = [R.from_matrix(rot_mats[i]).as_rotvec() for i in range(n_joints)]
        return np.concatenate(axis_angles)

    def load_mano_mean_poses(self):
        print("loading mano mean poses")
        mano_model_dirs = [
            '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/external/WiLoR/mano_data',
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
                    mean_left = mano_left_data['hands_mean']
                    mean_right = mano_right_data['hands_mean']
                    print("   ✅ Successfully loaded 'hands_mean' from both MANO_LEFT and MANO_RIGHT pkl files.")
                    return mean_left, mean_right
                except Exception as e:
                    print(f"   ⚠️  Found files, but failed to load from {directory}: {e}")
        raise FileNotFoundError(
            "Could not find MANO_LEFT.pkl and MANO_RIGHT.pkl. "
            "Please download the MANO models from https://mano.is.tue.mpg.de/ "
            "and place them in one of the searched directories."
        )

    def extract_wilor_hand_poses(self, wilor_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        print("\n🖐️  Extracting WiLoR hand poses (using SUPERVISOR'S a.k.a WiloR author's logic)...")
        hands_mean_left, hands_mean_right = self.load_mano_mean_poses()
        left_hand_pose_param = np.zeros(45)
        right_hand_pose_param = np.zeros(45)

        for hand in wilor_params.get('hands', []):
            if 'mano_parameters' not in hand:
                continue
            mano_params = hand['mano_parameters']['parameters']
            hand_type = hand.get('hand_type', '')
            if 'hand_pose' in mano_params:
                rot_matrices = np.array(mano_params['hand_pose']['values']).flatten()
                pose_parameter = self.convert_rotation_matrix_to_axis_angle(rot_matrices)
                if hand_type == 'left':
                    print("   Applying supervisor's special transform for LEFT hand...")
                    final_param = pose_parameter.copy()
                    final_param *= -1.0
                    final_param[::3] *= -1.0
                    left_hand_pose_param = final_param - hands_mean_left
                    print(f"   ✅ Processed LEFT hand with supervisor's logic.")
                elif hand_type == 'right':
                    right_hand_pose_param = pose_parameter - hands_mean_right
                    print(f"   ✅ Processed RIGHT hand.")
        return left_hand_pose_param, right_hand_pose_param
    
    # --- NEW: Correctly maps both expression and jaw pose from EMOCA ---
    def map_emoca_parameters(self, emoca_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        CORRECTED: Extracts both full expression AND jaw pose from EMOCA's output.
        This follows the supervisor's guidance to use all available parameters.
        """
        if not emoca_params:
            print("   ⚠️  No EMOCA data found, using neutral face and jaw.")
            return np.zeros(50), np.zeros(3) # Return neutral for both

        print("\n🎭 Mapping full EMOCA face parameters...")
        
        # 1. Extract FULL expression vector
        if 'expression' in emoca_params:
            emoca_exp = np.array(emoca_params['expression'])
            mapped_exp = emoca_exp[:10] * self.expression_scale
            print(f"   ✅ Extracted {len(mapped_exp)} expression parameters.")
        else:
            mapped_exp = np.zeros(50)
            print("   ⚠️  'expression' not in EMOCA params, using neutral expression.")
            
        # 2. Extract Jaw Pose from 'posecode'
        if 'pose' in emoca_params:
            emoca_pose = np.array(emoca_params['pose'])
            if len(emoca_pose) == 6:
                mapped_jaw = emoca_pose[3:] # Last 3 values are the jaw pose
                print(f"   ✅ Extracted jaw pose from 'posecode'.")
            else:
                mapped_jaw = np.zeros(3)
                print(f"   ⚠️  'pose' has unexpected length {len(emoca_pose)}, using neutral jaw.")
        else:
            mapped_jaw = np.zeros(3)
            print(f"   ⚠️  'pose' not in EMOCA params, using neutral jaw.")

        return mapped_exp, mapped_jaw

    # --- UPDATED: Integrates the full face fusion ---
    def create_fused_parameters(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict) -> Dict:
        print("\n🔧 Creating fused parameters (with full hand and face fusion)...")
        fused = {
            'betas': np.array(smplx_params['betas']),
            'body_pose': np.array(smplx_params['body_pose']),
            'root_pose': np.array(smplx_params['root_pose']),
            'translation': np.array(smplx_params['translation']),
        }
        
        # Hand Fusion
        left_hand, right_hand = self.extract_wilor_hand_poses(wilor_params)
        fused['left_hand_pose'] = left_hand
        fused['right_hand_pose'] = right_hand
        print("\n   ✅ Hands: FULL WiLoR replacement")
        
        # Face Fusion (now replaces both expression and jaw pose)
        fused_expression, fused_jaw_pose = self.map_emoca_parameters(emoca_params)
        fused['expression'] = fused_expression
        fused['jaw_pose'] = fused_jaw_pose
        print("   ✅ Face: FULL EMOCA replacement (expression + jaw pose)")

        # Diagnostic diffs
        left_diff = np.linalg.norm(left_hand - np.array(smplx_params['left_hand_pose']))
        right_diff = np.linalg.norm(right_hand - np.array(smplx_params['right_hand_pose']))
        jaw_diff = np.linalg.norm(fused_jaw_pose - np.array(smplx_params['jaw_pose']))
        print(f"\n   📊 Changes from original:")
        print(f"      Left hand norm diff: {left_diff:.3f}")
        print(f"      Right hand norm diff: {right_diff:.3f}")
        print(f"      Jaw pose norm diff: {jaw_diff:.3f}")
        return fused

    # --- UPDATED: Uses new EMOCA mapping for diagnostics ---
    def create_individual_parameter_sets(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict, fused_params: Dict) -> Dict:
        """Create individual parameter sets for each model's contribution"""
        print("\n🎯 Creating individual parameter sets for rendering...")
        
        smplestx_only = smplx_params.copy()
        
        emoca_contribution = smplx_params.copy()
        emoca_exp, emoca_jaw = self.map_emoca_parameters(emoca_params)
        emoca_contribution['expression'] = emoca_exp
        emoca_contribution['jaw_pose'] = emoca_jaw
        
        wilor_contribution = smplx_params.copy()
        left_hand, right_hand = self.extract_wilor_hand_poses(wilor_params)
        wilor_contribution['left_hand_pose'] = left_hand
        wilor_contribution['right_hand_pose'] = right_hand
        
        print("   ✅ Individual parameter sets created")
        
        return {
            'smplestx': smplestx_only,
            'emoca': emoca_contribution,
            'wilor': wilor_contribution,
            'fusion': fused_params
        }

    # --- UPDATED: Handles variable expression dimensions ---
    def generate_mesh(self, params: Dict, use_translation: bool = True) -> np.ndarray:
        print(f"\n🎯 Generating mesh...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # # Ensure the SMPL-X layer can handle the number of expression coefficients we provide
        # num_coeffs = len(params['expression'])
        # self.smplx_model.layer['neutral'].num_expression_coeffs = num_coeffs
        smplx_layer = self.smplx_model.layer['neutral'].to(device)
        # print(f"   SMPL-X layer configured for {num_coeffs} expression coefficients.")

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

    def render_mesh_on_image(self, img: np.ndarray, mesh: np.ndarray, cam_params: Dict, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        return render_mesh(img.copy(), mesh, self.smplx_model.face, cam_params, mesh_as_vertices=False, color=color)

    def create_render_gallery(self, individual_params: Dict):
        print("\n🎨 Creating comprehensive render gallery...")
        
        img_path = None
        for temp_dir in [self.results_dir / 'wilor_results' / 'temp_input', 
                        self.results_dir / 'emoca_results' / 'temp_input',
                        self.results_dir / 'smplestx_results' / 'temp_input']:
            if temp_dir.exists():
                for img_file in temp_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_path = img_file; break
                if img_path: break
        
        if not img_path:
            print("   ⚠️  No input image found for rendering"); return
        
        input_img = cv2.imread(str(img_path)); h, w = input_img.shape[:2]
        
        if not hasattr(self, 'camera_metadata'):
            print("   ⚠️  Camera metadata not loaded."); return
        
        cam_params = {'focal': np.array(self.camera_metadata['focal_length']), 'princpt': np.array(self.camera_metadata['principal_point'])}
        
        print("   🔧 Generating individual meshes for gallery...")
        meshes = {name: self.generate_mesh(params) for name, params in individual_params.items()}
        
        colors = {'smplestx': (255, 255, 255), 'emoca': (255, 0, 255), 'wilor': (255, 0, 0), 'fusion': (0, 255, 255)}
        
        cv2.imwrite(str(self.gallery_dir / '1_input.png'), input_img)
        smplestx_overlay = self.render_mesh_on_image(input_img, meshes['smplestx'], cam_params, colors['smplestx'])
        cv2.imwrite(str(self.gallery_dir / '2_smplestx_overlay.png'), smplestx_overlay)
        fusion_overlay = self.render_mesh_on_image(input_img, meshes['fusion'], cam_params, colors['fusion'])
        cv2.imwrite(str(self.gallery_dir / '3_fusion_overlay.png'), fusion_overlay)
        
        print("   🔗 Creating comparison stack...")
        comparison_stack = np.hstack([cv2.resize(input_img, (w, h)), cv2.resize(smplestx_overlay, (w, h)), cv2.resize(fusion_overlay, (w, h))])
        font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        cv2.putText(comparison_stack, 'Input', (10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(comparison_stack, 'SMPLest-X', (w + 10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(comparison_stack, 'Fusion', (2*w + 10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.imwrite(str(self.gallery_dir / 'comparison_stack.png'), comparison_stack)
        
        print(f"   ✅ Gallery complete! Saved to {self.gallery_dir}")

    def validate_parameters(self, params: Dict) -> bool:
        print("\n✅ Validating parameters..."); issues = []
        for pose_name in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
            if np.abs(params[pose_name]).max() > np.pi * 1.5: issues.append(f"{pose_name} has extreme values.")
        if np.abs(params['betas']).max() > 5: issues.append("Shape parameters extreme.")
        if issues: print("   ⚠️  Validation issues:", *issues); return False
        else: print("   ✅ All parameters within reasonable ranges"); return True

    def save_results(self, fused_params: Dict):
        print("\n💾 Saving results...")
        serializable_fused = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fused_params.items()}
        with open(self.fusion_dir / 'fused_parameters.json', 'w') as f: 
            json.dump(serializable_fused, f, indent=2)
        
        # Save OBJ files for Blender inspection (centered at origin)
        original_centered_mesh = self.generate_mesh(self.smplx_params_cache, use_translation=False)
        fused_centered_mesh = self.generate_mesh(fused_params, use_translation=False)
        
        # ADD THIS LINE for evaluator compatibility:
        np.save(self.fusion_dir / 'enhanced_mesh.npy', fused_centered_mesh)
        
        trimesh.Trimesh(vertices=original_centered_mesh, faces=self.smplx_model.face).export(str(self.fusion_dir / 'original_final.obj'))
        trimesh.Trimesh(vertices=fused_centered_mesh, faces=self.smplx_model.face).export(str(self.fusion_dir / 'fused_final.obj'))
        print(f"   ✅ Fused parameters, OBJ files, and enhanced_mesh.npy saved to {self.fusion_dir}")

    def run_fusion(self):
        print("\n" + "=" * 60 + "\n🚀 ENHANCED PARAMETER FUSION SYSTEM\n" + "=" * 60)
        try:
            self.smplx_params_cache, wilor_params, emoca_params = self.load_all_parameters()
            fused_params = self.create_fused_parameters(self.smplx_params_cache, wilor_params, emoca_params)
            individual_params = self.create_individual_parameter_sets(self.smplx_params_cache, wilor_params, emoca_params, fused_params)
            
            is_valid = self.validate_parameters(fused_params)
            self.create_render_gallery(individual_params)
            self.save_results(fused_params)
            
            print("\n" + "=" * 60 + "\n✅ FUSION COMPLETE!\n" + "=" * 60)
            print(f"🎨 Render gallery: {self.gallery_dir}")
            if not is_valid: print("\n⚠️  Warning: Some parameters may be extreme.")
        except Exception as e:
            print(f"\n❌ Fusion failed: {e}"); traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description='Enhanced Parameter Fusion System with Central Gallery Support')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing pipeline results')
    parser.add_argument('--gallery_dir', type=str, help='External gallery directory (for central gallery system)')
    args = parser.parse_args()
    
    fusion = EnhancedParameterFusion(args.results_dir, args.gallery_dir)
    fusion.run_fusion()

if __name__ == '__main__':
    main()