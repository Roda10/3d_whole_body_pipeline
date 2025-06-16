#!/usr/bin/env python3
"""
Enhanced Direct Parameter Fusion System with Central Gallery Support
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
    """Enhanced fusion system with central gallery support (no local gallery creation)"""

    def __init__(self, results_dir: str, gallery_dir: Optional[str] = None):
        self.results_dir = Path(results_dir)
        self.fusion_dir = self.results_dir / 'fusion_results'
        self.fusion_dir.mkdir(exist_ok=True)
        
        # Use provided gallery directory or create local one as fallback
        if gallery_dir:
            self.gallery_dir = Path(gallery_dir)
            self.gallery_dir.mkdir(exist_ok=True)
            print(f"📁 Using external gallery: {self.gallery_dir}")
        else:
            # Fallback for standalone usage
            self.gallery_dir = self.fusion_dir / 'render_gallery'
            self.gallery_dir.mkdir(exist_ok=True)
            print(f"📁 Using local gallery: {self.gallery_dir}")

        self.load_coordinate_analysis()
        self.setup_smplx_model()
        self.setup_rendering_config()

        self.hand_blend_weight = 0.8
        self.expression_scale = 0.3

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
                raise FileNotFoundError("CRITICAL: SMPLest-X camera_metadata.json not found!")
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

    # def create_fused_parameters(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict) -> Dict:
    #     print("\n🔧 Creating fused parameters (CORRECTED)...")
    #     fused = {
    #         'betas': np.array(smplx_params['betas']),
    #         'body_pose': np.array(smplx_params['body_pose']),
    #         'root_pose': np.array(smplx_params['root_pose']),
    #         'translation': np.array(smplx_params['translation']),
    #         'jaw_pose': np.array(smplx_params['jaw_pose'])
    #     }
    #     left_hand, right_hand = self.extract_wilor_hand_poses(wilor_params)
    #     fused['left_hand_pose'] = left_hand
    #     fused['right_hand_pose'] = right_hand
    #     print("\n   ✅ Hands: FULL WiLoR replacement (no mixing)")
    #     fused['expression'] = self.map_emoca_expression(emoca_params)
    #     left_diff = np.linalg.norm(left_hand - np.array(smplx_params['left_hand_pose']))
    #     right_diff = np.linalg.norm(right_hand - np.array(smplx_params['right_hand_pose']))
    #     print(f"\n   📊 Changes from original:")
    #     print(f"      Left hand: {left_diff:.3f}")
    #     print(f"      Right hand: {right_diff:.3f}")
    #     return fused

    # def map_emoca_expression(self, emoca_params: Dict) -> np.ndarray:
    #     print("\n🎭 Mapping EMOCA expression...")
    #     emoca_exp = np.array(emoca_params['expcode'])
    #     mapped_exp = emoca_exp[:10] * self.expression_scale
    #     return mapped_exp


    def create_fused_parameters(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict) -> Dict:
        """
        CORRECTED FUSION: Now integrates EMOCA's jaw pose as well.
        """
        print("\n🔧 Creating fused parameters (with full face fusion)...")
        
        # Start with SMPLest-X base
        fused = {
            'betas': np.array(smplx_params['betas']),
            'body_pose': np.array(smplx_params['body_pose']),
            'root_pose': np.array(smplx_params['root_pose']),
            'translation': np.array(smplx_params['translation']),
        }
        
        # --- HAND FUSION (No change here) ---
        left_hand, right_hand = self.extract_wilor_hand_poses(wilor_params)
        fused['left_hand_pose'] = left_hand
        fused['right_hand_pose'] = right_hand
        
        # --- FACE FUSION (UPDATED) ---
        # Get BOTH expression and jaw pose from our new function
        fused_expression, fused_jaw_pose = self.map_emoca_expression(emoca_params)
        fused['expression'] = fused_expression
        fused['jaw_pose'] = fused_jaw_pose
        
        print("\n   ✅ Face: FULL EMOCA replacement (expression + jaw pose)")
        return fused


    # ==============================================================================
    #  Replace your old map_emoca_expression function with this one
    # ==============================================================================
    def map_emoca_expression(self, emoca_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        CORRECTED: Extracts both full expression AND jaw pose from EMOCA's output.
        This follows the supervisor's guidance to use all available parameters.
        """
        print("\n🎭 Mapping full EMOCA face parameters...")
        emoca_exp = np.array(emoca_params['expcode'])
        # We can still apply a gentle scaling to avoid overly exaggerated expressions
        mapped_exp = emoca_exp[:10] * self.expression_scale 
        print(f" ✅ Extracted {len(mapped_exp)} expression parameters.")

        # 2. Extract Jaw Pose from 'posecode'
        emoca_pose = np.array(emoca_params['posecode'])
        mapped_jaw = emoca_pose[3:] # Take the last 3 values for the jaw
        print(f" ✅ Extracted {len(mapped_jaw)} jaw parameters.")
        return mapped_exp, mapped_jaw


    def create_individual_parameter_sets(self, smplx_params: Dict, wilor_params: Dict, emoca_params: Dict, fused_params: Dict) -> Dict:
        """Create individual parameter sets for each model's contribution"""
        print("\n🎯 Creating individual parameter sets for rendering...")
        
        # Base SMPLest-X parameters (unchanged)
        smplestx_only = smplx_params.copy()
        
        # EMOCA contribution: SMPLest-X base + EMOCA expression
        emoca_contribution = smplx_params.copy()
        emoca_contribution['expression'], emoca_contribution['jaw_pose'] = self.map_emoca_expression(emoca_params)
        
        # WiLoR contribution: SMPLest-X base + WiLoR hands
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

    def render_mesh_on_image(self, img: np.ndarray, mesh: np.ndarray, cam_params: Dict, color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Render mesh on image with specified color"""
        return render_mesh(img.copy(), mesh, self.smplx_model.face, cam_params, mesh_as_vertices=False, color=color)

    def render_standalone_mesh(self, mesh: np.ndarray, img_size: Tuple[int, int] = (512, 512), color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """Render standalone mesh without background image"""
        # Create black background
        img = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)
        
        # Use default camera parameters for standalone rendering
        cam_params = {
            'focal': np.array([1000.0, 1000.0]),
            'princpt': np.array([img_size[1]/2, img_size[0]/2])
        }
        
        return render_mesh(img, mesh, self.smplx_model.face, cam_params, mesh_as_vertices=False, color=color)

    def create_render_gallery(self, individual_params: Dict, original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        """Create comprehensive render gallery in the specified gallery directory"""
        print("\n🎨 Creating comprehensive render gallery...")
        
        # Find input image
        img_path = None
        for temp_dir in [self.results_dir / 'wilor_results' / 'temp_input', 
                        self.results_dir / 'emoca_results' / 'temp_input',
                        self.results_dir / 'smplestx_results' / 'temp_input']:
            if temp_dir.exists():
                for img_file in temp_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        img_path = img_file
                        break
                if img_path:
                    break
        
        if not img_path:
            print("   ⚠️  No input image found for rendering")
            return
        
        # Load input image
        input_img = cv2.imread(str(img_path))
        h, w = input_img.shape[:2]
        
        if not hasattr(self, 'camera_metadata'):
            print("   ⚠️  Camera metadata not loaded.")
            return
        
        cam_params = {
            'focal': np.array(self.camera_metadata['focal_length']),
            'princpt': np.array(self.camera_metadata['principal_point'])
        }
        
        # Generate individual meshes
        print("   🔧 Generating individual meshes...")
        meshes = {}
        for model_name, params in individual_params.items():
            meshes[model_name] = self.generate_mesh(params)
        
        # Define colors for each model (BGR format for OpenCV)
        colors = {
            'smplestx': (255, 255, 255),    # White
            'emoca': (255, 0, 255),         # Pink
            'wilor': (255, 0, 0),           # Blue
            'fusion': (0, 255, 255)         # Yellow
        }
        
        # 1. Save input image
        cv2.imwrite(str(self.gallery_dir / '1_input.png'), input_img)
        print("   ✅ Saved: 1_input.png")
        
        # 2. EMOCA mesh overlay
        emoca_overlay = self.render_mesh_on_image(input_img, meshes['emoca'], cam_params, colors['emoca'])
        cv2.imwrite(str(self.gallery_dir / '2_emoca_overlay.png'), emoca_overlay)
        print("   ✅ Saved: 2_emoca_overlay.png")
        
        # 3. SMPLest-X mesh overlay
        smplestx_overlay = self.render_mesh_on_image(input_img, meshes['smplestx'], cam_params, colors['smplestx'])
        cv2.imwrite(str(self.gallery_dir / '4_smplestx_overlay.png'), smplestx_overlay)
        print("   ✅ Saved: 4_smplestx_overlay.png")
        
        # 4. Fusion mesh overlay
        fusion_overlay = self.render_mesh_on_image(input_img, meshes['fusion'], cam_params, colors['fusion'])
        cv2.imwrite(str(self.gallery_dir / '6_fusion_overlay.png'), fusion_overlay)
        print("   ✅ Saved: 6_fusion_overlay.png")
        
        # Create 3-image comparison stack: Input | SMPLest-X mesh | Fusion mesh
        print("   🔗 Creating comparison stack...")
        
        # Resize images to same height if needed
        target_height = h
        input_resized = cv2.resize(input_img, (w, target_height))
        smplestx_resized = cv2.resize(smplestx_overlay, (w, target_height))
        fusion_resized = cv2.resize(fusion_overlay, (w, target_height))
        
        # Stack horizontally
        comparison_stack = np.hstack([input_resized, smplestx_resized, fusion_resized])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        
        cv2.putText(comparison_stack, 'Input', (10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(comparison_stack, 'SMPLest-X', (w + 10, 30), font, font_scale, (255, 255, 255), thickness)
        cv2.putText(comparison_stack, 'Fusion', (2*w + 10, 30), font, font_scale, (255, 255, 255), thickness)
        
        cv2.imwrite(str(self.gallery_dir / 'comparison_stack.png'), comparison_stack)
        print("   ✅ Saved: comparison_stack.png")
        
        # Create summary info file
        info_content = f"""RENDER GALLERY SUMMARY
========================

Generated render images:
1. 1_input.png - Original input image
2. 2_emoca_overlay.png - EMOCA expression mesh overlaid
4. 4_smplestx_overlay.png - SMPLest-X base mesh overlaid
6. 6_fusion_overlay.png - Final fusion mesh overlaid

Plus comparison:
- comparison_stack.png - Input | SMPLest-X | Fusion side-by-side

Colors used:
- SMPLest-X: White
- EMOCA: Pink  
- WiLoR: Blue
- Fusion: Yellow

Input image: {img_path.name}
Image dimensions: {w}x{h}
Gallery location: {self.gallery_dir}
"""
        
        with open(self.gallery_dir / 'gallery_info.txt', 'w') as f:
            f.write(info_content)
        
        print(f"   ✅ Gallery complete! Saved to {self.gallery_dir}")

    def validate_parameters(self, params: Dict) -> bool:
        print("\n✅ Validating parameters...")
        issues = []
        for pose_name in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
            pose = params[pose_name]
            if np.abs(pose).max() > np.pi * 1.5:
                issues.append(f"{pose_name} has extreme values.")
        if np.abs(params['betas']).max() > 5:
            issues.append("Shape parameters extreme.")
        if issues:
            print("   ⚠️  Validation issues:", *issues)
            return False
        else:
            print("   ✅ All parameters within reasonable ranges")
            return True

    def save_results(self, original_params: Dict, fused_params: Dict, original_mesh: np.ndarray, enhanced_mesh: np.ndarray):
        print("\n💾 Saving results...")
        serializable_fused = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in fused_params.items()}
        with open(self.fusion_dir / 'fused_parameters.json', 'w') as f:
            json.dump(serializable_fused, f, indent=2)
        np.save(self.fusion_dir / 'enhanced_mesh.npy', enhanced_mesh)
        print(f"   ✅ Results saved to {self.fusion_dir}")

    def run_fusion(self):
        print("\n" + "=" * 60 + "\n🚀 ENHANCED PARAMETER FUSION SYSTEM\n" + "=" * 60)
        try:
            smplx_params, wilor_params, emoca_params = self.load_all_parameters()
            fused_params = self.create_fused_parameters(smplx_params, wilor_params, emoca_params)
            
            # Create individual parameter sets for rendering
            individual_params = self.create_individual_parameter_sets(smplx_params, wilor_params, emoca_params, fused_params)
            
            is_valid = self.validate_parameters(fused_params)
            original_mesh = self.generate_mesh(smplx_params)
            enhanced_mesh = self.generate_mesh(fused_params)
            
            # Create comprehensive render gallery
            self.create_render_gallery(individual_params, original_mesh, enhanced_mesh)
            
            # Save results (no mesh comparison or .obj files)
            self.save_results(smplx_params, fused_params, original_mesh, enhanced_mesh)
            
            print("\n" + "=" * 60 + "\n✅ FUSION COMPLETE!\n" + "=" * 60)
            print(f"🎨 Render gallery: {self.gallery_dir}")
            if not is_valid:
                print("\n⚠️  Warning: Some parameters may be extreme.")
        except Exception as e:
            print(f"\n❌ Fusion failed: {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Enhanced Parameter Fusion System with Central Gallery Support')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing pipeline results')
    parser.add_argument('--gallery_dir', type=str, help='External gallery directory (for central gallery system)')
    args = parser.parse_args()
    
    fusion = EnhancedParameterFusion(args.results_dir, args.gallery_dir)
    fusion.run_fusion()


if __name__ == '__main__':
    main()