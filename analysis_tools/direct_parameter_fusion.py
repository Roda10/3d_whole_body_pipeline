# #!/usr/bin/env python3
# """
# Direct Parameter Replacement Fusion System
# Implements actual parameter fusion using coordinate analysis results
# """

# import numpy as np
# import json
# import torch
# import sys
# import os
# from pathlib import Path
# from typing import Dict, Tuple, Optional
# import cv2
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import argparse

# # Add paths for SMPL-X model access
# sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
# from human_models.human_models import SMPLX
# from utils.visualization_utils import render_mesh
# from main.config import Config

# class DirectParameterFusion:
#     """Implements direct parameter replacement fusion with coordinate transformation"""
    
#     def __init__(self, results_dir: str):
#         self.results_dir = Path(results_dir)
#         self.coordinate_analysis = None
#         self.transformation_params = None
#         self.smplx_model = None
#         self.config = None
#         self.load_analysis_results()
#         self.setup_smplx_model()
#         self.setup_rendering_config()
        
#     def load_analysis_results(self):
#         """Load coordinate analysis and transformation parameters"""
#         print("📥 Loading coordinate analysis results...")
        
#         # Load coordinate analysis summary
#         coord_file = self.results_dir / 'coordinate_analysis_summary.json'
#         if coord_file.exists():
#             with open(coord_file, 'r') as f:
#                 self.coordinate_analysis = json.load(f)
            
#             # Extract transformation parameters
#             self.transformation_params = {
#                 'scale_factor': self.coordinate_analysis['transformation_parameters']['scale_factor'],
#                 'translation_vector': np.array(self.coordinate_analysis['transformation_parameters']['translation_vector']),
#                 'homogeneous_matrix': np.array(self.coordinate_analysis['transformation_parameters']['homogeneous_matrix'])
#             }
            
#             print(f"   ✅ Loaded transformation: scale={self.transformation_params['scale_factor']:.4f}")
#             print(f"   ✅ Translation: {self.transformation_params['translation_vector']}")
            
#         else:
#             raise FileNotFoundError(f"Coordinate analysis file not found: {coord_file}")
    
#     def setup_rendering_config(self):
#         """Setup rendering configuration similar to smplestx_adapter.py"""
#         print("🎨 Setting up rendering configuration...")
        
#         # Try to find config file
#         config_paths = [
#             'pretrained_models/smplest_x/config_base.py',
#             '../pretrained_models/smplest_x/config_base.py',
#             'external/SMPLest-X/configs/config_smplest_x_h.py'
#         ]
        
#         config_path = None
#         for path in config_paths:
#             if os.path.exists(path):
#                 config_path = path
#                 break
        
#         if config_path is None:
#             print("   ⚠️  Config file not found, using default rendering parameters")
#             # Use default values similar to SMPLest-X
#             self.config = type('Config', (), {
#                 'model': type('Model', (), {
#                     'focal': [5000.0, 5000.0],
#                     'princpt': [128.0, 128.0],
#                     'input_body_shape': [256, 256],
#                     'input_img_shape': [256, 256]
#                 })()
#             })()
#         else:
#             try:
#                 self.config = Config.load_config(config_path)
#                 print(f"   ✅ Config loaded from: {config_path}")
#             except Exception as e:
#                 print(f"   ⚠️  Config loading failed: {e}, using defaults")
#                 self.config = type('Config', (), {
#                     'model': type('Model', (), {
#                         'focal': [5000.0, 5000.0],
#                         'princpt': [128.0, 128.0],
#                         'input_body_shape': [256, 256],
#                         'input_img_shape': [256, 256]
#                     })()
#                 })()
    
#     def render_enhanced_mesh(self, enhanced_mesh: np.ndarray, original_mesh: np.ndarray, 
#                            bbox: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
#         """Render both original and enhanced meshes onto the original input image"""
#         if enhanced_mesh is None:
#             print("⚠️  Cannot render enhanced mesh - mesh generation failed")
#             return None, None
        
#         print("🎨 Rendering meshes onto original input image...")
        
#         try:
#             # Load the original input image from the pipeline results
#             original_img = self.load_original_input_image()
#             if original_img is None:
#                 print("   ⚠️  Could not load original image, using blank canvas")
#                 img_size = 512
#                 original_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
#             # Load bbox from SMPLest-X results if available
#             if bbox is None:
#                 bbox = self.load_bbox_from_smplestx_results()
#                 if bbox is None:
#                     # Create default bbox if we can't find the original one
#                     h, w = original_img.shape[:2]
#                     bbox = np.array([w//4, h//4, w//2, h//2])  # [x, y, w, h]
            
#             print(f"   📦 Using bbox: {bbox}")
            
#             # Calculate camera parameters exactly like smplestx_adapter.py
#             focal = [self.config.model.focal[0] / self.config.model.input_body_shape[1] * bbox[2], 
#                     self.config.model.focal[1] / self.config.model.input_body_shape[0] * bbox[3]]
#             princpt = [self.config.model.princpt[0] / self.config.model.input_body_shape[1] * bbox[2] + bbox[0], 
#                       self.config.model.princpt[1] / self.config.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
#             camera_params = {'focal': focal, 'princpt': princpt}
            
#             print(f"   📷 Camera parameters: focal={focal}, princpt={princpt}")
            
#             # Create copies of the original image for rendering
#             original_vis_img = original_img.copy()
#             enhanced_vis_img = original_img.copy()
            
#             # Draw bbox on both images (like in smplestx_adapter.py)
#             x1, y1 = int(bbox[0]), int(bbox[1])
#             x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            
#             original_vis_img = cv2.rectangle(original_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             enhanced_vis_img = cv2.rectangle(enhanced_vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
#             # Render original mesh onto image
#             original_vis_img = render_mesh(original_vis_img, original_mesh, self.smplx_model.face, 
#                                          camera_params, mesh_as_vertices=False)
            
#             # Render enhanced mesh onto image  
#             enhanced_vis_img = render_mesh(enhanced_vis_img, enhanced_mesh, self.smplx_model.face, 
#                                          camera_params, mesh_as_vertices=False)
            
#             print("   ✅ Meshes rendered onto original image successfully")
#             return original_vis_img, enhanced_vis_img
            
#         except Exception as e:
#             print(f"   ❌ Mesh rendering failed: {e}")
#             import traceback
#             traceback.print_exc()
#             return None, None
    
#     def load_original_input_image(self) -> Optional[np.ndarray]:
#         """Load the original input image from pipeline results"""
#         try:
#             # Check multiple possible locations for the original image
#             image_sources = [
#                 # From WiLoR temp input
#                 self.results_dir / 'wilor_results' / 'temp_input',
#                 # From EMOCA temp input
#                 self.results_dir / 'emoca_results' / 'temp_input',
#                 # From pipeline summary
#                 self.results_dir
#             ]
            
#             # Try to get image path from pipeline summary
#             summary_file = self.results_dir / 'pipeline_summary.json'
#             if summary_file.exists():
#                 with open(summary_file, 'r') as f:
#                     summary = json.load(f)
#                     input_image_path = summary.get('input_image', '')
#                     if input_image_path and os.path.exists(input_image_path):
#                         print(f"   📷 Loading original image from: {input_image_path}")
#                         return cv2.imread(input_image_path)
            
#             # Search in temp directories
#             for source_dir in image_sources:
#                 if source_dir.exists():
#                     for img_file in source_dir.glob('*.jpg'):
#                         print(f"   📷 Loading original image from: {img_file}")
#                         return cv2.imread(str(img_file))
#                     for img_file in source_dir.glob('*.png'):
#                         print(f"   📷 Loading original image from: {img_file}")
#                         return cv2.imread(str(img_file))
            
#             print("   ⚠️  Could not find original input image")
#             return None
            
#         except Exception as e:
#             print(f"   ⚠️  Error loading original image: {e}")
#             return None
    
#     def load_bbox_from_smplestx_results(self) -> Optional[np.ndarray]:
#         """Load bbox information from SMPLest-X results"""
#         try:
#             # Look for SMPLest-X parameter files that might contain bbox info
#             for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_shapes_*.json'):
#                 with open(param_file, 'r') as f:
#                     shapes_data = json.load(f)
#                     if 'bbox_xyxy' in shapes_data:
#                         bbox_xyxy = shapes_data['bbox_xyxy']
#                         # Convert from xyxy to xywh format
#                         bbox = np.array([bbox_xyxy[0], bbox_xyxy[1], 
#                                        bbox_xyxy[2] - bbox_xyxy[0], 
#                                        bbox_xyxy[3] - bbox_xyxy[1]])
#                         print(f"   📦 Loaded bbox from SMPLest-X: {bbox}")
#                         return bbox
            
#             print("   ⚠️  Could not find bbox in SMPLest-X results")
#             return None
            
#         except Exception as e:
#             print(f"   ⚠️  Error loading bbox: {e}")
#             return None
    
#     def create_mesh_comparison_image(self, original_rendered: np.ndarray, 
#                                    enhanced_rendered: np.ndarray) -> np.ndarray:
#         """Create side-by-side comparison image"""
#         if original_rendered is None or enhanced_rendered is None:
#             return None
        
#         print("🖼️  Creating mesh comparison image...")
        
#         # Create side-by-side comparison
#         h, w = original_rendered.shape[:2]
#         comparison_img = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
#         comparison_img.fill(255)  # White background
        
#         # Place images side by side
#         comparison_img[:, :w] = original_rendered
#         comparison_img[:, w+20:] = enhanced_rendered
        
#         # Add labels
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         font_scale = 1
#         color = (0, 0, 0)  # Black text
#         thickness = 2
        
#         cv2.putText(comparison_img, 'Original (SMPLest-X)', (10, 30), font, font_scale, color, thickness)
#         cv2.putText(comparison_img, 'Enhanced (Fused)', (w + 30, 30), font, font_scale, color, thickness)
        
#         print("   ✅ Comparison image created")
#         return comparison_img
    
#     def load_model_parameters(self) -> Tuple[Dict, Dict, Dict]:
#         """Load parameters from all three models"""
#         print("📥 Loading model parameters...")
        
#         # Load SMPLest-X parameters
#         smplx_data = None
#         for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
#             with open(param_file, 'r') as f:
#                 smplx_data = json.load(f)
#             break
        
#         if smplx_data is None:
#             raise FileNotFoundError("SMPLest-X parameters not found")
        
#         # Load WiLoR parameters
#         wilor_data = None
#         for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
#             with open(param_file, 'r') as f:
#                 wilor_data = json.load(f)
#             break
        
#         if wilor_data is None:
#             raise FileNotFoundError("WiLoR parameters not found")
        
#         # Load EMOCA parameters
#         emoca_data = None
#         for param_file in self.results_dir.glob('emoca_results/*/codes.json'):
#             with open(param_file, 'r') as f:
#                 emoca_data = json.load(f)
#             break
        
#         if emoca_data is None:
#             print("⚠️  EMOCA parameters not found, proceeding without facial enhancement")
#             emoca_data = {}
        
#         print(f"   ✅ Loaded SMPLest-X: {len(smplx_data)} parameter types")
#         print(f"   ✅ Loaded WiLoR: {len(wilor_data.get('hands', []))} hands")
#         print(f"   ✅ Loaded EMOCA: {len(emoca_data)} parameter types")
        
#         return smplx_data, wilor_data, emoca_data
    
#     def transform_wilor_coordinates(self, wilor_data: Dict) -> Dict:
#         """Apply coordinate transformation to WiLoR hand data"""
#         print("🔄 Transforming WiLoR coordinates to SMPLest-X space...")
        
#         scale = self.transformation_params['scale_factor']
#         translation = self.transformation_params['translation_vector']
        
#         transformed_hands = []
        
#         for hand in wilor_data.get('hands', []):
#             transformed_hand = hand.copy()
            
#             # Transform 3D vertices
#             if 'vertices_3d' in hand:
#                 vertices = np.array(hand['vertices_3d'])
#                 transformed_vertices = vertices * scale + translation
#                 transformed_hand['vertices_3d'] = transformed_vertices.tolist()
                
#                 print(f"   🖐️  Transformed {hand['hand_type']} hand vertices: scale={scale:.4f}")
#                 print(f"      📏 Original range: [{vertices.min():.4f}, {vertices.max():.4f}]")
#                 print(f"      📏 Transformed range: [{transformed_vertices.min():.4f}, {transformed_vertices.max():.4f}]")
            
#             # Transform 3D keypoints
#             if 'keypoints_3d' in hand:
#                 keypoints = np.array(hand['keypoints_3d'])
#                 transformed_keypoints = keypoints * scale + translation
#                 transformed_hand['keypoints_3d'] = transformed_keypoints.tolist()
            
#             transformed_hands.append(transformed_hand)
        
#         transformed_wilor = wilor_data.copy()
#         transformed_wilor['hands'] = transformed_hands
#         transformed_wilor['transformation_applied'] = {
#             'scale_factor': scale,
#             'translation_vector': translation.tolist(),
#             'coordinate_system': 'SMPLest-X_aligned'
#         }
        
#         return transformed_wilor
    
#     def map_emoca_expression(self, emoca_data: Dict, target_dim: int = 10) -> np.ndarray:
#         """Map EMOCA 50D expression to SMPL-X 10D using PCA"""
#         if not emoca_data or 'expcode' not in emoca_data:
#             print("⚠️  No EMOCA expression data, using zero expression")
#             return np.zeros(target_dim)
        
#         print(f"🎭 Mapping EMOCA expression: 50D → {target_dim}D...")
        
#         # Get EMOCA expression vector
#         emoca_exp = np.array(emoca_data['expcode'])
        
#         # For now, use simple truncation/padding approach
#         # TODO: Could be improved with learned mapping
#         if len(emoca_exp) >= target_dim:
#             # Take first target_dim components (assumes PCA-ordered importance)
#             mapped_exp = emoca_exp[:target_dim]
#         else:
#             # Pad with zeros if needed
#             mapped_exp = np.pad(emoca_exp, (0, target_dim - len(emoca_exp)))
        
#         # Normalize to reasonable range for SMPL-X
#         mapped_exp = mapped_exp * 0.5  # Scale down to prevent extreme expressions
        
#         print(f"   ✅ Expression mapped: range=[{mapped_exp.min():.4f}, {mapped_exp.max():.4f}]")
        
#         return mapped_exp
    
#     def extract_hand_pose_parameters(self, wilor_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
#         """Extract hand pose parameters from transformed WiLoR data"""
#         print("🖐️  Extracting hand pose parameters...")
        
#         left_hand_pose = np.zeros(45)   # 15 joints × 3 axis-angle
#         right_hand_pose = np.zeros(45)
        
#         for hand in wilor_data.get('hands', []):
#             hand_type = hand.get('hand_type', '')
            
#             # Check if MANO parameters are available
#             if 'mano_parameters' in hand and 'parameters' in hand['mano_parameters']:
#                 mano_params = hand['mano_parameters']['parameters']
                
#                 # Extract hand pose (finger joint rotations)
#                 if 'hand_pose' in mano_params:
#                     pose_data = mano_params['hand_pose']['values']
#                     pose_array = np.array(pose_data).flatten()
                    
#                     if hand_type == 'left' and len(pose_array) >= 45:
#                         left_hand_pose = pose_array[:45]
#                         print(f"   ✅ Left hand pose extracted: {len(pose_array)} → 45 params")
#                     elif hand_type == 'right' and len(pose_array) >= 45:
#                         right_hand_pose = pose_array[:45]
#                         print(f"   ✅ Right hand pose extracted: {len(pose_array)} → 45 params")
                
#             else:
#                 print(f"   ⚠️  No MANO parameters for {hand_type} hand, using zero pose")
        
#         return left_hand_pose, right_hand_pose
    
#     def create_fused_parameters(self, smplx_data: Dict, wilor_data: Dict, emoca_data: Dict) -> Dict:
#         """Create fused parameter set replacing hand poses and expressions"""
#         print("🔧 Creating fused parameters...")
        
#         # Start with SMPLest-X as foundation
#         fused_params = {
#             # Keep body structure from SMPLest-X
#             'betas': np.array(smplx_data['betas']),
#             'body_pose': np.array(smplx_data['body_pose']),
#             'root_pose': np.array(smplx_data['root_pose']),
#             'translation': np.array(smplx_data['translation']),
#             'jaw_pose': np.array(smplx_data['jaw_pose']),
            
#             # Will be replaced with enhanced versions
#             'left_hand_pose': np.array(smplx_data['left_hand_pose']),
#             'right_hand_pose': np.array(smplx_data['right_hand_pose']),
#             'expression': np.array(smplx_data['expression'])
#         }
        
#         print("   ✅ Foundation parameters from SMPLest-X")
        
#         # Transform WiLoR coordinates
#         transformed_wilor = self.transform_wilor_coordinates(wilor_data)
        
#         # Replace hand poses with WiLoR parameters
#         left_hand_pose, right_hand_pose = self.extract_hand_pose_parameters(transformed_wilor)
#         fused_params['left_hand_pose'] = left_hand_pose
#         fused_params['right_hand_pose'] = right_hand_pose
        
#         print("   ✅ Hand poses replaced with WiLoR parameters")
        
#         # Replace expression with EMOCA mapping
#         mapped_expression = self.map_emoca_expression(emoca_data)
#         fused_params['expression'] = mapped_expression
        
#         print("   ✅ Expression replaced with EMOCA parameters")
        
#         # Store transformation metadata
#         fused_params['fusion_metadata'] = {
#             'source_body': 'SMPLest-X',
#             'source_hands': 'WiLoR_transformed',
#             'source_expression': 'EMOCA_mapped',
#             'transformation_applied': True,
#             'coordinate_system': 'SMPLest-X_space'
#         }
        
#         return fused_params
    
#     def generate_enhanced_mesh(self, fused_params: Dict) -> Optional[np.ndarray]:
#         """Generate mesh using fused parameters"""
#         if self.smplx_model is None:
#             print("⚠️  SMPL-X model not available, cannot generate mesh")
#             return None
        
#         print("🎯 Generating enhanced mesh with fused parameters...")
        
#         try:
#             # Convert parameters to torch tensors
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
#             # Prepare parameters for SMPL-X model (using neutral gender)
#             smplx_layer = self.smplx_model.layer['neutral'].to(device)
            
#             # Convert numpy arrays to torch tensors with proper shapes for SMPL-X
#             # Include all required parameters for SMPL-X
#             torch_params = {
#                 'betas': torch.tensor(fused_params['betas']).float().unsqueeze(0).to(device),
#                 'body_pose': torch.tensor(fused_params['body_pose']).float().unsqueeze(0).to(device),
#                 'global_orient': torch.tensor(fused_params['root_pose']).float().unsqueeze(0).to(device),
#                 'left_hand_pose': torch.tensor(fused_params['left_hand_pose']).float().unsqueeze(0).to(device),
#                 'right_hand_pose': torch.tensor(fused_params['right_hand_pose']).float().unsqueeze(0).to(device),
#                 'expression': torch.tensor(fused_params['expression']).float().unsqueeze(0).to(device),
#                 'jaw_pose': torch.tensor(fused_params['jaw_pose']).float().unsqueeze(0).to(device),
#                 'transl': torch.tensor(fused_params['translation']).float().unsqueeze(0).to(device),
#                 # Add missing eye pose parameters (set to zero since we don't have eye tracking)
#                 'leye_pose': torch.zeros(1, 3).float().to(device),
#                 'reye_pose': torch.zeros(1, 3).float().to(device)
#             }
            
#             print(f"   🔧 Parameter shapes:")
#             for name, tensor in torch_params.items():
#                 print(f"      {name}: {tensor.shape}")
            
#             # Generate mesh using SMPL-X layer
#             with torch.no_grad():
#                 output = smplx_layer(**torch_params)
#                 enhanced_mesh = output.vertices[0].detach().cpu().numpy()
            
#             print(f"   ✅ Enhanced mesh generated: {enhanced_mesh.shape[0]} vertices")
#             return enhanced_mesh
            
#         except Exception as e:
#             print(f"   ❌ Mesh generation failed: {e}")
#             print(f"   Debug info:")
#             print(f"      SMPL-X model type: {type(self.smplx_model)}")
#             print(f"      Available layers: {list(self.smplx_model.layer.keys()) if hasattr(self.smplx_model, 'layer') else 'No layer attribute'}")
#             if hasattr(self.smplx_model, 'layer'):
#                 print(f"      Layer type: {type(self.smplx_model.layer['neutral'])}")
            
#             # Print expected parameters for debugging
#             try:
#                 import inspect
#                 sig = inspect.signature(smplx_layer.forward)
#                 print(f"      Expected parameters: {list(sig.parameters.keys())}")
#             except:
#                 print("      Could not inspect expected parameters")
            
#             return None
    
#     def compare_parameters(self, original_params: Dict, fused_params: Dict):
#         """Compare original and fused parameters"""
#         print("📊 Parameter comparison:")
        
#         comparisons = [
#             ('left_hand_pose', 'Left Hand'),
#             ('right_hand_pose', 'Right Hand'), 
#             ('expression', 'Expression')
#         ]
        
#         for param_name, display_name in comparisons:
#             orig = original_params[param_name]
#             fused = fused_params[param_name]
            
#             orig_norm = np.linalg.norm(orig)
#             fused_norm = np.linalg.norm(fused)
#             difference = np.linalg.norm(orig - fused)
            
#             print(f"   {display_name}:")
#             print(f"     Original norm: {orig_norm:.4f}")
#             print(f"     Fused norm: {fused_norm:.4f}")
#             print(f"     Difference: {difference:.4f}")
#             print(f"     Change: {((fused_norm - orig_norm) / orig_norm * 100):.1f}%")
    
#     def save_results(self, original_params: Dict, fused_params: Dict, 
#                     enhanced_mesh: Optional[np.ndarray]):
#         """Save fusion results"""
#         output_dir = self.results_dir / 'fusion_results'
#         output_dir.mkdir(exist_ok=True)
        
#         print(f"💾 Saving fusion results to: {output_dir}")
        
#         # Save fused parameters
#         serializable_fused = {}
#         for key, value in fused_params.items():
#             if isinstance(value, np.ndarray):
#                 serializable_fused[key] = value.tolist()
#             else:
#                 serializable_fused[key] = value
        
#         with open(output_dir / 'fused_parameters.json', 'w') as f:
#             json.dump(serializable_fused, f, indent=2)

#         # Save comparison report
#         with open(output_dir / 'parameter_comparison.txt', 'w') as f:
#             f.write("PARAMETER FUSION COMPARISON REPORT\n")
#             f.write("==================================\n\n")
            
#             for param_name in ['left_hand_pose', 'right_hand_pose', 'expression']:
#                 orig = np.array(original_params[param_name])
#                 fused = np.array(fused_params[param_name])
                
#                 f.write(f"{param_name.replace('_', ' ').title()}:\n")
#                 f.write(f"  Original norm: {np.linalg.norm(orig):.6f}\n")
#                 f.write(f"  Fused norm: {np.linalg.norm(fused):.6f}\n")
#                 f.write(f"  Difference norm: {np.linalg.norm(orig - fused):.6f}\n")
#                 f.write(f"  Max absolute change: {np.abs(orig - fused).max():.6f}\n")
#                 f.write(f"  Mean absolute change: {np.abs(orig - fused).mean():.6f}\n\n")
        
#         print("   ✅ Fused parameters saved")
#         if enhanced_mesh is not None:
#             print("   ✅ Enhanced mesh saved")
#         print("   ✅ Comparison report saved")
        
#     def setup_smplx_model(self):
#         """Initialize SMPL-X model for mesh generation"""
#         print("🤖 Setting up SMPL-X model...")
        
#         # Try to find SMPL-X model path
#         possible_paths = [
#             '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/pretrained_models/smplx',
#             './pretrained_models/smplx',
#             '../pretrained_models/smplx',
#             'human_models/human_model_files/',
#             './human_models/human_model_files/'
#         ]
        
#         smplx_path = None
#         for path in possible_paths:
#             if os.path.exists(path):
#                 smplx_path = path
#                 break
        
#         if smplx_path is None:
#             print("⚠️  SMPL-X model path not found. Will use parameter-only fusion.")
#             self.smplx_model = None
#         else:
#             try:
#                 self.smplx_model = SMPLX(smplx_path)
#                 print(f"   ✅ SMPL-X model loaded from: {smplx_path}")
#             except Exception as e:
#                 print(f"   ⚠️  Could not load SMPL-X model: {e}")
#                 self.smplx_model = None
        
    
#     def run_fusion(self):
#         """Execute the complete fusion process"""
#         print("\n" + "="*60)
#         print("🚀 DIRECT PARAMETER FUSION SYSTEM")
#         print("="*60 + "\n")
        
#         # Load all model parameters
#         smplx_data, wilor_data, emoca_data = self.load_model_parameters()
        
#         # Create fused parameters
#         fused_params = self.create_fused_parameters(smplx_data, wilor_data, emoca_data)
        
#         # Compare parameters
#         self.compare_parameters(smplx_data, fused_params)
        
#         # Generate enhanced mesh
#         enhanced_mesh = self.generate_enhanced_mesh(fused_params)
        
#         # Save results
#         self.save_results(smplx_data, fused_params, enhanced_mesh)
        
#         print("\n" + "="*60)
#         print("✅ FUSION COMPLETE!")
#         print("="*60)
#         print("\n🎯 Results:")
#         print("   - Fused parameters with WiLoR hands + EMOCA expression")
#         print("   - Coordinate transformation applied")
#         print("   - Enhanced mesh generated (if SMPL-X model available)")
#         print("   - Comparison metrics computed")
#         print(f"\n📁 Output directory: {self.results_dir / 'fusion_results'}")
        
#         return fused_params, enhanced_mesh

# def main():
#     parser = argparse.ArgumentParser(description='Direct Parameter Fusion System')
#     parser.add_argument('--results_dir', type=str, required=True,
#                        help='Directory containing pipeline results and coordinate analysis')
    
#     args = parser.parse_args()
    
#     # Validate input directory
#     results_path = Path(args.results_dir)
#     if not results_path.exists():
#         print(f"❌ Error: Results directory not found: {results_path}")
#         return
    
#     # Check for required analysis files
#     coord_file = results_path / 'coordinate_analysis_summary.json'
#     if not coord_file.exists():
#         print(f"❌ Error: Coordinate analysis not found. Run coordinate analyzer first.")
#         print(f"   Expected: {coord_file}")
#         return
    
#     # Run fusion
#     fusion_system = DirectParameterFusion(args.results_dir)
#     fused_params, enhanced_mesh = fusion_system.run_fusion()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
"""
Enhanced Direct Parameter Replacement Fusion System
Implements actual parameter fusion with mesh rendering and fixed EMOCA loading
"""

import numpy as np
import json
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

# Add paths for SMPL-X model access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
from human_models.human_models import SMPLX
from utils.visualization_utils import render_mesh
from main.config import Config

class EnhancedDirectParameterFusion:
    """Enhanced fusion with proper mesh rendering and fixed EMOCA loading"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.coordinate_analysis = None
        self.transformation_params = None
        self.smplx_model = None
        self.config = None
        self.fusion_output_dir = None
        self.load_analysis_results()
        self.setup_smplx_model()
        self.setup_rendering_config()
        
    def load_analysis_results(self):
        """Load coordinate analysis and transformation parameters"""
        print("📥 Loading coordinate analysis results...")
        
        # Load coordinate analysis summary
        coord_file = self.results_dir / 'coordinate_analysis_summary.json'
        if coord_file.exists():
            with open(coord_file, 'r') as f:
                self.coordinate_analysis = json.load(f)
            
            # Extract transformation parameters
            self.transformation_params = {
                'scale_factor': self.coordinate_analysis['transformation_parameters']['scale_factor'],
                'translation_vector': np.array(self.coordinate_analysis['transformation_parameters']['translation_vector']),
                'homogeneous_matrix': np.array(self.coordinate_analysis['transformation_parameters']['homogeneous_matrix'])
            }
            
            print(f"   ✅ Loaded transformation: scale={self.transformation_params['scale_factor']:.4f}")
            print(f"   ✅ Translation: {self.transformation_params['translation_vector']}")
            
        else:
            raise FileNotFoundError(f"Coordinate analysis file not found: {coord_file}")
    
    def setup_rendering_config(self):
        """Setup rendering configuration similar to smplestx_adapter.py"""
        print("🎨 Setting up rendering configuration...")
        
        # Try to find config file
        config_paths = [
            'pretrained_models/smplest_x/config_base.py',
            '../pretrained_models/smplest_x/config_base.py',
            'external/SMPLest-X/configs/config_smplest_x_h.py'
        ]
        
        config_path = None
        for path in config_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if config_path is None:
            print("   ⚠️  Config file not found, using default rendering parameters")
            # Use default values similar to SMPLest-X
            self.config = type('Config', (), {
                'model': type('Model', (), {
                    'focal': [5000.0, 5000.0],
                    'princpt': [128.0, 128.0],
                    'input_body_shape': [256, 256],
                    'input_img_shape': [256, 256]
                })()
            })()
        else:
            try:
                self.config = Config.load_config(config_path)
                print(f"   ✅ Config loaded from: {config_path}")
            except Exception as e:
                print(f"   ⚠️  Config loading failed: {e}, using defaults")
                self.config = type('Config', (), {
                    'model': type('Model', (), {
                        'focal': [5000.0, 5000.0],
                        'princpt': [128.0, 128.0],
                        'input_body_shape': [256, 256],
                        'input_img_shape': [256, 256]
                    })()
                })()
    
    def load_original_input_image(self) -> Optional[np.ndarray]:
        """Load the original input image from pipeline results"""
        try:
            # Check multiple possible locations for the original image
            image_sources = [
                # From WiLoR temp input
                self.results_dir / 'wilor_results' / 'temp_input',
                # From EMOCA temp input
                self.results_dir / 'emoca_results' / 'temp_input',
                # From pipeline summary
                self.results_dir
            ]
            
            # Try to get image path from pipeline summary
            summary_file = self.results_dir / 'pipeline_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                    input_image_path = summary.get('input_image', '')
                    if input_image_path and os.path.exists(input_image_path):
                        print(f"   📷 Loading original image from: {input_image_path}")
                        img = cv2.imread(input_image_path)
                        if img is not None:
                            return img
            
            # Search in temp directories
            for source_dir in image_sources:
                if source_dir.exists():
                    for img_file in source_dir.glob('*.jpg'):
                        print(f"   📷 Loading original image from: {img_file}")
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            return img
                    for img_file in source_dir.glob('*.png'):
                        print(f"   📷 Loading original image from: {img_file}")
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            return img
            
            print("   ⚠️  Could not find original input image")
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error loading original image: {e}")
            return None
    
    def load_bbox_from_smplestx_results(self) -> Optional[np.ndarray]:
        """Load bbox information from SMPLest-X results"""
        try:
            # Look for SMPLest-X parameter files that might contain bbox info
            for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_shapes_*.json'):
                with open(param_file, 'r') as f:
                    shapes_data = json.load(f)
                    if 'bbox_xyxy' in shapes_data:
                        bbox_xyxy = shapes_data['bbox_xyxy']
                        # Convert from xyxy to xywh format
                        bbox = np.array([bbox_xyxy[0], bbox_xyxy[1], 
                                       bbox_xyxy[2] - bbox_xyxy[0], 
                                       bbox_xyxy[3] - bbox_xyxy[1]])
                        print(f"   📦 Loaded bbox from SMPLest-X: {bbox}")
                        return bbox
            
            print("   ⚠️  Could not find bbox in SMPLest-X results")
            return None
            
        except Exception as e:
            print(f"   ⚠️  Error loading bbox: {e}")
            return None
    
    def render_enhanced_mesh(self, enhanced_mesh: np.ndarray, original_mesh: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Render both original and enhanced meshes onto the original input image"""
        if enhanced_mesh is None:
            print("⚠️  Cannot render enhanced mesh - mesh generation failed")
            return None, None, None
        
        print("🎨 Rendering meshes onto original input image...")
        
        try:
            # Load the original input image
            original_img = self.load_original_input_image()
            if original_img is None:
                print("   ⚠️  Could not load original image, using blank canvas")
                img_size = 512
                original_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # Load bbox from SMPLest-X results
            bbox = self.load_bbox_from_smplestx_results()
            if bbox is None:
                # Create default bbox if we can't find the original one
                h, w = original_img.shape[:2]
                bbox = np.array([w//4, h//4, w//2, h//2])  # [x, y, w, h]
            
            print(f"   📦 Using bbox: {bbox}")
            
            # Calculate camera parameters exactly like smplestx_adapter.py
            focal = [self.config.model.focal[0] / self.config.model.input_body_shape[1] * bbox[2], 
                    self.config.model.focal[1] / self.config.model.input_body_shape[0] * bbox[3]]
            princpt = [self.config.model.princpt[0] / self.config.model.input_body_shape[1] * bbox[2] + bbox[0], 
                      self.config.model.princpt[1] / self.config.model.input_body_shape[0] * bbox[3] + bbox[1]]
            
            camera_params = {'focal': focal, 'princpt': princpt}
            
            print(f"   📷 Camera parameters: focal={focal}, princpt={princpt}")
            
            # Create copies of the original image for rendering
            original_vis_img = original_img.copy()
            enhanced_vis_img = original_img.copy()
            
            # Draw bbox on both images (like in smplestx_adapter.py)
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
            
            original_vis_img = cv2.rectangle(original_vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            enhanced_vis_img = cv2.rectangle(enhanced_vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Render original mesh onto image
            original_vis_img = render_mesh(original_vis_img, original_mesh, self.smplx_model.face, 
                                         camera_params, mesh_as_vertices=False)
            
            # Render enhanced mesh onto image  
            enhanced_vis_img = render_mesh(enhanced_vis_img, enhanced_mesh, self.smplx_model.face, 
                                         camera_params, mesh_as_vertices=False)
            
            # Create side-by-side comparison
            comparison_img = self.create_mesh_comparison_image(original_vis_img, enhanced_vis_img)
            
            print("   ✅ Meshes rendered onto original image successfully")
            return original_vis_img, enhanced_vis_img, comparison_img
            
        except Exception as e:
            print(f"   ❌ Mesh rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def create_mesh_comparison_image(self, original_rendered: np.ndarray, 
                                   enhanced_rendered: np.ndarray) -> np.ndarray:
        """Create side-by-side comparison image"""
        if original_rendered is None or enhanced_rendered is None:
            return None
        
        print("🖼️  Creating mesh comparison image...")
        
        # Create side-by-side comparison
        h, w = original_rendered.shape[:2]
        comparison_img = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        comparison_img.fill(255)  # White background
        
        # Place images side by side
        comparison_img[:, :w] = original_rendered
        comparison_img[:, w+20:] = enhanced_rendered
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (0, 0, 0)  # Black text
        thickness = 2
        
        cv2.putText(comparison_img, 'Original (SMPLest-X)', (10, 30), font, font_scale, color, thickness)
        cv2.putText(comparison_img, 'Enhanced (Fused)', (w + 30, 30), font, font_scale, color, thickness)
        
        print("   ✅ Comparison image created")
        return comparison_img
    
    def load_model_parameters(self) -> Tuple[Dict, Dict, Dict]:
        """Load parameters from all three models with fixed EMOCA loading"""
        print("📥 Loading model parameters...")
        
        # Load SMPLest-X parameters
        smplx_data = None
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            with open(param_file, 'r') as f:
                smplx_data = json.load(f)
            break
        
        if smplx_data is None:
            raise FileNotFoundError("SMPLest-X parameters not found")
        
        # Load WiLoR parameters
        wilor_data = None
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            with open(param_file, 'r') as f:
                wilor_data = json.load(f)
            break
        
        if wilor_data is None:
            raise FileNotFoundError("WiLoR parameters not found")
        
        # Load EMOCA parameters - FIXED SEARCH PATTERNS
        emoca_data = None
        
        # Updated search patterns to match actual structure
        search_patterns = [
            'emoca_results/*/test*/codes.json',  # Match EMOCA_v2_lr_mse_20/test200/codes.json
            'emoca_results/*/*/codes.json',     # Broader pattern
            'emoca_results/EMOCA_*/test*/codes.json'  # Specific EMOCA pattern
        ]
        
        print("   🔍 Searching for EMOCA parameters...")
        for pattern in search_patterns:
            for param_file in self.results_dir.glob(pattern):
                print(f"   📁 Found EMOCA file: {param_file}")
                try:
                    with open(param_file, 'r') as f:
                        emoca_data = json.load(f)
                    print(f"   ✅ Successfully loaded EMOCA parameters from: {param_file}")
                    break
                except Exception as e:
                    print(f"   ⚠️  Failed to load {param_file}: {e}")
                    continue
            if emoca_data is not None:
                break
        
        if emoca_data is None:
            print("   ⚠️  EMOCA parameters not found, checking available files:")
            emoca_dir = self.results_dir / 'emoca_results'
            if emoca_dir.exists():
                for file in emoca_dir.rglob('*.json'):
                    print(f"     📄 Available: {file.relative_to(self.results_dir)}")
            emoca_data = {}
        
        print(f"   ✅ Loaded SMPLest-X: {len(smplx_data)} parameter types")
        print(f"   ✅ Loaded WiLoR: {len(wilor_data.get('hands', []))} hands")
        print(f"   ✅ Loaded EMOCA: {len(emoca_data)} parameter types")
        
        return smplx_data, wilor_data, emoca_data
    
    def transform_wilor_coordinates(self, wilor_data: Dict) -> Dict:
        """Apply coordinate transformation to WiLoR hand data"""
        print("🔄 Transforming WiLoR coordinates to SMPLest-X space...")
        
        scale = self.transformation_params['scale_factor']
        translation = self.transformation_params['translation_vector']
        
        transformed_hands = []
        
        for hand in wilor_data.get('hands', []):
            transformed_hand = hand.copy()
            
            # Transform 3D vertices
            if 'vertices_3d' in hand:
                vertices = np.array(hand['vertices_3d'])
                transformed_vertices = vertices * scale + translation
                transformed_hand['vertices_3d'] = transformed_vertices.tolist()
                
                print(f"   🖐️  Transformed {hand['hand_type']} hand vertices: scale={scale:.4f}")
                print(f"      📏 Original range: [{vertices.min():.4f}, {vertices.max():.4f}]")
                print(f"      📏 Transformed range: [{transformed_vertices.min():.4f}, {transformed_vertices.max():.4f}]")
            
            # Transform 3D keypoints
            if 'keypoints_3d' in hand:
                keypoints = np.array(hand['keypoints_3d'])
                transformed_keypoints = keypoints * scale + translation
                transformed_hand['keypoints_3d'] = transformed_keypoints.tolist()
            
            transformed_hands.append(transformed_hand)
        
        transformed_wilor = wilor_data.copy()
        transformed_wilor['hands'] = transformed_hands
        transformed_wilor['transformation_applied'] = {
            'scale_factor': scale,
            'translation_vector': translation.tolist(),
            'coordinate_system': 'SMPLest-X_aligned'
        }
        
        return transformed_wilor
    
    def map_emoca_expression(self, emoca_data: Dict, target_dim: int = 10) -> np.ndarray:
        """Map EMOCA 50D expression to SMPL-X 10D using intelligent mapping"""
        if not emoca_data or 'expcode' not in emoca_data:
            print("⚠️  No EMOCA expression data, using zero expression")
            return np.zeros(target_dim)
        
        print(f"🎭 Mapping EMOCA expression: 50D → {target_dim}D...")
        
        # Get EMOCA expression vector
        emoca_exp = np.array(emoca_data['expcode'])
        print(f"   📊 EMOCA expression range: [{emoca_exp.min():.4f}, {emoca_exp.max():.4f}]")
        
        # Use improved mapping strategy
        if len(emoca_exp) >= target_dim:
            # Take first target_dim components and normalize
            mapped_exp = emoca_exp[:target_dim]
            
            # Apply reasonable scaling for SMPL-X expression space
            # SMPL-X expressions typically range from -3 to +3
            mapped_exp = np.clip(mapped_exp * 0.3, -2.0, 2.0)
        else:
            # Pad with zeros if needed
            mapped_exp = np.pad(emoca_exp, (0, target_dim - len(emoca_exp)))
            mapped_exp = np.clip(mapped_exp * 0.3, -2.0, 2.0)
        
        print(f"   ✅ Expression mapped: range=[{mapped_exp.min():.4f}, {mapped_exp.max():.4f}]")
        print(f"   📈 Expression magnitude: {np.linalg.norm(mapped_exp):.4f}")
        
        return mapped_exp
    
    def extract_hand_pose_parameters(self, wilor_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract hand pose parameters from transformed WiLoR data"""
        print("🖐️  Extracting hand pose parameters...")
        
        left_hand_pose = np.zeros(45)   # 15 joints × 3 axis-angle
        right_hand_pose = np.zeros(45)
        
        for hand in wilor_data.get('hands', []):
            hand_type = hand.get('hand_type', '')
            
            # Check if MANO parameters are available
            if 'mano_parameters' in hand and 'parameters' in hand['mano_parameters']:
                mano_params = hand['mano_parameters']['parameters']
                
                # Extract hand pose (finger joint rotations)
                if 'hand_pose' in mano_params:
                    pose_data = mano_params['hand_pose']['values']
                    pose_array = np.array(pose_data).flatten()
                    
                    if hand_type == 'left' and len(pose_array) >= 45:
                        left_hand_pose = pose_array[:45]
                        print(f"   ✅ Left hand pose extracted: {len(pose_array)} → 45 params")
                    elif hand_type == 'right' and len(pose_array) >= 45:
                        right_hand_pose = pose_array[:45]
                        print(f"   ✅ Right hand pose extracted: {len(pose_array)} → 45 params")
                
            else:
                print(f"   ⚠️  No MANO parameters for {hand_type} hand, using zero pose")
        
        return left_hand_pose, right_hand_pose
    
    def create_fused_parameters(self, smplx_data: Dict, wilor_data: Dict, emoca_data: Dict) -> Dict:
        """Create fused parameter set replacing hand poses and expressions"""
        print("🔧 Creating fused parameters...")
        
        # Start with SMPLest-X as foundation
        fused_params = {
            # Keep body structure from SMPLest-X
            'betas': np.array(smplx_data['betas']),
            'body_pose': np.array(smplx_data['body_pose']),
            'root_pose': np.array(smplx_data['root_pose']),
            'translation': np.array(smplx_data['translation']),
            'jaw_pose': np.array(smplx_data['jaw_pose']),
            
            # Will be replaced with enhanced versions
            'left_hand_pose': np.array(smplx_data['left_hand_pose']),
            'right_hand_pose': np.array(smplx_data['right_hand_pose']),
            'expression': np.array(smplx_data['expression'])
        }
        
        print("   ✅ Foundation parameters from SMPLest-X")
        
        # Transform WiLoR coordinates
        transformed_wilor = self.transform_wilor_coordinates(wilor_data)
        
        # Replace hand poses with WiLoR parameters
        left_hand_pose, right_hand_pose = self.extract_hand_pose_parameters(transformed_wilor)
        fused_params['left_hand_pose'] = left_hand_pose
        fused_params['right_hand_pose'] = right_hand_pose
        
        print("   ✅ Hand poses replaced with WiLoR parameters")
        
        # Replace expression with EMOCA mapping
        mapped_expression = self.map_emoca_expression(emoca_data)
        fused_params['expression'] = mapped_expression
        
        print("   ✅ Expression replaced with EMOCA parameters")
        
        # Store transformation metadata
        fused_params['fusion_metadata'] = {
            'source_body': 'SMPLest-X',
            'source_hands': 'WiLoR_transformed',
            'source_expression': 'EMOCA_mapped',
            'transformation_applied': True,
            'coordinate_system': 'SMPLest-X_space'
        }
        
        return fused_params
    
    def generate_enhanced_mesh(self, fused_params: Dict) -> Optional[np.ndarray]:
        """Generate mesh using fused parameters"""
        if self.smplx_model is None:
            print("⚠️  SMPL-X model not available, cannot generate mesh")
            return None
        
        print("🎯 Generating enhanced mesh with fused parameters...")
        
        try:
            # Convert parameters to torch tensors
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare parameters for SMPL-X model (using neutral gender)
            smplx_layer = self.smplx_model.layer['neutral'].to(device)
            
            # Convert numpy arrays to torch tensors with proper shapes for SMPL-X
            torch_params = {
                'betas': torch.tensor(fused_params['betas']).float().unsqueeze(0).to(device),
                'body_pose': torch.tensor(fused_params['body_pose']).float().unsqueeze(0).to(device),
                'global_orient': torch.tensor(fused_params['root_pose']).float().unsqueeze(0).to(device),
                'left_hand_pose': torch.tensor(fused_params['left_hand_pose']).float().unsqueeze(0).to(device),
                'right_hand_pose': torch.tensor(fused_params['right_hand_pose']).float().unsqueeze(0).to(device),
                'expression': torch.tensor(fused_params['expression']).float().unsqueeze(0).to(device),
                'jaw_pose': torch.tensor(fused_params['jaw_pose']).float().unsqueeze(0).to(device),
                'transl': torch.tensor(fused_params['translation']).float().unsqueeze(0).to(device),
                # Add missing eye pose parameters (set to zero since we don't have eye tracking)
                'leye_pose': torch.zeros(1, 3).float().to(device),
                'reye_pose': torch.zeros(1, 3).float().to(device)
            }
            
            print(f"   🔧 Parameter shapes:")
            for name, tensor in torch_params.items():
                print(f"      {name}: {tensor.shape}")
            
            # Generate mesh using SMPL-X layer
            with torch.no_grad():
                output = smplx_layer(**torch_params)
                enhanced_mesh = output.vertices[0].detach().cpu().numpy()
            
            print(f"   ✅ Enhanced mesh generated: {enhanced_mesh.shape[0]} vertices")
            return enhanced_mesh
            
        except Exception as e:
            print(f"   ❌ Mesh generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compare_parameters(self, original_params: Dict, fused_params: Dict):
        """Compare original and fused parameters"""
        print("📊 Parameter comparison:")
        
        comparisons = [
            ('left_hand_pose', 'Left Hand'),
            ('right_hand_pose', 'Right Hand'), 
            ('expression', 'Expression')
        ]
        
        for param_name, display_name in comparisons:
            orig = original_params[param_name]
            fused = fused_params[param_name]
            
            orig_norm = np.linalg.norm(orig)
            fused_norm = np.linalg.norm(fused)
            difference = np.linalg.norm(orig - fused)
            
            print(f"   {display_name}:")
            print(f"     Original norm: {orig_norm:.4f}")
            print(f"     Fused norm: {fused_norm:.4f}")
            print(f"     Difference: {difference:.4f}")
            if orig_norm > 0:
                print(f"     Change: {((fused_norm - orig_norm) / orig_norm * 100):.1f}%")
            else:
                print(f"     Change: N/A (original was zero)")
    
    def save_results(self, original_params: Dict, fused_params: Dict, 
                    enhanced_mesh: Optional[np.ndarray],
                    original_rendered: Optional[np.ndarray],
                    enhanced_rendered: Optional[np.ndarray],
                    comparison_img: Optional[np.ndarray]):
        """Save fusion results including rendered images"""
        self.fusion_output_dir = self.results_dir / 'fusion_results'
        self.fusion_output_dir.mkdir(exist_ok=True)
        
        print(f"💾 Saving fusion results to: {self.fusion_output_dir}")
        
        # Save fused parameters
        serializable_fused = {}
        for key, value in fused_params.items():
            if isinstance(value, np.ndarray):
                serializable_fused[key] = value.tolist()
            else:
                serializable_fused[key] = value
        
        with open(self.fusion_output_dir / 'fused_parameters.json', 'w') as f:
            json.dump(serializable_fused, f, indent=2)
        
        # Save enhanced mesh
        if enhanced_mesh is not None:
            np.save(self.fusion_output_dir / 'enhanced_mesh.npy', enhanced_mesh)
            
            # Save mesh info
            with open(self.fusion_output_dir / 'enhanced_mesh_info.txt', 'w') as f:
                f.write("ENHANCED MESH INFORMATION\n")
                f.write("========================\n\n")
                f.write(f"Vertices: {enhanced_mesh.shape[0]}\n")
                f.write(f"Coordinates: {enhanced_mesh.shape[1]}D\n")
                f.write(f"Mesh bounds:\n")
                f.write(f"  X: [{enhanced_mesh[:, 0].min():.4f}, {enhanced_mesh[:, 0].max():.4f}]\n")
                f.write(f"  Y: [{enhanced_mesh[:, 1].min():.4f}, {enhanced_mesh[:, 1].max():.4f}]\n")
                f.write(f"  Z: [{enhanced_mesh[:, 2].min():.4f}, {enhanced_mesh[:, 2].max():.4f}]\n")
                f.write(f"Centroid: [{enhanced_mesh[:, 0].mean():.4f}, {enhanced_mesh[:, 1].mean():.4f}, {enhanced_mesh[:, 2].mean():.4f}]\n")
        
        # Save rendered images
        if original_rendered is not None:
            cv2.imwrite(str(self.fusion_output_dir / 'original_mesh_rendered.jpg'), 
                       original_rendered[:, :, ::-1])  # Convert RGB to BGR for OpenCV
            print("   ✅ Original mesh rendering saved")
        
        if enhanced_rendered is not None:
            cv2.imwrite(str(self.fusion_output_dir / 'enhanced_mesh_rendered.jpg'), 
                       enhanced_rendered[:, :, ::-1])  # Convert RGB to BGR for OpenCV
            print("   ✅ Enhanced mesh rendering saved")
        
        if comparison_img is not None:
            cv2.imwrite(str(self.fusion_output_dir / 'original_vs_enhanced_comparison.jpg'), 
                       comparison_img[:, :, ::-1])  # Convert RGB to BGR for OpenCV
            print("   ✅ Mesh comparison image saved")
        
        # Save comparison report
        with open(self.fusion_output_dir / 'parameter_comparison.txt', 'w') as f:
            f.write("PARAMETER FUSION COMPARISON REPORT\n")
            f.write("==================================\n\n")
            
            for param_name in ['left_hand_pose', 'right_hand_pose', 'expression']:
                orig = np.array(original_params[param_name])
                fused = np.array(fused_params[param_name])
                
                f.write(f"{param_name.replace('_', ' ').title()}:\n")
                f.write(f"  Original norm: {np.linalg.norm(orig):.6f}\n")
                f.write(f"  Fused norm: {np.linalg.norm(fused):.6f}\n")
                f.write(f"  Difference norm: {np.linalg.norm(orig - fused):.6f}\n")
                f.write(f"  Max absolute change: {np.abs(orig - fused).max():.6f}\n")
                f.write(f"  Mean absolute change: {np.abs(orig - fused).mean():.6f}\n\n")
        
        print("   ✅ Fused parameters saved")
        print("   ✅ Comparison report saved")
        
        # Print summary of saved files
        print(f"\n📁 Saved files in {self.fusion_output_dir}:")
        for file in self.fusion_output_dir.iterdir():
            if file.is_file():
                print(f"   📄 {file.name}")
        
    def setup_smplx_model(self):
        """Initialize SMPL-X model for mesh generation"""
        print("🤖 Setting up SMPL-X model...")
        
        # Try to find SMPL-X model path
        possible_paths = [
            'human_models/human_model_files/',
            './human_models/human_model_files/',
            '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/pretrained_models/smplx',
            './pretrained_models/smplx',
            '../pretrained_models/smplx'
        ]
        
        smplx_path = None
        for path in possible_paths:
            if os.path.exists(path):
                smplx_path = path
                break
        
        if smplx_path is None:
            print("⚠️  SMPL-X model path not found. Will use parameter-only fusion.")
            self.smplx_model = None
        else:
            try:
                self.smplx_model = SMPLX(smplx_path)
                print(f"   ✅ SMPL-X model loaded from: {smplx_path}")
            except Exception as e:
                print(f"   ⚠️  Could not load SMPL-X model: {e}")
                self.smplx_model = None
    
    def run_enhanced_fusion(self):
        """Execute the complete enhanced fusion process with rendering"""
        print("\n" + "="*60)
        print("🚀 ENHANCED DIRECT PARAMETER FUSION SYSTEM")
        print("="*60 + "\n")
        
        # Load all model parameters
        smplx_data, wilor_data, emoca_data = self.load_model_parameters()
        
        # Create fused parameters
        fused_params = self.create_fused_parameters(smplx_data, wilor_data, emoca_data)
        
        # Compare parameters
        self.compare_parameters(smplx_data, fused_params)
        
        # Generate enhanced mesh
        enhanced_mesh = self.generate_enhanced_mesh(fused_params)
        
        # Get original mesh
        original_mesh = np.array(smplx_data['mesh']) if 'mesh' in smplx_data else None
        
        # Render meshes
        original_rendered, enhanced_rendered, comparison_img = None, None, None
        if enhanced_mesh is not None and original_mesh is not None:
            original_rendered, enhanced_rendered, comparison_img = self.render_enhanced_mesh(
                enhanced_mesh, original_mesh)
        else:
            print("⚠️  Skipping mesh rendering - missing mesh data")
        
        # Save results
        self.save_results(smplx_data, fused_params, enhanced_mesh,
                         original_rendered, enhanced_rendered, comparison_img)
        
        print("\n" + "="*60)
        print("✅ ENHANCED FUSION COMPLETE!")
        print("="*60)
        print("\n🎯 Results:")
        print("   - Fused parameters with WiLoR hands + EMOCA expression")
        print("   - Coordinate transformation applied")
        print("   - Enhanced mesh generated and rendered")
        print("   - Comparison visualizations created")
        print("   - Parameter analysis completed")
        
        if self.fusion_output_dir:
            print(f"\n📁 Output directory: {self.fusion_output_dir}")
            print("\n📸 Rendered outputs:")
            if original_rendered is not None:
                print("   - original_mesh_rendered.jpg")
            if enhanced_rendered is not None:
                print("   - enhanced_mesh_rendered.jpg")  
            if comparison_img is not None:
                print("   - original_vs_enhanced_comparison.jpg")
        
        return fused_params, enhanced_mesh, original_rendered, enhanced_rendered

def main():
    parser = argparse.ArgumentParser(description='Enhanced Direct Parameter Fusion System')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing pipeline results and coordinate analysis')
    
    args = parser.parse_args()
    
    # Validate input directory
    results_path = Path(args.results_dir)
    if not results_path.exists():
        print(f"❌ Error: Results directory not found: {results_path}")
        return
    
    # Check for required analysis files
    coord_file = results_path / 'coordinate_analysis_summary.json'
    if not coord_file.exists():
        print(f"❌ Error: Coordinate analysis not found. Run coordinate analyzer first.")
        print(f"   Expected: {coord_file}")
        return
    
    # Run enhanced fusion
    fusion_system = EnhancedDirectParameterFusion(args.results_dir)
    fused_params, enhanced_mesh, original_rendered, enhanced_rendered = fusion_system.run_enhanced_fusion()

if __name__ == '__main__':
    main()