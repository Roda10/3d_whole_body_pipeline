# import numpy as np
# import json
# from typing import Dict, List, Any, Optional
# import torch

# class WiLoRParameterExtractor:
#     """Simple extractor for MANO parameters and basic summary statistics"""
    
#     def __init__(self, model_cfg):
#         self.model_cfg = model_cfg
        
#     def extract_parameters(self, batch: Dict, out: Dict, pred_cam_t_full: np.ndarray, 
#                           img_path: str, scaled_focal_length: float) -> Dict[str, Any]:
#         """Extract MANO parameters with shape information"""
        
#         batch_size = batch['img'].shape[0]
#         results = {
#             "metadata": {
#                 "image_path": str(img_path),
#                 "batch_size": batch_size,
#                 "detection_count": batch_size
#             },
#             "hands": []
#         }
        
#         for n in range(batch_size):
#             hand_data = self._extract_single_hand(batch, out, pred_cam_t_full, n)
#             results["hands"].append(hand_data)
            
#         return results
    
#     def _extract_single_hand(self, batch: Dict, out: Dict, pred_cam_t_full: np.ndarray, idx: int) -> Dict[str, Any]:
#         """Extract parameters for a single hand detection"""
        
#         # Basic hand info
#         is_right = bool(batch['right'][idx].cpu().numpy())
        
#         # 3D vertices and joints shapes
#         verts_shape = list(out['pred_vertices'][idx].shape)
#         joints_shape = list(out['pred_keypoints_3d'][idx].shape)
        
#         # Camera parameters
#         pred_cam = out['pred_cam'][idx].detach().cpu().numpy()
#         cam_t = pred_cam_t_full[idx]
        
#         hand_data = {
#             "hand_id": idx,
#             "hand_type": "right" if is_right else "left",
#             "shapes": {
#                 "vertices_3d": verts_shape,
#                 "keypoints_3d": joints_shape,
#                 "camera_prediction": list(pred_cam.shape),
#                 "camera_translation": list(cam_t.shape)
#             }
#         }
        
#         # MANO parameters (if available in output)
#         if 'pred_mano_params' in out:
#             mano_raw = out['pred_mano_params'][idx].detach().cpu().numpy()
#             hand_data["mano_parameters"] = {
#                 "total_shape": list(mano_raw.shape),
#                 "shape_coefficients": {
#                     "values": mano_raw[:10].tolist() if len(mano_raw) >= 10 else mano_raw.tolist(),
#                     "shape": [min(10, len(mano_raw))]
#                 },
#                 "pose_coefficients": {
#                     "values": mano_raw[10:].tolist() if len(mano_raw) > 10 else [],
#                     "shape": [max(0, len(mano_raw) - 10)]
#                 }
#             }
#         elif 'pred_shape' in out and 'pred_pose' in out:
#             # Separate shape and pose parameters
#             shape_params = out['pred_shape'][idx].detach().cpu().numpy()
#             pose_params = out['pred_pose'][idx].detach().cpu().numpy()
#             hand_data["mano_parameters"] = {
#                 "shape_coefficients": {
#                     "values": shape_params.tolist(),
#                     "shape": list(shape_params.shape)
#                 },
#                 "pose_coefficients": {
#                     "values": pose_params.tolist(),
#                     "shape": list(pose_params.shape)
#                 }
#             }
#         else:
#             hand_data["mano_parameters"] = {
#                 "note": "MANO parameters not found in model output"
#             }
            
#         return hand_data

# def save_wilor_parameters_json(extractor: WiLoRParameterExtractor, 
#                               batch: Dict, out: Dict, pred_cam_t_full: np.ndarray,
#                               img_path: str, scaled_focal_length: float,
#                               output_path: str) -> None:
#     """Save extracted parameters to a simple JSON file"""
    
#     parameters = extractor.extract_parameters(batch, out, pred_cam_t_full, 
#                                             img_path, scaled_focal_length)
    
#     with open(output_path, 'w') as f:
#         json.dump(parameters, f, indent=2, ensure_ascii=False)


import numpy as np
import json
from typing import Dict, List, Any, Optional
import torch

class WiLoRParameterExtractor:
    """Fixed extractor that saves actual MANO parameters and 3D coordinates"""
    
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        
    def extract_parameters(self, batch: Dict, out: Dict, pred_cam_t_full: np.ndarray, 
                          img_path: str, scaled_focal_length: float) -> Dict[str, Any]:
        """Extract MANO parameters with actual coordinate data"""
        
        batch_size = batch['img'].shape[0]
        results = {
            "metadata": {
                "image_path": str(img_path),
                "batch_size": batch_size,
                "detection_count": batch_size,
                "scaled_focal_length": float(scaled_focal_length)
            },
            "hands": []
        }
        
        for n in range(batch_size):
            hand_data = self._extract_single_hand(batch, out, pred_cam_t_full, n, scaled_focal_length)
            results["hands"].append(hand_data)
            
        return results
    
    def _extract_single_hand(self, batch: Dict, out: Dict, pred_cam_t_full: np.ndarray, 
                           idx: int, scaled_focal_length: float) -> Dict[str, Any]:
        """Extract parameters for a single hand detection with actual 3D data"""
        
        # Basic hand info
        is_right = bool(batch['right'][idx].cpu().numpy())
        
        # Extract actual 3D coordinates (this is the key fix!)
        vertices_3d = out['pred_vertices'][idx].detach().cpu().numpy()
        keypoints_3d = out['pred_keypoints_3d'][idx].detach().cpu().numpy()
        
        # Apply hand orientation correction (from original adapter)
        vertices_3d[:, 0] = (2 * is_right - 1) * vertices_3d[:, 0]
        keypoints_3d[:, 0] = (2 * is_right - 1) * keypoints_3d[:, 0]
        
        # Camera parameters
        pred_cam = out['pred_cam'][idx].detach().cpu().numpy()
        cam_t = pred_cam_t_full[idx]
        
        # Box info for coordinate transformation
        box_center = batch["box_center"][idx].float().cpu().numpy()
        box_size = batch["box_size"][idx].float().cpu().numpy()
        img_size = batch["img_size"][idx].float().cpu().numpy()
        
        hand_data = {
            "hand_id": idx,
            "hand_type": "right" if is_right else "left",
            "is_right": is_right,
            
            # ACTUAL 3D COORDINATE DATA (the key addition!)
            "vertices_3d": vertices_3d.tolist(),
            "keypoints_3d": keypoints_3d.tolist(),
            
            # Camera and transformation data
            "camera_prediction": pred_cam.tolist(),
            "camera_translation": cam_t.tolist(),
            "box_center": box_center.tolist(),
            "box_size": box_size.tolist(), 
            "img_size": img_size.tolist(),
            
            # Shape information for reference
            "shapes": {
                "vertices_3d": list(vertices_3d.shape),
                "keypoints_3d": list(keypoints_3d.shape),
                "camera_prediction": list(pred_cam.shape),
                "camera_translation": list(cam_t.shape)
            }
        }
        
        # Extract MANO parameters if available
        hand_data["mano_parameters"] = self._extract_mano_parameters(out, idx)
        
        # Add coordinate statistics for debugging
        hand_data["coordinate_stats"] = {
            "vertices_range": {
                "x": [float(vertices_3d[:, 0].min()), float(vertices_3d[:, 0].max())],
                "y": [float(vertices_3d[:, 1].min()), float(vertices_3d[:, 1].max())],
                "z": [float(vertices_3d[:, 2].min()), float(vertices_3d[:, 2].max())]
            },
            "vertices_center": [
                float(vertices_3d[:, 0].mean()),
                float(vertices_3d[:, 1].mean()),
                float(vertices_3d[:, 2].mean())
            ],
            "keypoints_range": {
                "x": [float(keypoints_3d[:, 0].min()), float(keypoints_3d[:, 0].max())],
                "y": [float(keypoints_3d[:, 1].min()), float(keypoints_3d[:, 1].max())],
                "z": [float(keypoints_3d[:, 2].min()), float(keypoints_3d[:, 2].max())]
            }
        }
            
        return hand_data
    
    def _extract_mano_parameters(self, out: Dict, idx: int) -> Dict[str, Any]:
        """Extract MANO parameters from model output"""
        
        # Try different ways to extract MANO parameters from WiLoR output
        if 'pred_mano_params' in out:
            mano_raw = out['pred_mano_params'][idx].detach().cpu().numpy()
            return {
                "source": "pred_mano_params",
                "total_params": mano_raw.tolist(),
                "total_shape": list(mano_raw.shape),
                "shape_coefficients": {
                    "values": mano_raw[:10].tolist() if len(mano_raw) >= 10 else mano_raw.tolist(),
                    "dimensions": min(10, len(mano_raw))
                },
                "pose_coefficients": {
                    "values": mano_raw[10:].tolist() if len(mano_raw) > 10 else [],
                    "dimensions": max(0, len(mano_raw) - 10)
                }
            }
        elif 'pred_shape' in out and 'pred_pose' in out:
            # Separate shape and pose parameters
            shape_params = out['pred_shape'][idx].detach().cpu().numpy()
            pose_params = out['pred_pose'][idx].detach().cpu().numpy()
            return {
                "source": "separate_shape_pose",
                "shape_coefficients": {
                    "values": shape_params.tolist(),
                    "dimensions": list(shape_params.shape)
                },
                "pose_coefficients": {
                    "values": pose_params.tolist(),
                    "dimensions": list(pose_params.shape)
                }
            }
        elif 'pred_hand_pose' in out:
            # Hand pose only
            hand_pose = out['pred_hand_pose'][idx].detach().cpu().numpy()
            return {
                "source": "pred_hand_pose",
                "pose_coefficients": {
                    "values": hand_pose.tolist(),
                    "dimensions": list(hand_pose.shape)
                },
                "shape_coefficients": {
                    "note": "Shape parameters not found"
                }
            }
        else:
            # Check what keys are actually available
            available_keys = [k for k in out.keys() if 'pred' in k.lower()]
            return {
                "note": "MANO parameters not found in model output",
                "available_output_keys": available_keys,
                "suggestion": "Check WiLoR model output keys for parameter extraction"
            }

def save_wilor_parameters_json(extractor: WiLoRParameterExtractor, 
                              batch: Dict, out: Dict, pred_cam_t_full: np.ndarray,
                              img_path: str, scaled_focal_length: float,
                              output_path: str) -> None:
    """Save extracted parameters with full 3D coordinate data to JSON"""
    
    parameters = extractor.extract_parameters(batch, out, pred_cam_t_full, 
                                            img_path, scaled_focal_length)
    
    # Add some metadata about the extraction
    parameters["extraction_info"] = {
        "extractor_version": "fixed_v1.0",
        "includes_3d_coordinates": True,
        "coordinate_system": "WiLoR_hand_centric",
        "notes": "Fixed version that saves actual 3D coordinates instead of just shapes"
    }
    
    with open(output_path, 'w') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved WiLoR parameters with 3D coordinates to: {output_path}")
    
    # Print summary for verification
    hand_count = len(parameters["hands"])
    print(f"   📊 Extracted {hand_count} hands")
    for i, hand in enumerate(parameters["hands"]):
        vertices_count = len(hand["vertices_3d"])
        keypoints_count = len(hand["keypoints_3d"])
        hand_type = hand["hand_type"]
        print(f"   🖐️  Hand {i+1} ({hand_type}): {vertices_count} vertices, {keypoints_count} keypoints")
        
        # Show coordinate ranges for quick validation
        stats = hand["coordinate_stats"]
        v_range = stats["vertices_range"]
        print(f"      Vertex range: X[{v_range['x'][0]:.3f}, {v_range['x'][1]:.3f}], "
              f"Y[{v_range['y'][0]:.3f}, {v_range['y'][1]:.3f}], "
              f"Z[{v_range['z'][0]:.3f}, {v_range['z'][1]:.3f}]")