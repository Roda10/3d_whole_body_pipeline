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
        """Extract MANO parameters with shape information"""
        
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
        """Extract parameters for a single hand detection"""
        
        # Basic hand info
        is_right = bool(batch['right'][idx].cpu().numpy())
        
        # Extract actual 3D coordinates
        vertices_3d = out['pred_vertices'][idx].detach().cpu().numpy()
        keypoints_3d = out['pred_keypoints_3d'][idx].detach().cpu().numpy()
        
        # Apply hand orientation correction
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
            
            # 3D coordinate data
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
        
        # Check if pred_mano_params exists in output
        if 'pred_mano_params' in out and out['pred_mano_params']:
            mano_params = out['pred_mano_params']
            extracted_params = {
                "source": "pred_mano_params",
                "parameters": {}
            }
            
            # Extract each parameter type
            if 'global_orient' in mano_params:
                global_orient = mano_params['global_orient'][idx].detach().cpu().numpy()
                extracted_params['parameters']['global_orient'] = {
                    "values": global_orient.tolist(),
                    "shape": list(global_orient.shape),
                    "type": "rotation_matrix"
                }
            
            if 'hand_pose' in mano_params:
                hand_pose = mano_params['hand_pose'][idx].detach().cpu().numpy()
                extracted_params['parameters']['hand_pose'] = {
                    "values": hand_pose.tolist(),
                    "shape": list(hand_pose.shape),
                    "type": "rotation_matrix"
                }
            
            if 'betas' in mano_params:
                betas = mano_params['betas'][idx].detach().cpu().numpy()
                extracted_params['parameters']['betas'] = {
                    "values": betas.tolist(),
                    "shape": list(betas.shape),
                    "type": "shape_parameters"
                }
            
            # Add any other parameters that might be present
            for key in mano_params.keys():
                if key not in ['global_orient', 'hand_pose', 'betas']:
                    param = mano_params[key][idx].detach().cpu().numpy()
                    extracted_params['parameters'][key] = {
                        "values": param.tolist(),
                        "shape": list(param.shape),
                        "type": "unknown"
                    }
            
            return extracted_params
        
        else:
            # If no MANO parameters found, return a note
            return {
                "note": "MANO parameters not found in model output",
                "available_output_keys": list(out.keys()),
                "suggestion": "Check WiLoR model configuration"
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
        "extractor_version": "fixed_v2.0",
        "includes_3d_coordinates": True,
        "includes_mano_parameters": True,
        "coordinate_system": "WiLoR_hand_centric",
        "notes": "Fixed version that extracts MANO parameters from pred_mano_params"
    }
    
    with open(output_path, 'w') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved WiLoR parameters to: {output_path}")
    
    # Print summary for verification
    hand_count = len(parameters["hands"])
    print(f"   📊 Extracted {hand_count} hands")
    for i, hand in enumerate(parameters["hands"]):
        vertices_count = len(hand["vertices_3d"])
        keypoints_count = len(hand["keypoints_3d"])
        hand_type = hand["hand_type"]
        print(f"   🖐️  Hand {i+1} ({hand_type}): {vertices_count} vertices, {keypoints_count} keypoints")
        
        # Show if MANO parameters were extracted
        if "mano_parameters" in hand:
            mano_info = hand["mano_parameters"]
            if "parameters" in mano_info:
                param_names = list(mano_info["parameters"].keys())
                print(f"      📦 MANO parameters: {', '.join(param_names)}")
            else:
                print(f"      ⚠️  {mano_info.get('note', 'No MANO parameters')}")