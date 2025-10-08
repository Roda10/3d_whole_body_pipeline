import numpy as np
import json
from typing import Dict, List, Any, Optional
import torch

class EMOCAParameterExtractor:
    """Extract EMOCA FLAME parameters and 3D geometry to JSON"""
    
    def __init__(self):
        pass
        
    def extract_parameters(self, vals: Dict, img_path: str, bbox: Optional[List] = None) -> Dict[str, Any]:
        """Extract FLAME codes and 3D geometry from EMOCA output"""
        
        results = {
            "metadata": {
                "image_path": str(img_path),
                "detection_count": 1
            },
            "faces": []
        }
        
        # Add bbox if available
        if bbox is not None:
            results["metadata"]["bbox"] = bbox
        
        # Extract single face data
        face_data = self._extract_single_face(vals, 0)
        results["faces"].append(face_data)
        
        return results
    
    def _extract_single_face(self, vals: Dict, idx: int) -> Dict[str, Any]:
        """Extract parameters for a single face"""
        
        face_data = {
            "face_id": idx,
            
            # FLAME parametric codes
            "flame_codes": {
                "shape": vals["shapecode"][idx].detach().cpu().numpy().tolist(),
                "expression": vals["expcode"][idx].detach().cpu().numpy().tolist(),
                "texture": vals["texcode"][idx].detach().cpu().numpy().tolist(),
                "pose": vals["posecode"][idx].detach().cpu().numpy().tolist(),
                "detail": vals["detailcode"][idx].detach().cpu().numpy().tolist(),
                "light": vals["lightcode"][idx].detach().cpu().numpy().tolist()
            },
            
            # Camera parameters
            "camera": vals["cam"][idx].detach().cpu().numpy().tolist(),
            
            # 3D Geometry
            "vertices_3d": vals["verts"][idx].detach().cpu().numpy().tolist(),
            "trans_vertices_3d": vals["trans_verts"][idx].detach().cpu().numpy().tolist(),
            "landmarks_2d": vals["landmarks2d"][idx].detach().cpu().numpy().tolist(),
            "landmarks_3d": vals["landmarks3d"][idx].detach().cpu().numpy().tolist(),
            
            # Shape information for reference
            "shapes": {
                "shape": [100],
                "expression": [50],
                "texture": [50],
                "pose": [6],
                "detail": [128],
                "light": [9, 3],
                "camera": [3],
                "vertices_3d": [5023, 3],
                "trans_vertices_3d": [5023, 3],
                "landmarks_2d": [68, 3],
                "landmarks_3d": [68, 3]
            }
        }
        
        return face_data

def save_emoca_parameters_json(extractor: EMOCAParameterExtractor,
                               vals: Dict, img_path: str,
                               output_path: str,
                               bbox: Optional[List] = None) -> None:
    """Save extracted EMOCA parameters to JSON"""
    
    parameters = extractor.extract_parameters(vals, img_path, bbox)
    
    # Add extraction metadata
    parameters["extraction_info"] = {
        "extractor_version": "v1.0",
        "model": "EMOCA",
        "includes_flame_codes": True,
        "includes_3d_geometry": True,
        "coordinate_system": "EMOCA_face_centric"
    }
    
    with open(output_path, 'w') as f:
        json.dump(parameters, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved EMOCA parameters to: {output_path}")
    
    # Print summary
    face_count = len(parameters["faces"])
    print(f"   📊 Extracted {face_count} face(s)")
    
    face = parameters["faces"][0]
    vertices_count = len(face["vertices_3d"])
    landmarks_count = len(face["landmarks_3d"])
    print(f"   👤 Face: {vertices_count} vertices, {landmarks_count} landmarks")
    print(f"   📦 FLAME codes: shape(100), exp(50), tex(50), pose(6), detail(128), light(9x3)")