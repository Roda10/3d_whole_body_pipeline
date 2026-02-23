"""
Camera parameter validation for EHF dataset
"""

import numpy as np
from pathlib import Path
import json
import cv2
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt

def load_camera_parameters(ehf_path: Path) -> Optional[Dict]:
    """
    Load camera parameters from EHF dataset format
    
    Args:
        ehf_path: Path to EHF dataset root directory
        
    Returns:
        Dict with camera parameters or None if not found
    """
    # EHF dataset uses a single camera parameter file
    cam_txt = ehf_path / "EHF_camera.txt"
    
    if cam_txt.exists():
        # Try text format as fallback
        with open(cam_txt, 'r') as f:
            lines = f.readlines()
            params = {}
            try:
                for i, line in enumerate(lines):
                    if "Focal length" in line:
                        params['focal_length'] = [
                            float(lines[i+1].strip().strip('[]')),
                            float(lines[i+2].strip().strip('[]'))
                        ]
                    elif "Principal Point" in line:
                        params['principal_point'] = [
                            float(lines[i+1].strip().strip('[]')),
                            float(lines[i+2].strip().strip('[]'))
                        ]
                    elif "Translation" in line:
                        params['translation'] = [
                            float(lines[i+1+j].strip().strip('[]'))
                            for j in range(3)
                        ]
                    elif "Rotation" in line:
                        params['rotation'] = [
                            float(lines[i+1+j].strip().strip('[]'))
                            for j in range(3)
                        ]
                
                if all(k in params for k in ['focal_length', 'principal_point', 'translation', 'rotation']):
                    return {
                        'format': 'txt',
                        'intrinsics': {
                            'focal_length': params['focal_length'],
                            'principal_point': params['principal_point'],
                            'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]  # Assume no distortion
                        },
                        'extrinsics': {
                            'rotation': params['rotation'],
                            'translation': params['translation']
                        }
                    }
            except Exception as e:
                print(f"Failed to parse camera text file: {cam_txt}\nError: {str(e)}")
    
    return None

def validate_camera_matrix(focal_length: list, principal_point: list, image_size: Tuple[int, int] = (1920, 1080)) -> Dict:
    """Validate camera intrinsic parameters against common constraints"""
    validation = {
        'is_valid': True,
        'warnings': [],
        'focal_length': {
            'min': min(focal_length),
            'max': max(focal_length),
            'ratio': focal_length[0] / focal_length[1]
        },
        'principal_point': {
            'x_offset': abs(principal_point[0] - image_size[0]/2) / image_size[0],
            'y_offset': abs(principal_point[1] - image_size[1]/2) / image_size[1]
        }
    }
    
    # Check focal length constraints
    if any(f <= 0 for f in focal_length):
        validation['is_valid'] = False
        validation['warnings'].append("Focal length must be positive")
    
    if not 0.9 < validation['focal_length']['ratio'] < 1.1:
        validation['warnings'].append("Unusual focal length ratio (should be close to 1.0)")
    
    # Check principal point constraints
    if validation['principal_point']['x_offset'] > 0.1:
        validation['warnings'].append("Principal point X significantly off-center (>10%)")
    if validation['principal_point']['y_offset'] > 0.1:
        validation['warnings'].append("Principal point Y significantly off-center (>10%)")
        
    return validation

def validate_camera_pose(rotation: np.ndarray, translation: np.ndarray) -> Dict:
    """Validate camera extrinsic parameters"""
    validation = {
        'is_valid': True,
        'warnings': [],
        'rotation': {
            'magnitude_deg': float(np.linalg.norm(rotation) * 180 / np.pi),
            'components_deg': [float(r * 180 / np.pi) for r in rotation]
        },
        'translation': {
            'magnitude_m': float(np.linalg.norm(translation)),
            'components_m': [float(t) for t in translation]
        }
    }
    
    # Check rotation constraints
    if validation['rotation']['magnitude_deg'] > 180:
        validation['warnings'].append("Rotation magnitude exceeds 180 degrees")
    
    # Check translation constraints (assuming meters)
    if validation['translation']['magnitude_m'] > 10:
        validation['warnings'].append("Translation magnitude exceeds 10 meters")
    if validation['translation']['magnitude_m'] < 0.1:
        validation['warnings'].append("Translation magnitude less than 0.1 meters")
    
    return validation

def validate_camera_parameters(ehf_path: str, frame_id: str) -> Dict:
    """
    Validate camera parameters for a frame
    
    Args:
        ehf_path: Path to EHF dataset
        frame_id: Frame ID to analyze
        
    Returns:
        Dict containing validation results
    """
    # Load camera parameters from EHF dataset
    camera_params = load_camera_parameters(Path(ehf_path))
    if camera_params is None:
        return {
            'is_valid': False,
            'error': "Failed to load camera parameters"
        }
    
    # Extract parameters
    focal_length = camera_params['intrinsics']['focal_length']
    principal_point = camera_params['intrinsics']['principal_point']
    rotation = np.array(camera_params['extrinsics']['rotation'])
    translation = np.array(camera_params['extrinsics']['translation'])
    
    # Validate each component
    validation = {
        'camera_format': camera_params['format'],
        'intrinsics': validate_camera_matrix(focal_length, principal_point),
        'extrinsics': validate_camera_pose(rotation, translation),
        'raw_parameters': camera_params
    }
    
    # Determine overall validity
    validation['is_valid'] = (
        validation['intrinsics']['is_valid'] and 
        not validation['intrinsics']['warnings'] and
        not validation['extrinsics']['warnings']
    )
    
    return validation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate camera parameters")
    parser.add_argument("--ehf_path", type=str, default="data/EHF", help="Path to EHF dataset")
    parser.add_argument("--frame_id", type=str, required=True, help="Frame ID to analyze")
    
    args = parser.parse_args()
    
    results = validate_camera_parameters(args.ehf_path, args.frame_id)
    
    print("\n📊 Camera Parameter Validation Results:")
    print(f"\nFormat: {results.get('camera_format', 'unknown')}")
    
    if not results.get('is_valid', False):
        print("\n❌ Invalid camera parameters!")
        if 'error' in results:
            print(f"Error: {results['error']}")
    else:
        print("\n✅ Camera parameters are valid!")
        
    if 'intrinsics' in results:
        intr = results['intrinsics']
        print("\nIntrinsic Parameters:")
        print(f"Focal Length Ratio: {intr['focal_length']['ratio']:.3f}")
        print(f"Principal Point Offset: ({intr['principal_point']['x_offset']*100:.1f}%, {intr['principal_point']['y_offset']*100:.1f}%)")
        if intr['warnings']:
            print("\nIntrinsic Warnings:")
            for w in intr['warnings']:
                print(f"- {w}")
                
    if 'extrinsics' in results:
        extr = results['extrinsics']
        print("\nExtrinsic Parameters:")
        print(f"Rotation Magnitude: {extr['rotation']['magnitude_deg']:.1f}°")
        print(f"Translation Magnitude: {extr['translation']['magnitude_m']:.2f}m")
        if extr['warnings']:
            print("\nExtrinsic Warnings:")
            for w in extr['warnings']:
                print(f"- {w}")
                
    if 'raw_parameters' in results:
        print("\nRaw Parameters:")
        params = results['raw_parameters']
        print(f"Focal Length: {params['intrinsics']['focal_length']}")
        print(f"Principal Point: {params['intrinsics']['principal_point']}")
        print(f"Rotation: {params['extrinsics']['rotation']}")
        print(f"Translation: {params['extrinsics']['translation']}")