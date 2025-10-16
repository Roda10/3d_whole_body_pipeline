"""
Combined camera and keypoint validation for EHF dataset
"""

import numpy as np
from pathlib import Path
import json
import cv2
from typing import Dict, Tuple, Optional, List

def load_camera_parameters(ehf_path: Path) -> Optional[Dict]:
    """Load camera parameters from EHF dataset"""
    cam_txt = ehf_path / "EHF_camera.txt"
    if cam_txt.exists():
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
                            'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
                        },
                        'extrinsics': {
                            'rotation': params['rotation'],
                            'translation': params['translation']
                        }
                    }
            except Exception as e:
                print(f"Failed to parse camera text file: {cam_txt}\nError: {str(e)}")
    
    return None

def load_keypoints(ehf_path: Path, frame_id: str) -> Optional[Dict]:
    """Load OpenPose keypoints from frame data"""
    joints_2d_path = ehf_path / f"{frame_id}_2Djnt.json"
    if not joints_2d_path.exists():
        return None
        
    with open(joints_2d_path, 'r') as f:
        joints_data = json.load(f)
        if 'people' not in joints_data or not joints_data['people']:
            return None
            
        person = joints_data['people'][0]  # Take first person
        keypoint_types = {
            'body': ('pose_keypoints_2d', 25),
            'face': ('face_keypoints_2d', 70),
            'hand_left': ('hand_left_keypoints_2d', 21),
            'hand_right': ('hand_right_keypoints_2d', 21)
        }
        
        keypoints = {}
        for part_name, (key, _) in keypoint_types.items():
            if key in person:
                points = np.array(person[key]).reshape(-1, 3)
                keypoints[part_name] = {
                    'points': points[:, :2],  # xy coordinates
                    'confidence': points[:, 2]  # confidence scores
                }
        
        return keypoints

def project_points(points_3d: np.ndarray, camera_params: Dict) -> np.ndarray:
    """Project 3D points to 2D using camera parameters"""
    focal_length = np.array(camera_params['intrinsics']['focal_length'])
    principal_point = np.array(camera_params['intrinsics']['principal_point'])
    rotation = np.array(camera_params['extrinsics']['rotation'])
    translation = np.array(camera_params['extrinsics']['translation'])
    
    # Convert rotation vector to matrix
    R = cv2.Rodrigues(rotation)[0]
    
    # Project points
    points_camera = (R @ points_3d.T + translation.reshape(3,1)).T
    points_2d = np.zeros((len(points_camera), 2))
    points_2d[:,0] = focal_length[0] * points_camera[:,0] / points_camera[:,2] + principal_point[0]
    points_2d[:,1] = focal_length[1] * points_camera[:,1] / points_camera[:,2] + principal_point[1]
    
    return points_2d

def analyze_camera_coverage(keypoints: Dict, image_size: Tuple[int, int] = (1920, 1080)) -> Dict:
    """Analyze how well keypoints cover the camera's field of view"""
    all_points = []
    all_confidences = []
    
    for part_data in keypoints.values():
        valid_mask = part_data['confidence'] > 0.1
        all_points.extend(part_data['points'][valid_mask])
        all_confidences.extend(part_data['confidence'][valid_mask])
    
    points = np.array(all_points)
    confidences = np.array(all_confidences)
    
    # Calculate coverage metrics
    x_coverage = (points[:, 0].max() - points[:, 0].min()) / image_size[0]
    y_coverage = (points[:, 1].max() - points[:, 1].min()) / image_size[1]
    
    # Create spatial density heatmap
    heatmap_res = (32, 32)
    heatmap = np.zeros(heatmap_res)
    
    for pt, conf in zip(points, confidences):
        x_bin = int(pt[0] * heatmap_res[0] / image_size[0])
        y_bin = int(pt[1] * heatmap_res[1] / image_size[1])
        x_bin = min(max(x_bin, 0), heatmap_res[0]-1)
        y_bin = min(max(y_bin, 0), heatmap_res[1]-1)
        heatmap[y_bin, x_bin] += conf
    
    return {
        'x_coverage': float(x_coverage),
        'y_coverage': float(y_coverage),
        'num_points': len(points),
        'mean_confidence': float(np.mean(confidences)),
        'spatial_density': heatmap.tolist()
    }

def validate_frame(ehf_path: str, frame_id: str) -> Dict:
    """
    Perform combined validation of camera parameters and keypoints
    
    Args:
        ehf_path: Path to EHF dataset
        frame_id: Frame ID to analyze
        
    Returns:
        Dict containing validation results
    """
    validation = {
        'frame_id': frame_id,
        'status': 'initialized',
        'camera_params': None,
        'keypoints': None,
        'coverage_analysis': None,
        'warnings': []
    }
    
    # Load and validate camera parameters
    camera_params = load_camera_parameters(Path(ehf_path))
    if camera_params is None:
        validation['status'] = 'failed'
        validation['warnings'].append("Failed to load camera parameters")
        return validation
    
    validation['camera_params'] = camera_params
    
    # Load and validate keypoints
    keypoints = load_keypoints(Path(ehf_path), frame_id)
    if keypoints is None:
        validation['status'] = 'failed'
        validation['warnings'].append("Failed to load keypoints")
        return validation
    
    validation['keypoints'] = {
        'counts': {k: len(v['points']) for k, v in keypoints.items()},
        'mean_confidence': {k: float(v['confidence'].mean()) for k, v in keypoints.items()}
    }
    
    # Analyze camera coverage
    validation['coverage_analysis'] = analyze_camera_coverage(keypoints)
    
    # Validate parameter ranges
    focal_length = camera_params['intrinsics']['focal_length']
    principal_point = camera_params['intrinsics']['principal_point']
    
    if any(f <= 0 for f in focal_length):
        validation['warnings'].append("Invalid focal length (must be positive)")
    
    if abs(focal_length[0]/focal_length[1] - 1.0) > 0.1:
        validation['warnings'].append("Unusual focal length ratio")
    
    if not validation['warnings']:
        validation['status'] = 'valid'
    
    return validation

if __name__ == "__main__":
    import argparse
    import json
    from pprint import pprint
    
    parser = argparse.ArgumentParser(description="Validate frame data")
    parser.add_argument("--ehf_path", type=str, default="data/EHF", help="Path to EHF dataset")
    parser.add_argument("--frame_id", type=str, required=True, help="Frame ID to analyze")
    parser.add_argument("--save", action="store_true", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    results = validate_frame(args.ehf_path, args.frame_id)
    
    print("\n📊 Frame Validation Results:")
    print(f"Status: {results['status']}")
    
    if results['warnings']:
        print("\n⚠️ Warnings:")
        for warning in results['warnings']:
            print(f"- {warning}")
    
    if results['camera_params']:
        print("\n📷 Camera Parameters:")
        cam = results['camera_params']
        print(f"Format: {cam['format']}")
        print(f"Focal Length: {cam['intrinsics']['focal_length']}")
        print(f"Principal Point: {cam['intrinsics']['principal_point']}")
    
    if results['keypoints']:
        print("\n🔑 Keypoint Detection:")
        for part, count in results['keypoints']['counts'].items():
            conf = results['keypoints']['mean_confidence'][part]
            print(f"{part}: {count} points (mean conf: {conf:.3f})")
    
    if results['coverage_analysis']:
        print("\n📏 Coverage Analysis:")
        cov = results['coverage_analysis']
        print(f"X Coverage: {cov['x_coverage']*100:.1f}%")
        print(f"Y Coverage: {cov['y_coverage']*100:.1f}%")
        print(f"Total Points: {cov['num_points']}")
        print(f"Mean Confidence: {cov['mean_confidence']:.3f}")
    
    if args.save:
        output_path = Path(f"validation_results_{args.frame_id}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")