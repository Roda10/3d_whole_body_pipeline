"""
Camera parameter validation and drift analysis for EHF dataset
"""

import numpy as np
import trimesh
from pathlib import Path
import json
import cv2
from typing import Dict, Tuple
import matplotlib.pyplot as plt

def validate_camera_parameters(ehf_path: str, frame_id: str) -> Dict:
    """
    Validate camera parameter usage by analyzing projections
    
    Args:
        ehf_path: Path to EHF dataset
        frame_id: Frame ID to analyze
    
    Returns:
        Dict containing validation results
    """
    # Initialize validation results
    validation = {
        'frame_id': frame_id,
        'camera_params': {},
        'projection_error': {},
        'keypoint_coverage': {}
    }
    
    # Load camera parameters from JSON file
    json_path = Path(ehf_path) / f"{frame_id}_cam.json"
    # Load camera parameters from camera file
    camera_file = Path(ehf_path) / "camera_parameters.json"  # Adjust this path based on actual file location
    if not camera_file.exists():
        print(f"Camera file not found at {camera_file}. Looking for camera info in frame data...")
        # Try loading from frame data
        frame_json = Path(ehf_path) / f"{frame_id}_2Djnt.json"
        if frame_json.exists():
            with open(frame_json, 'r') as f:
                frame_data = json.load(f)
                # TODO: Extract camera parameters from frame data if available
                print("Found frame data, but camera parameters not implemented yet")
        return validation
    
    with open(camera_file, 'r') as f:
        cam_data = json.load(f)
        # Adapt this based on the actual format of your camera parameter file
        if 'intrinsics' in cam_data:
            focal_length = np.array(cam_data['intrinsics']['focal_length'])
            principal_point = np.array(cam_data['intrinsics']['principal_point'])
        else:
            focal_length = np.array([1000.0, 1000.0])  # Default values
            principal_point = np.array([960.0, 540.0])  # Default values for 1920x1080
            
        if 'extrinsics' in cam_data:
            rotation = np.array(cam_data['extrinsics']['rotation'])
            translation = np.array(cam_data['extrinsics']['translation'])
        else:
            rotation = np.zeros(3)  # Default values
            translation = np.zeros(3)
        
        print(f"Found focal length: {focal_length}")
        print(f"Found principal point: {principal_point}")
        print(f"Found translation: {translation}")
        print(f"Found rotation: {rotation}")
        
        validation['camera_params'] = {
            'focal_length': focal_length.tolist(),
            'principal_point': principal_point.tolist(),
            'rotation': rotation.tolist(),
            'translation': translation.tolist()
        }
    
    # Load ground truth data
    gt_mesh_path = Path(ehf_path) / f"{frame_id}_align.ply"
    gt_mesh = trimesh.load(str(gt_mesh_path))
    gt_vertices = np.array(gt_mesh.vertices)
    
    # Project ground truth vertices to 2D
    R = cv2.Rodrigues(rotation)[0]
    projected_points = project_3d_to_2d(gt_vertices, focal_length, principal_point, R, translation)
    latest_run = sorted(results_dir.glob("ehf_compatible_opt_*"))[-1]
    result_file = latest_run / "evaluation_comparison_results.json"
    
    with open(result_file, 'r') as f:
        eval_results = json.load(f)
    
    # Initialize validation results
    validation = {
        'frame_id': frame_id,
        'camera_params': {},
        'projection_error': {},
        'keypoint_coverage': {}
    }
    
    # Load camera parameters
    # Note: Camera parameters are loaded directly in the validation function
    
    # Store camera parameters in validation results
    validation['camera_params'] = {
        'focal_length': focal_length.tolist(),
        'principal_point': principal_point.tolist(),
        'rotation': rotation.tolist(),
        'translation': translation.tolist()
    }
    
    # 1. Check 3D-to-2D Projection Accuracy
    R = cv2.Rodrigues(rotation)[0]
    projected_points = project_3d_to_2d(gt_vertices, focal_length, principal_point, R, translation)
    
    # Load 2D ground truth joints if available
    joints_2d_path = Path(ehf_path) / f"{frame_id}_2Djnt.json"
    if joints_2d_path.exists():
        with open(joints_2d_path, 'r') as f:
            joints_2d_gt = json.load(f)
            if 'people' in joints_2d_gt and len(joints_2d_gt['people']) > 0:
                # OpenPose format: [x1,y1,c1,x2,y2,c2,...]
                pose_keypoints = np.array(joints_2d_gt['people'][0]['pose_keypoints_2d'])
                # Reshape to (N,3) and extract only x,y coordinates
                pose_keypoints = pose_keypoints.reshape(-1, 3)[:, :2]
                # Calculate reprojection error
                validation['projection_error']['mean_error_pixels'] = calculate_reprojection_error(
                    projected_points, pose_keypoints
                )
                validation['projection_error']['num_keypoints'] = len(pose_keypoints)
            else:
                print("No people found in the joints file")
    
    # Calculate coverage statistics for keypoints
    if 'num_keypoints' in validation['projection_error']:
        validation['keypoint_coverage']['total_keypoints'] = validation['projection_error']['num_keypoints']
        validation['keypoint_coverage']['valid_keypoints'] = len([x for x in pose_keypoints if np.all(np.isfinite(x))])
        
        # Calculate spatial distribution of keypoints
        if validation['keypoint_coverage']['valid_keypoints'] > 0:
            valid_points = pose_keypoints[np.all(np.isfinite(pose_keypoints), axis=1)]
            min_coords = np.min(valid_points, axis=0)
            max_coords = np.max(valid_points, axis=0)
            validation['keypoint_coverage']['bounding_box'] = {
                'min_x': float(min_coords[0]),
                'min_y': float(min_coords[1]),
                'max_x': float(max_coords[0]),
                'max_y': float(max_coords[1])
            }
    
    return validation

def project_3d_to_2d(points_3d: np.ndarray, focal_length: list, principal_point: list,
                     rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using camera parameters"""
    # Apply extrinsics
    points_camera = (rotation @ points_3d.T + translation.reshape(3,1)).T
    
    # Project to 2D
    points_2d = np.zeros((len(points_camera), 2))
    points_2d[:,0] = focal_length[0] * points_camera[:,0] / points_camera[:,2] + principal_point[0]
    points_2d[:,1] = focal_length[1] * points_camera[:,1] / points_camera[:,2] + principal_point[1]
    
    return points_2d

def calculate_reprojection_error(projected_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Calculate mean reprojection error in pixels for corresponding points
    
    Args:
        projected_points: All projected mesh vertices (N,2)
        gt_points: OpenPose keypoints (K,2)
    """
    # For each gt point, find the closest projected point
    # This is a simple nearest neighbor approach - in practice you might want to use
    # semantic correspondences between mesh vertices and OpenPose keypoints
    errors = []
    for gt_point in gt_points:
        # Find closest projected point
        dists = np.linalg.norm(projected_points - gt_point, axis=1)
        min_dist = np.min(dists)
        errors.append(min_dist)
    
    return float(np.mean(errors))

def procrustes_align(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform Procrustes alignment"""
    # Center the point sets
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    
    # Calculate optimal rotation
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = target_centroid - (R @ source_centroid)
    
    # Apply alignment
    aligned_source = (R @ source.T).T + t
    
    return aligned_source, R, t

def analyze_drift(predicted: np.ndarray, ground_truth: np.ndarray) -> Dict:
    """Analyze various types of drift between predicted and ground truth meshes"""
    drift_analysis = {}
    
    # Global translation drift
    pred_centroid = np.mean(predicted, axis=0)
    gt_centroid = np.mean(ground_truth, axis=0)
    translation_drift = np.linalg.norm(pred_centroid - gt_centroid)
    drift_analysis['global_translation_drift_mm'] = float(translation_drift * 1000)  # Convert to mm
    
    # Scale drift
    pred_scale = np.mean(np.linalg.norm(predicted - pred_centroid, axis=1))
    gt_scale = np.mean(np.linalg.norm(ground_truth - gt_centroid, axis=1))
    scale_drift = abs(pred_scale - gt_scale) / gt_scale * 100  # as percentage
    drift_analysis['scale_drift_percent'] = float(scale_drift)
    
    # Local deformation analysis
    # Compare local neighborhood structures
    k = 5  # number of neighbors to consider
    local_deformation = analyze_local_deformation(predicted, ground_truth, k)
    drift_analysis['local_deformation_mm'] = float(local_deformation * 1000)  # Convert to mm
    
    return drift_analysis

def analyze_local_deformation(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Analyze local neighborhood preservation"""
    # No longer needed
    
    # Build kd-trees
    pred_tree = cKDTree(predicted)
    gt_tree = cKDTree(ground_truth)
    
    # Get k-nearest neighbors for each point
    _, pred_nn = pred_tree.query(predicted, k=k)
    _, gt_nn = gt_tree.query(ground_truth, k=k)
    
    # Compare neighborhood distances
    pred_distances = np.linalg.norm(
        predicted[pred_nn[:, 1:]] - predicted[:, np.newaxis], axis=2
    )
    gt_distances = np.linalg.norm(
        ground_truth[gt_nn[:, 1:]] - ground_truth[:, np.newaxis], axis=2
    )
    
    # Calculate mean difference in neighborhood distances
    local_deformation = np.mean(np.abs(pred_distances - gt_distances))
    
    return float(local_deformation)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate camera parameters and keypoint projections")
    parser.add_argument("--ehf_path", type=str, default="data/EHF", help="Path to EHF dataset")
    parser.add_argument("--frame_id", type=str, required=True, help="Frame ID to analyze")
    
    args = parser.parse_args()
    
    validation_results = validate_camera_parameters(args.ehf_path, args.frame_id)
    
    print("\n📊 Camera Parameter Validation Results:")
    print("\nCamera Parameters:")
    print(f"Focal Length: {validation_results['camera_params']['focal_length']}")
    print(f"Principal Point: {validation_results['camera_params']['principal_point']}")
    print(f"Translation: {validation_results['camera_params']['translation']}")
    print(f"Rotation: {validation_results['camera_params']['rotation']}")
    
    if 'projection_error' in validation_results:
        print("\nProjection Analysis:")
        print(f"Mean Error (pixels): {validation_results['projection_error'].get('mean_error_pixels', 'N/A')}")
        print(f"Total Keypoints: {validation_results['keypoint_coverage'].get('total_keypoints', 'N/A')}")
        print(f"Valid Keypoints: {validation_results['keypoint_coverage'].get('valid_keypoints', 'N/A')}")
        
        if 'bounding_box' in validation_results.get('keypoint_coverage', {}):
            bb = validation_results['keypoint_coverage']['bounding_box']
            print("\nKeypoint Spatial Coverage:")
            print(f"X-Range: {bb['min_x']:.1f} to {bb['max_x']:.1f}")
            print(f"Y-Range: {bb['min_y']:.1f} to {bb['max_y']:.1f}")