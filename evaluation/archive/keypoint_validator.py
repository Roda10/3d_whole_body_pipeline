"""
OpenPose keypoint validation script for EHF dataset
"""

import numpy as np
from pathlib import Path
import json
from typing import Dict
import matplotlib.pyplot as plt

def validate_keypoints(ehf_path: str, frame_id: str) -> Dict:
    """
    Validate 2D keypoint data from OpenPose format
    
    Args:
        ehf_path: Path to EHF dataset
        frame_id: Frame ID to analyze
    
    Returns:
        Dict containing keypoint analysis
    """
    validation = {
        'frame_id': frame_id,
        'keypoint_stats': {},
        'coverage': {},
        'confidence': {}
    }
    
    # Load 2D keypoints from OpenPose format
    joints_2d_path = Path(ehf_path) / f"{frame_id}_2Djnt.json"
    if not joints_2d_path.exists():
        print(f"No joint data found at {joints_2d_path}")
        return validation
        
    with open(joints_2d_path, 'r') as f:
        joints_data = json.load(f)
        if 'people' not in joints_data or not joints_data['people']:
            print("No people detected in the frame")
            return validation
            
        person = joints_data['people'][0]  # Take first person
        keypoint_types = {
            'body': ('pose_keypoints_2d', 25),  # OpenPose body has 25 keypoints
            'face': ('face_keypoints_2d', 70),  # OpenPose face has 70 keypoints
            'hand_left': ('hand_left_keypoints_2d', 21),  # Each hand has 21 keypoints
            'hand_right': ('hand_right_keypoints_2d', 21)
        }
        
        for part_name, (key, num_keypoints) in keypoint_types.items():
            if key in person:
                # Extract keypoints array [x1,y1,c1,x2,y2,c2,...]
                keypoints = np.array(person[key]).reshape(-1, 3)
                
                # Calculate statistics
                valid_points = keypoints[keypoints[:, 2] > 0.1]  # Points with confidence > 0.1
                if len(valid_points) > 0:
                    validation['keypoint_stats'][part_name] = {
                        'total': num_keypoints,
                        'detected': len(valid_points),
                        'detection_rate': len(valid_points) / num_keypoints
                    }
                    validation['coverage'][part_name] = {
                        'x_min': float(np.min(valid_points[:, 0])),
                        'x_max': float(np.max(valid_points[:, 0])),
                        'y_min': float(np.min(valid_points[:, 1])),
                        'y_max': float(np.max(valid_points[:, 1]))
                    }
                    validation['confidence'][part_name] = {
                        'mean': float(np.mean(valid_points[:, 2])),
                        'min': float(np.min(valid_points[:, 2])),
                        'max': float(np.max(valid_points[:, 2]))
                    }
                    
        print("\n✅ Successfully analyzed keypoints")
    
    return validation

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate OpenPose keypoints")
    parser.add_argument("--ehf_path", type=str, default="data/EHF", help="Path to EHF dataset")
    parser.add_argument("--frame_id", type=str, required=True, help="Frame ID to analyze")
    
    args = parser.parse_args()
    
    results = validate_keypoints(args.ehf_path, args.frame_id)
    
    print("\n📊 Keypoint Analysis Results:")
    for part_name in results['keypoint_stats']:
        print(f"\n{part_name.upper()}:")
        stats = results['keypoint_stats'][part_name]
        conf = results['confidence'][part_name]
        cover = results['coverage'][part_name]
        
        print(f"Detection Rate: {stats['detection_rate']*100:.1f}% ({stats['detected']}/{stats['total']} points)")
        print(f"Confidence: Mean={conf['mean']:.3f}, Range=[{conf['min']:.3f}, {conf['max']:.3f}]")
        print(f"X Coverage: {cover['x_min']:.1f} to {cover['x_max']:.1f}")
        print(f"Y Coverage: {cover['y_min']:.1f} to {cover['y_max']:.1f}")