"""
Camera utilities for handling different benchmark datasets

Example Usage:
    # For EHF dataset
    camera_params = CameraParameters.from_ehf(Path("data/EHF/EHF_camera.txt"))

    # For custom dataset
    custom_camera = CameraParameters(
        focal_length=(1000.0, 1000.0),
        principal_point=(512, 512),
        dataset_type='custom'
    )
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import cv2

class CameraParameters:
    """Unified camera parameter handler for different benchmarks"""
    
    def __init__(self, 
                 focal_length: Tuple[float, float] = None,
                 principal_point: Tuple[float, float] = None,
                 rotation: Optional[np.ndarray] = None,
                 translation: Optional[np.ndarray] = None,
                 distortion: Optional[np.ndarray] = None,
                 dataset_type: str = 'custom'):
        """
        Initialize camera parameters
        
        Args:
            focal_length: (fx, fy)
            principal_point: (cx, cy)
            rotation: 3x3 rotation matrix or (3,) Rodrigues vector
            translation: (3,) translation vector
            distortion: (5,) distortion coefficients [k1, k2, p1, p2, k3]
            dataset_type: Type of dataset ('EHF', 'H36M', 'custom', etc.)
        """
        self.focal_length = focal_length or (1000.0, 1000.0)
        self.principal_point = principal_point or (0.0, 0.0)
        self.rotation = rotation if rotation is not None else np.eye(3)
        self.translation = translation if translation is not None else np.zeros(3)
        self.distortion = distortion if distortion is not None else np.zeros(5)
        self.dataset_type = dataset_type
        
        # Convert rotation to matrix if in Rodrigues form
        if self.rotation.shape == (3,):
            self.rotation = cv2.Rodrigues(self.rotation)[0]
    
    @classmethod
    def from_ehf(cls, camera_file: Path) -> 'CameraParameters':
        """Load camera parameters from EHF format"""
        if not camera_file.exists():
            raise FileNotFoundError(f"Camera file not found: {camera_file}")
            
        with open(camera_file, 'r') as f:
            lines = f.readlines()
            
        params = {}
        for i, line in enumerate(lines):
            if "Focal length" in line:
                focal = [float(lines[i+1].strip().split('[')[1].split(']')[0]),
                        float(lines[i+2].strip().split('[')[1].split(']')[0])]
            elif "Principal Point" in line:
                princpt = [float(lines[i+1].strip().split('[')[1].split(']')[0]),
                          float(lines[i+2].strip().split('[')[1].split(']')[0])]
            elif "Translation" in line:
                trans = np.array([float(x) for x in lines[i+1].strip().replace('[','').replace(']','').split()])
            elif "Rotation" in line:
                rot = np.array([float(x) for x in lines[i+1].strip().replace('[','').replace(']','').split()])
                
        return cls(focal_length=tuple(focal),
                  principal_point=tuple(princpt),
                  rotation=rot,
                  translation=trans,
                  dataset_type='EHF')
    
    @classmethod
    def from_json(cls, json_file: Path) -> 'CameraParameters':
        """Load camera parameters from JSON format"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return cls(
            focal_length=tuple(data.get('focal_length', (1000.0, 1000.0))),
            principal_point=tuple(data.get('principal_point', (0.0, 0.0))),
            rotation=np.array(data.get('rotation', np.eye(3).tolist())),
            translation=np.array(data.get('translation', [0,0,0])),
            dataset_type=data.get('dataset_type', 'custom')
        )
    
    def to_json(self, json_file: Path):
        """Save camera parameters to JSON format"""
        data = {
            'focal_length': self.focal_length,
            'principal_point': self.principal_point,
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
            'distortion': self.distortion.tolist(),
            'dataset_type': self.dataset_type
        }
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def project_points(self, points_3d: np.ndarray, 
                      with_extrinsics: bool = True,
                      normalize: bool = False) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates
        
        Args:
            points_3d: (N,3) array of 3D points in world coordinates
            with_extrinsics: Whether to apply extrinsic transformation
            normalize: Whether to normalize coordinates to [-1,1]
            
        Returns:
            points_2d: (N,2) array of 2D image coordinates
        """
        points_3d = np.asarray(points_3d)
        if points_3d.ndim == 2 and points_3d.shape[1] == 3:
            points_3d = points_3d.reshape(-1, 3)
        else:
            raise ValueError(f"Expected (N,3) points, got {points_3d.shape}")
            
        # Apply extrinsics if requested
        if with_extrinsics:
            points_camera = (self.rotation @ points_3d.T + self.translation.reshape(3,1)).T
        else:
            points_camera = points_3d
            
        # Project to 2D
        points_2d = np.zeros((len(points_camera), 2))
        z = points_camera[:,2].reshape(-1,1)
        points_2d[:,0] = self.focal_length[0] * points_camera[:,0] / z + self.principal_point[0]
        points_2d[:,1] = self.focal_length[1] * points_camera[:,1] / z + self.principal_point[1]
        
        if normalize:
            points_2d[:,0] = (points_2d[:,0] - self.principal_point[0]) / self.focal_length[0]
            points_2d[:,1] = (points_2d[:,1] - self.principal_point[1]) / self.focal_length[1]
            
        return points_2d
    
    def get_projection_matrix(self) -> np.ndarray:
        """Get 3x4 projection matrix"""
        K = np.array([[self.focal_length[0], 0, self.principal_point[0]],
                     [0, self.focal_length[1], self.principal_point[1]],
                     [0, 0, 1]])
        RT = np.hstack([self.rotation, self.translation.reshape(3,1)])
        return K @ RT
    
    def verify_units(self, points_3d: np.ndarray, expected_unit: str = 'meters') -> bool:
        """
        Verify that 3D points are in the expected units
        
        Args:
            points_3d: (N,3) array of 3D points
            expected_unit: 'meters' or 'millimeters'
            
        Returns:
            bool: Whether points appear to be in expected units
        """
        max_val = np.max(np.abs(points_3d))
        
        if expected_unit == 'meters':
            # Most human bodies should be 1-2.5m in size
            return max_val < 10.0
        elif expected_unit == 'millimeters':
            # Most human bodies should be 1000-2500mm in size
            return max_val < 10000.0
        else:
            raise ValueError(f"Unknown unit: {expected_unit}")
            
    def __str__(self):
        return f"CameraParameters({self.dataset_type}):\n" + \
               f"  Focal length: {self.focal_length}\n" + \
               f"  Principal point: {self.principal_point}\n" + \
               f"  Rotation shape: {self.rotation.shape}\n" + \
               f"  Translation shape: {self.translation.shape}"