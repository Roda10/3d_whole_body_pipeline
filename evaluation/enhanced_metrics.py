"""
Enhanced metrics calculation with proper unit handling and multi-dataset support
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
import trimesh
from pathlib import Path
import json

class EnhancedMetricsCalculator:
    """Calculate metrics with proper unit handling and dataset-specific adjustments"""
    
    def __init__(self, 
                 dataset_type: str = 'EHF',
                 unit: str = 'meters',
                 verbose: bool = False):
        """
        Initialize metrics calculator
        
        Args:
            dataset_type: Type of dataset ('EHF', 'H36M', etc.)
            unit: Expected unit of measurements ('meters' or 'millimeters')
            verbose: Whether to print detailed information
        """
        self.dataset_type = dataset_type
        self.unit = unit
        self.verbose = verbose
        
        # Dataset-specific joint configurations
        self.joint_configs = {
            'EHF': {
                'body': list(range(23)),  # SMPL-X body joints
                'hand_left': list(range(778, 799)),  # Left hand joints
                'hand_right': list(range(799, 820)),  # Right hand joints
                'face': list(range(820, 870))  # Face landmarks
            },
            'H36M': {
                'body': list(range(17))  # H36M body joints
            }
            # Add more datasets as needed
        }
        
    def convert_units(self, points: np.ndarray, 
                     source_unit: str,
                     target_unit: str) -> np.ndarray:
        """Convert between different units"""
        if source_unit == target_unit:
            return points
            
        conversion = {
            ('millimeters', 'meters'): 0.001,
            ('meters', 'millimeters'): 1000.0
        }
        
        key = (source_unit, target_unit)
        if key not in conversion:
            raise ValueError(f"Unsupported unit conversion: {key}")
            
        return points * conversion[key]
    
    def load_mesh(self, 
                 mesh_path: Path,
                 expected_unit: Optional[str] = None) -> trimesh.Trimesh:
        """
        Load mesh with unit verification
        
        Args:
            mesh_path: Path to mesh file
            expected_unit: Expected unit of the mesh ('meters' or 'millimeters')
            
        Returns:
            trimesh.Trimesh: Loaded mesh in the correct units
        """
        mesh = trimesh.load(mesh_path)
        if expected_unit is None:
            expected_unit = self.unit
            
        # Verify units
        max_extent = mesh.bounding_box.extents.max()
        if expected_unit == 'meters' and max_extent > 10:
            if self.verbose:
                print(f"Converting mesh from mm to m (max extent: {max_extent})")
            mesh.vertices = mesh.vertices * 0.001
        elif expected_unit == 'millimeters' and max_extent < 1:
            if self.verbose:
                print(f"Converting mesh from m to mm (max extent: {max_extent})")
            mesh.vertices = mesh.vertices * 1000.0
            
        return mesh
    
    def calculate_v2v(self, 
                     pred_vertices: np.ndarray,
                     gt_vertices: np.ndarray,
                     mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Vertex-to-Vertex (V2V) error
        
        Args:
            pred_vertices: (N,3) predicted vertex positions
            gt_vertices: (N,3) ground truth vertex positions
            mask: Optional boolean mask for vertex subset
            
        Returns:
            float: Mean V2V error in current units
        """
        if pred_vertices.shape != gt_vertices.shape:
            raise ValueError(f"Shape mismatch: {pred_vertices.shape} vs {gt_vertices.shape}")
            
        if mask is not None:
            pred_vertices = pred_vertices[mask]
            gt_vertices = gt_vertices[mask]
            
        distances = np.sqrt(np.sum((pred_vertices - gt_vertices) ** 2, axis=1))
        return float(np.mean(distances))
    
    def calculate_pa_v2v(self,
                        pred_vertices: np.ndarray,
                        gt_vertices: np.ndarray,
                        mask: Optional[np.ndarray] = None) -> float:
        """
        Calculate Procrustes-aligned Vertex-to-Vertex (PA-V2V) error
        
        Args:
            pred_vertices: (N,3) predicted vertex positions
            gt_vertices: (N,3) ground truth vertex positions
            mask: Optional boolean mask for vertex subset
            
        Returns:
            float: Mean PA-V2V error in current units
        """
        if mask is not None:
            pred_vertices = pred_vertices[mask]
            gt_vertices = gt_vertices[mask]
            
        # Center both point sets
        pred_mean = np.mean(pred_vertices, axis=0)
        gt_mean = np.mean(gt_vertices, axis=0)
        
        pred_centered = pred_vertices - pred_mean
        gt_centered = gt_vertices - gt_mean
        
        # Compute optimal rotation
        H = pred_centered.T @ gt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[2,:] *= -1
            R = Vt.T @ U.T
            
        # Apply alignment
        pred_aligned = (R @ pred_centered.T).T + gt_mean
        
        # Calculate error
        return self.calculate_v2v(pred_aligned, gt_vertices)
    
    def calculate_mpjpe(self,
                       pred_joints: np.ndarray,
                       gt_joints: np.ndarray,
                       joint_set: str = 'body') -> float:
        """
        Calculate Mean Per Joint Position Error (MPJPE)
        
        Args:
            pred_joints: (J,3) predicted joint positions
            gt_joints: (J,3) ground truth joint positions
            joint_set: Which joints to evaluate ('body', 'hand_left', etc.)
            
        Returns:
            float: MPJPE in current units
        """
        if self.dataset_type not in self.joint_configs:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        if joint_set not in self.joint_configs[self.dataset_type]:
            raise ValueError(f"Unknown joint set '{joint_set}' for {self.dataset_type}")
            
        joint_indices = self.joint_configs[self.dataset_type][joint_set]
        return self.calculate_v2v(pred_joints[joint_indices], gt_joints[joint_indices])
    
    def calculate_pa_mpjpe(self,
                          pred_joints: np.ndarray,
                          gt_joints: np.ndarray,
                          joint_set: str = 'body') -> float:
        """
        Calculate Procrustes-aligned Mean Per Joint Position Error (PA-MPJPE)
        
        Args:
            pred_joints: (J,3) predicted joint positions
            gt_joints: (J,3) ground truth joint positions
            joint_set: Which joints to evaluate ('body', 'hand_left', etc.)
            
        Returns:
            float: PA-MPJPE in current units
        """
        if self.dataset_type not in self.joint_configs:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")
            
        if joint_set not in self.joint_configs[self.dataset_type]:
            raise ValueError(f"Unknown joint set '{joint_set}' for {self.dataset_type}")
            
        joint_indices = self.joint_configs[self.dataset_type][joint_set]
        return self.calculate_pa_v2v(pred_joints[joint_indices], gt_joints[joint_indices])
    
    def calculate_surface_metrics(self,
                                pred_mesh: trimesh.Trimesh,
                                gt_mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Calculate surface-based metrics
        
        Args:
            pred_mesh: Predicted mesh
            gt_mesh: Ground truth mesh
            
        Returns:
            dict: Dictionary of metric names and values
        """
        # Chamfer distance (bi-directional)
        pred_points = pred_mesh.sample(5000)
        gt_points = gt_mesh.sample(5000)
        
        pred_tree = trimesh.PointCloud(pred_points).kdtree
        gt_tree = trimesh.PointCloud(gt_points).kdtree
        
        # Closest point distances
        pred_to_gt_dist, _ = gt_tree.query(pred_points)
        gt_to_pred_dist, _ = pred_tree.query(gt_points)
        
        return {
            'chamfer_pred_to_gt': float(np.mean(pred_to_gt_dist)),
            'chamfer_gt_to_pred': float(np.mean(gt_to_pred_dist)),
            'chamfer_symmetric': float(np.mean(pred_to_gt_dist) + np.mean(gt_to_pred_dist)) / 2
        }
    
    def calculate_all_metrics(self,
                            pred_mesh: trimesh.Trimesh,
                            gt_mesh: trimesh.Trimesh,
                            pred_joints: Optional[np.ndarray] = None,
                            gt_joints: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate all available metrics
        
        Args:
            pred_mesh: Predicted mesh
            gt_mesh: Ground truth mesh
            pred_joints: Optional predicted joint positions
            gt_joints: Optional ground truth joint positions
            
        Returns:
            dict: Dictionary of all metrics
        """
        metrics = {}
        
        # Vertex-based metrics
        metrics['v2v'] = self.calculate_v2v(pred_mesh.vertices, gt_mesh.vertices)
        metrics['pa_v2v'] = self.calculate_pa_v2v(pred_mesh.vertices, gt_mesh.vertices)
        
        # Surface metrics
        surface_metrics = self.calculate_surface_metrics(pred_mesh, gt_mesh)
        metrics.update(surface_metrics)
        
        # Joint metrics if available
        if pred_joints is not None and gt_joints is not None:
            for joint_set in self.joint_configs[self.dataset_type].keys():
                try:
                    metrics[f'mpjpe_{joint_set}'] = self.calculate_mpjpe(
                        pred_joints, gt_joints, joint_set)
                    metrics[f'pa_mpjpe_{joint_set}'] = self.calculate_pa_mpjpe(
                        pred_joints, gt_joints, joint_set)
                except:
                    if self.verbose:
                        print(f"Skipping joint metrics for {joint_set}")
        
        return metrics