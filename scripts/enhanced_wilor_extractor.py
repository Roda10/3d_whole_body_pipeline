import numpy as np
import torch
import json
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.transform import Rotation as R

class EnhancedWiLoRExtractor:
    """Enhanced extractor for WiLoR MANO parameters with proper coordinate handling"""
    
    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode
    
    def debug_print(self, message: str, data: Any = None):
        """Debug printing with structured output"""
        if self.debug_mode:
            print(f"🔧 [WiLoR Extractor] {message}")
            if data is not None:
                if isinstance(data, (np.ndarray, torch.Tensor)):
                    print(f"   📊 Shape: {data.shape}, Type: {type(data)}")
                    if hasattr(data, 'dtype'):
                        print(f"   📋 Dtype: {data.dtype}")
                    if data.size < 20:  # Only print small arrays
                        print(f"   📄 Values: {data}")
                elif isinstance(data, dict):
                    print(f"   📚 Dict keys: {list(data.keys())}")
                else:
                    print(f"   📝 Value: {data}")
    
    def rotation_matrix_to_axis_angle(self, rot_matrices: np.ndarray) -> np.ndarray:
        """Convert rotation matrices to axis-angle representation"""
        self.debug_print("Converting rotation matrices to axis-angle")
        
        # Handle single matrix
        if rot_matrices.ndim == 2:
            rot_matrices = rot_matrices[np.newaxis, ...]
            single_matrix = True
        else:
            single_matrix = False
            
        self.debug_print(f"Input rotation matrices", rot_matrices)
        
        # Convert each rotation matrix
        axis_angles = []
        for i, rot_mat in enumerate(rot_matrices):
            try:
                # Ensure it's a proper rotation matrix
                if not self.is_valid_rotation_matrix(rot_mat):
                    self.debug_print(f"⚠️  Invalid rotation matrix at index {i}, using identity")
                    rot_mat = np.eye(3)
                
                # Use scipy for robust conversion
                rotation = R.from_matrix(rot_mat)
                axis_angle = rotation.as_rotvec()
                axis_angles.append(axis_angle)
                
                self.debug_print(f"Matrix {i}: converted to axis-angle", axis_angle)
                
            except Exception as e:
                self.debug_print(f"❌ Error converting matrix {i}: {e}")
                axis_angles.append(np.zeros(3))  # Fallback to identity
        
        result = np.array(axis_angles)
        
        # Return original shape
        if single_matrix:
            result = result[0]
            
        self.debug_print(f"Final axis-angle result", result)
        return result
    
    def is_valid_rotation_matrix(self, R: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if matrix is a valid rotation matrix"""
        if R.shape != (3, 3):
            return False
            
        # Check if orthogonal: R @ R.T should be identity
        should_be_identity = np.dot(R, R.T)
        identity = np.eye(3)
        if not np.allclose(should_be_identity, identity, atol=tolerance):
            return False
            
        # Check if determinant is 1 (not -1, which would be reflection)
        if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
            return False
            
        return True
    
    def extract_mano_parameters_from_wilor(self, wilor_output: Dict, hand_idx: int = 0) -> Dict[str, Any]:
        """Extract and convert MANO parameters from WiLoR output"""
        self.debug_print("="*60)
        self.debug_print("EXTRACTING MANO PARAMETERS FROM WiLoR")
        self.debug_print("="*60)
        
        # Check if pred_mano_params exists
        if 'pred_mano_params' not in wilor_output:
            self.debug_print("❌ No 'pred_mano_params' found in WiLoR output")
            self.debug_print("Available keys:", list(wilor_output.keys()))
            return self.create_default_mano_params()
        
        mano_params = wilor_output['pred_mano_params']
        self.debug_print("Found pred_mano_params", mano_params)
        
        # Extract each parameter type
        extracted_params = {}
        
        # 1. Extract Global Orientation (wrist rotation)
        extracted_params['global_orient'] = self.extract_global_orientation(
            mano_params, hand_idx)
        
        # 2. Extract Hand Pose (finger joint rotations)
        extracted_params['hand_pose'] = self.extract_hand_pose(
            mano_params, hand_idx)
        
        # 3. Extract Shape Parameters (if available)
        extracted_params['betas'] = self.extract_shape_parameters(
            mano_params, hand_idx)
        
        # 4. Add metadata
        extracted_params['extraction_metadata'] = {
            'extractor_version': 'enhanced_v1.0',
            'coordinate_system': 'WiLoR_MANO_native',
            'parameter_format': 'axis_angle_for_smplx',
            'hand_index': hand_idx,
            'extraction_success': True
        }
        
        self.debug_print("✅ Successfully extracted MANO parameters")
        return extracted_params
    
    def extract_global_orientation(self, mano_params: Dict, hand_idx: int) -> np.ndarray:
        """Extract global wrist orientation"""
        self.debug_print("\n🔄 Extracting global orientation...")
        
        if 'global_orient' in mano_params:
            global_orient_tensor = mano_params['global_orient'][hand_idx]
            self.debug_print("Found global_orient tensor", global_orient_tensor)
            
            # Convert to numpy
            if torch.is_tensor(global_orient_tensor):
                global_orient = global_orient_tensor.detach().cpu().numpy()
            else:
                global_orient = np.array(global_orient_tensor)
            
            # Check if it's already axis-angle (shape 3,) or rotation matrix (3, 3)
            if global_orient.shape == (3,):
                self.debug_print("Global orient already in axis-angle format")
                return global_orient
            elif global_orient.shape == (3, 3):
                self.debug_print("Global orient is rotation matrix, converting...")
                return self.rotation_matrix_to_axis_angle(global_orient)
            else:
                self.debug_print(f"⚠️  Unexpected global_orient shape: {global_orient.shape}")
                return np.zeros(3)
        else:
            self.debug_print("❌ No global_orient found, using zero rotation")
            return np.zeros(3)
    
    def extract_hand_pose(self, mano_params: Dict, hand_idx: int) -> np.ndarray:
        """Extract hand pose (finger joint rotations)"""
        self.debug_print("\n🖐️  Extracting hand pose...")
        
        if 'hand_pose' in mano_params:
            hand_pose_tensor = mano_params['hand_pose'][hand_idx]
            self.debug_print("Found hand_pose tensor", hand_pose_tensor)
            
            # Convert to numpy
            if torch.is_tensor(hand_pose_tensor):
                hand_pose = hand_pose_tensor.detach().cpu().numpy()
            else:
                hand_pose = np.array(hand_pose_tensor)
            
            self.debug_print(f"Hand pose shape: {hand_pose.shape}")
            
            # Handle different possible formats
            if hand_pose.shape == (45,):
                # Already in axis-angle format
                self.debug_print("Hand pose already in axis-angle format")
                return hand_pose
            elif hand_pose.shape == (15, 3):
                # Axis-angle per joint, flatten
                self.debug_print("Hand pose in (15, 3) format, flattening...")
                return hand_pose.flatten()
            elif hand_pose.shape == (15, 3, 3):
                # Rotation matrices per joint
                self.debug_print("Hand pose in rotation matrix format, converting...")
                axis_angles = self.rotation_matrix_to_axis_angle(hand_pose)
                return axis_angles.flatten()
            else:
                self.debug_print(f"⚠️  Unexpected hand_pose shape: {hand_pose.shape}")
                return np.zeros(45)
        else:
            self.debug_print("❌ No hand_pose found, using zero pose")
            return np.zeros(45)
    
    def extract_shape_parameters(self, mano_params: Dict, hand_idx: int) -> np.ndarray:
        """Extract shape parameters (betas)"""
        self.debug_print("\n📐 Extracting shape parameters...")
        
        if 'betas' in mano_params:
            betas_tensor = mano_params['betas'][hand_idx]
            self.debug_print("Found betas tensor", betas_tensor)
            
            # Convert to numpy
            if torch.is_tensor(betas_tensor):
                betas = betas_tensor.detach().cpu().numpy()
            else:
                betas = np.array(betas_tensor)
            
            # Ensure correct shape
            if betas.shape == (10,):
                return betas
            elif len(betas) >= 10:
                self.debug_print("Truncating betas to first 10 components")
                return betas[:10]
            else:
                self.debug_print(f"⚠️  Padding betas from {len(betas)} to 10")
                padded = np.zeros(10)
                padded[:len(betas)] = betas
                return padded
        else:
            self.debug_print("❌ No betas found, using zero shape")
            return np.zeros(10)
    
    def create_default_mano_params(self) -> Dict[str, Any]:
        """Create default MANO parameters when extraction fails"""
        self.debug_print("Creating default MANO parameters")
        
        return {
            'global_orient': np.zeros(3),
            'hand_pose': np.zeros(45),
            'betas': np.zeros(10),
            'extraction_metadata': {
                'extractor_version': 'enhanced_v1.0',
                'coordinate_system': 'default_identity',
                'parameter_format': 'axis_angle_for_smplx',
                'extraction_success': False,
                'note': 'Used default parameters due to extraction failure'
            }
        }
    
    def validate_extracted_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate the extracted parameters are reasonable"""
        self.debug_print("\n✅ Validating extracted parameters...")
        
        issues = []
        
        # Check global orientation
        global_orient = params['global_orient']
        if global_orient.shape != (3,):
            issues.append(f"Global orient shape {global_orient.shape} != (3,)")
        if np.abs(global_orient).max() > np.pi:
            issues.append(f"Global orient values too large: {np.abs(global_orient).max()}")
        
        # Check hand pose
        hand_pose = params['hand_pose']
        if hand_pose.shape != (45,):
            issues.append(f"Hand pose shape {hand_pose.shape} != (45,)")
        if np.abs(hand_pose).max() > np.pi:
            issues.append(f"Hand pose values too large: {np.abs(hand_pose).max()}")
        
        # Check betas
        betas = params['betas']
        if betas.shape != (10,):
            issues.append(f"Betas shape {betas.shape} != (10,)")
        if np.abs(betas).max() > 5.0:
            issues.append(f"Beta values too large: {np.abs(betas).max()}")
        
        if issues:
            self.debug_print("⚠️  Validation issues found:")
            for issue in issues:
                self.debug_print(f"   - {issue}")
            return False
        else:
            self.debug_print("✅ All parameters validated successfully")
            return True
    
    def extract_from_wilor_batch(self, wilor_batch: Dict, wilor_output: Dict) -> List[Dict[str, Any]]:
        """Extract MANO parameters for all hands in a batch"""
        self.debug_print("\n" + "="*60)
        self.debug_print("BATCH MANO PARAMETER EXTRACTION")
        self.debug_print("="*60)
        
        batch_size = wilor_batch['img'].shape[0] if 'img' in wilor_batch else 1
        self.debug_print(f"Processing batch of size: {batch_size}")
        
        extracted_hands = []
        
        for hand_idx in range(batch_size):
            self.debug_print(f"\n--- Processing hand {hand_idx + 1}/{batch_size} ---")
            
            # Extract parameters for this hand
            hand_params = self.extract_mano_parameters_from_wilor(wilor_output, hand_idx)
            
            # Add batch context
            hand_params['batch_metadata'] = {
                'hand_index': hand_idx,
                'batch_size': batch_size,
                'hand_type': wilor_batch['right'][hand_idx].item() if 'right' in wilor_batch else 'unknown'
            }
            
            # Validate
            is_valid = self.validate_extracted_parameters(hand_params)
            hand_params['extraction_metadata']['validation_passed'] = is_valid
            
            extracted_hands.append(hand_params)
        
        self.debug_print(f"\n✅ Successfully processed {len(extracted_hands)} hands")
        return extracted_hands