#!/usr/bin/env python3
"""
Enhanced EHF Evaluator with Literature Metrics - Based on Your Working Evaluator
ONLY replaces calculate_metrics function while keeping your working pipeline intact
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import trimesh
import time as pytime
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime
from tqdm import tqdm
import tempfile
import psutil

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from main.config import Config
    from human_models.human_models import SMPL, SMPLX
except ImportError as e:
    print(f"Error importing SMPLest-X modules: {e}")
    sys.exit(1)

class LiteratureMetricsCalculator:
    """Literature-based metrics using your existing data structure"""
    
    def __init__(self):
        print(f"✅ Literature metrics calculator initialized")
        print(f"   Using established metrics: MPJPE-2D, Hand-MPJPE, PCK@k, Face-NME")
    
    def load_ehf_ground_truth(self, joints_2d_path: Path) -> Dict:
        """Load EHF 2D joint ground truth - same as before"""
        if not joints_2d_path.exists():
            raise FileNotFoundError(f"EHF ground truth not found: {joints_2d_path}")
        
        with open(joints_2d_path, 'r') as f:
            ehf_data = json.load(f)
        
        if 'people' not in ehf_data or len(ehf_data['people']) == 0:
            raise ValueError(f"Invalid EHF format in {joints_2d_path}")
        
        person = ehf_data['people'][0]
        
        def parse_keypoints(keypoint_list):
            return np.array(keypoint_list).reshape(-1, 3)
        
        gt_data = {
            'body_keypoints': parse_keypoints(person['pose_keypoints_2d']),      # (25, 3)
            'face_keypoints': parse_keypoints(person['face_keypoints_2d']),      # (68, 3)
            'left_hand_keypoints': parse_keypoints(person['hand_left_keypoints_2d']),   # (21, 3)
            'right_hand_keypoints': parse_keypoints(person['hand_right_keypoints_2d']), # (21, 3)
        }
        
        return gt_data
    
    def calculate_mpjpe_2d(self, pred_joints: np.ndarray, gt_joints: np.ndarray, 
                          confidence_threshold: float = 0.3) -> float:
        """Standard MPJPE-2D calculation"""
        try:
            if len(gt_joints) == 0 or len(pred_joints) == 0:
                return float('inf')
            
            # Filter by confidence
            valid_mask = gt_joints[:, 2] > confidence_threshold
            if not np.any(valid_mask):
                return float('inf')
            
            # Get valid joints
            valid_pred = pred_joints[valid_mask]
            valid_gt = gt_joints[valid_mask, :2]
            
            # Handle size mismatch
            min_len = min(len(valid_pred), len(valid_gt))
            if min_len == 0:
                return float('inf')
            
            valid_pred = valid_pred[:min_len]
            valid_gt = valid_gt[:min_len]
            
            # Calculate L2 distances
            errors = np.linalg.norm(valid_pred - valid_gt, axis=1)
            return float(np.mean(errors))
            
        except Exception as e:
            print(f"⚠️ MPJPE-2D calculation failed: {e}")
            return float('inf')
    
    def calculate_pck(self, pred_joints: np.ndarray, gt_joints: np.ndarray, 
                     threshold: float, confidence_threshold: float = 0.3) -> float:
        """PCK@k calculation"""
        try:
            if len(gt_joints) == 0 or len(pred_joints) == 0:
                return 0.0
            
            # Filter by confidence
            valid_mask = gt_joints[:, 2] > confidence_threshold
            if not np.any(valid_mask):
                return 0.0
            
            # Get valid joints
            valid_pred = pred_joints[valid_mask]
            valid_gt = gt_joints[valid_mask, :2]
            
            # Handle size mismatch
            min_len = min(len(valid_pred), len(valid_gt))
            if min_len == 0:
                return 0.0
            
            valid_pred = valid_pred[:min_len]
            valid_gt = valid_gt[:min_len]
            
            # Calculate distances and check threshold
            distances = np.linalg.norm(valid_pred - valid_gt, axis=1)
            correct_keypoints = distances < threshold
            
            return float(np.mean(correct_keypoints.astype(float)))
            
        except Exception as e:
            print(f"⚠️ PCK calculation failed: {e}")
            return 0.0
    
    def calculate_literature_metrics(self, predicted_joints_2d: np.ndarray, gt_data: Dict) -> Dict:
        """
        Calculate all literature metrics using your existing joints_2d data
        
        Args:
            predicted_joints_2d: Your existing joints_2d from baseline_result/fusion_result
            gt_data: EHF ground truth data
        """
        results = {}
        
        try:
            # Determine how many joints we have from your SMPL-X output
            num_pred_joints = len(predicted_joints_2d)
            print(f"      📊 SMPL-X joints available: {num_pred_joints}")
            
            # 1. Overall MPJPE-2D (using first 25 joints if available)
            if num_pred_joints >= 25:
                body_pred = predicted_joints_2d[:25]
                results['MPJPE_2D'] = self.calculate_mpjpe_2d(body_pred, gt_data['body_keypoints'])
            else:
                # Use all available joints
                results['MPJPE_2D'] = self.calculate_mpjpe_2d(predicted_joints_2d, gt_data['body_keypoints'])
            
            # 2. Hand-specific metrics (try different joint ranges)
            hand_mpjpe_left = float('inf')
            hand_mpjpe_right = float('inf')
            
            # Try to find hand joints in your SMPL-X output
            # Common SMPL-X configurations have hands at different indices
            possible_left_hand_ranges = [
                (25, 46),   # Standard SMPL-X
                (22, 43),   # Alternative
                (num_pred_joints-42, num_pred_joints-21) if num_pred_joints > 42 else (0, 0)  # End of joints
            ]
            
            for start, end in possible_left_hand_ranges:
                if start < num_pred_joints and end <= num_pred_joints and end > start:
                    try:
                        left_hand_pred = predicted_joints_2d[start:end]
                        if len(left_hand_pred) == 21:  # Correct hand joint count
                            hand_mpjpe_left = self.calculate_mpjpe_2d(left_hand_pred, gt_data['left_hand_keypoints'])
                            if hand_mpjpe_left != float('inf'):
                                break
                    except:
                        continue
            
            # Right hand (usually follows left hand)
            possible_right_hand_ranges = [
                (46, 67),   # Standard SMPL-X
                (43, 64),   # Alternative  
                (num_pred_joints-21, num_pred_joints) if num_pred_joints > 21 else (0, 0)  # Very end
            ]
            
            for start, end in possible_right_hand_ranges:
                if start < num_pred_joints and end <= num_pred_joints and end > start:
                    try:
                        right_hand_pred = predicted_joints_2d[start:end]
                        if len(right_hand_pred) == 21:  # Correct hand joint count
                            hand_mpjpe_right = self.calculate_mpjpe_2d(right_hand_pred, gt_data['right_hand_keypoints'])
                            if hand_mpjpe_right != float('inf'):
                                break
                    except:
                        continue
            
            # Store hand metrics
            results['Hand_MPJPE_Left'] = float(hand_mpjpe_left)
            results['Hand_MPJPE_Right'] = float(hand_mpjpe_right)
            
            # Average hand MPJPE
            if hand_mpjpe_left != float('inf') and hand_mpjpe_right != float('inf'):
                results['Hand_MPJPE'] = float((hand_mpjpe_left + hand_mpjpe_right) / 2.0)
            elif hand_mpjpe_left != float('inf'):
                results['Hand_MPJPE'] = float(hand_mpjpe_left)
            elif hand_mpjpe_right != float('inf'):
                results['Hand_MPJPE'] = float(hand_mpjpe_right)
            else:
                results['Hand_MPJPE'] = float('inf')
            
            # 3. PCK@k for body joints
            if num_pred_joints >= 25:
                body_pred = predicted_joints_2d[:25]
                results['PCK@20'] = float(self.calculate_pck(body_pred, gt_data['body_keypoints'], 20.0))
                results['PCK@50'] = float(self.calculate_pck(body_pred, gt_data['body_keypoints'], 50.0))
            else:
                results['PCK@20'] = float(self.calculate_pck(predicted_joints_2d, gt_data['body_keypoints'], 20.0))
                results['PCK@50'] = float(self.calculate_pck(predicted_joints_2d, gt_data['body_keypoints'], 50.0))
            
            # 4. Hand PCK@20
            hand_pck_scores = []
            if hand_mpjpe_left != float('inf'):
                # Find the left hand joints that worked
                for start, end in possible_left_hand_ranges:
                    if start < num_pred_joints and end <= num_pred_joints and end > start:
                        try:
                            left_hand_pred = predicted_joints_2d[start:end]
                            if len(left_hand_pred) == 21:
                                pck_left = self.calculate_pck(left_hand_pred, gt_data['left_hand_keypoints'], 20.0)
                                if pck_left > 0:
                                    hand_pck_scores.append(pck_left)
                                    break
                        except:
                            continue
            
            if hand_mpjpe_right != float('inf'):
                # Find the right hand joints that worked
                for start, end in possible_right_hand_ranges:
                    if start < num_pred_joints and end <= num_pred_joints and end > start:
                        try:
                            right_hand_pred = predicted_joints_2d[start:end]
                            if len(right_hand_pred) == 21:
                                pck_right = self.calculate_pck(right_hand_pred, gt_data['right_hand_keypoints'], 20.0)
                                if pck_right > 0:
                                    hand_pck_scores.append(pck_right)
                                    break
                        except:
                            continue
            
            results['Hand_PCK@20'] = float(np.mean(hand_pck_scores)) if hand_pck_scores else 0.0
            
            # 5. Face NME (if face joints available)
            face_nme = float('inf')
            if num_pred_joints > 67:  # Face joints typically start after hands
                possible_face_ranges = [
                    (67, 127),  # Standard SMPL-X
                    (68, 136),  # Alternative
                ]
                
                for start, end in possible_face_ranges:
                    if start < num_pred_joints and end <= num_pred_joints:
                        try:
                            face_pred = predicted_joints_2d[start:end]
                            gt_face = gt_data['face_keypoints']
                            
                            # Handle size mismatch
                            min_joints = min(len(face_pred), len(gt_face))
                            if min_joints > 10:  # Need reasonable number of face points
                                face_pred = face_pred[:min_joints]
                                gt_face = gt_face[:min_joints]
                                
                                # Simple NME calculation
                                valid_mask = gt_face[:, 2] > 0.3
                                if np.any(valid_mask):
                                    valid_pred = face_pred[valid_mask]
                                    valid_gt = gt_face[valid_mask, :2]
                                    
                                    # Use bounding box for normalization
                                    bbox_diag = np.linalg.norm(np.max(valid_gt, axis=0) - np.min(valid_gt, axis=0))
                                    if bbox_diag > 0:
                                        errors = np.linalg.norm(valid_pred - valid_gt, axis=1)
                                        face_nme = np.mean(errors) / bbox_diag
                                        break
                        except:
                            continue
            
            results['Face_NME'] = float(face_nme)
            results['Face_FR'] = float(1.0 if face_nme > 0.1 else 0.0) if face_nme != float('inf') else 1.0
            results['face_available'] = bool(face_nme != float('inf'))
            
            return results
            
        except Exception as e:
            print(f"⚠️ Literature metrics calculation failed: {e}")
            return {
                'MPJPE_2D': float('inf'),
                'Hand_MPJPE': float('inf'),
                'Hand_MPJPE_Left': float('inf'),
                'Hand_MPJPE_Right': float('inf'),
                'PCK@20': 0.0,
                'PCK@50': 0.0,
                'Hand_PCK@20': 0.0,
                'Face_NME': float('inf'),
                'Face_FR': 1.0,
                'face_available': False,
                'calculation_error': str(e)
            }

class CompatibleOptimizedEHFEvaluator:
    """YOUR EXACT WORKING EVALUATOR with only calculate_metrics replaced"""
    
    def __init__(self, ehf_path: str = "data/EHF", 
                 smplestx_config: str = "pretrained_models/smplest_x/config_base.py"):
        self.ehf_path = Path(ehf_path)
        self.config = Config.load_config(smplestx_config)
        
        # Initialize literature metrics calculator
        self.lit_metrics = LiteratureMetricsCalculator()
        
        # Setup output directory
        self.output_dir = Path("evaluation_results") / f"ehf_compatible_opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories
        self.central_gallery = self.output_dir / "gallery"
        self.central_gallery.mkdir(exist_ok=True)
        self.shared_temp_dir = self.output_dir / "shared_temp"
        self.shared_temp_dir.mkdir(exist_ok=True)
        
        # Load frame list and setup
        self.frames = self._load_frame_list()
        self.camera_params = self._load_camera_params()
        
        # Setup config and models
        self.setup_config()
        self.setup_models()
        
        # Performance tracking
        self.timing_stats = {
            'main_pipeline': [],
            'coordinate_analysis': [],
            'fusion': [],
            'visualization': []
        }
        
        print(f"✅ Compatible Optimized EHF Evaluator initialized")
        print(f"   Dataset: {len(self.frames)} frames")
        print(f"   Output: {self.output_dir}")
        print(f"   Shared temp: {self.shared_temp_dir}")

    def _load_frame_list(self) -> List[str]:
        """Load list of EHF frame IDs"""
        frames = []
        for img_file in self.ehf_path.glob("*_img.jpg"):
            frame_id = img_file.stem.replace("_img", "")
            frames.append(frame_id)
        frames.sort()
        return frames
    
    def _load_camera_params(self) -> Dict:
        """Load EHF camera parameters"""
        camera_file = self.ehf_path / "EHF_camera.txt"
        if not camera_file.exists():
            raise FileNotFoundError(f"Camera file not found: {camera_file}")
        
        camera_params = {
            'focal': [1498.22426237, 1498.22426237],
            'princpt': [790.263706, 578.90334]
        }
        return camera_params

    def setup_config(self):
        """Setup config"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        new_config = {
            "log": {
                'exp_name': f'ehf_compatible_opt_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'output_dir': str(self.output_dir),
                'model_dir': str(self.output_dir / 'model_dump'),
                'log_dir': str(log_dir),
                'result_dir': str(self.output_dir / 'result'),
            },
            "model": {
                "pretrained_model_path": "pretrained_models/smplest_x/smplest_x_h.pth.tar",
            }
        }
        self.config.update_config(new_config)

    def setup_models(self):
        """Initialize models for metrics calculation"""
        self.smpl = SMPL(self.config.model.human_model_path)
        self.smplx = SMPLX(self.config.model.human_model_path)

    def get_system_info(self):
        """Get system resource info"""
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / 1024**3
        return f"GPU: {gpu_mem:.1f}GB, CPU: {cpu_count} cores, RAM: {ram_gb:.1f}GB"

    def cleanup_previous_runs(self):
        """Clean up GPU memory and temp files from previous runs"""
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up any leftover temp directories in shared temp
        for temp_dir in self.shared_temp_dir.glob("temp_*"):
            if temp_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass

    def run_optimized_single_frame_pipeline(self, frame_data: Dict) -> Tuple[Dict, Dict]:
        """YOUR EXACT WORKING PIPELINE - unchanged"""
        frame_id = frame_data['frame_id']
        img_path = frame_data['img_path']
        
        # Create frame-specific gallery
        frame_gallery = self.central_gallery / frame_id
        frame_gallery.mkdir(exist_ok=True)
        
        # Use shared temp directory to reduce I/O
        temp_dir = self.shared_temp_dir / f"temp_{frame_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Clean up before starting
            self.cleanup_previous_runs()
            
            # Step 1: Run main pipeline with monitoring
            print(f"      🚀 Running main pipeline...")
            start_time = pytime.time()
            
            main_cmd = [
                sys.executable, "main.py",
                "--input_image", str(img_path),
                "--output_dir", str(temp_dir)
            ]
            
            # Monitor subprocess
            result = subprocess.run(main_cmd, capture_output=True, text=True, 
                                  cwd=os.getcwd(), timeout=300)  # 5 min timeout
            
            if result.returncode != 0:
                print(f"      ❌ Main pipeline failed: {result.stderr}")
                return None, None
            
            main_time = pytime.time() - start_time
            self.timing_stats['main_pipeline'].append(main_time)
            print(f"      ⏱️ Main pipeline: {main_time:.1f}s")
            
            # Find run directory
            run_dirs = list(temp_dir.glob("run_*"))
            if not run_dirs:
                print(f"      ❌ No run directory found")
                return None, None
            
            run_dir = run_dirs[0]
            
            # Load baseline results immediately
            baseline_result = self._load_baseline_from_pipeline(run_dir)
            if baseline_result is None:
                print(f"      ❌ Failed to load baseline results")
                return None, None
            
            # Step 2: Quick coordinate analysis
            print(f"      📐 Running coordinate analysis...")
            start_time = pytime.time()
            
            coord_cmd = [
                sys.executable, "analysis_tools/coordinate_analyzer_fixed.py",
                str(run_dir)
            ]
            
            try:
                result = subprocess.run(coord_cmd, capture_output=True, text=True, 
                                      cwd=os.getcwd(), timeout=45)
                coord_time = pytime.time() - start_time
                self.timing_stats['coordinate_analysis'].append(coord_time)
                print(f"      ⏱️ Coordinate analysis: {coord_time:.1f}s")
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Coordinate analysis timeout")
                coord_time = 45
                self.timing_stats['coordinate_analysis'].append(coord_time)
            
            # Step 3: Fusion with timeout
            print(f"      🔄 Running fusion...")
            start_time = pytime.time()
            
            fusion_cmd = [
                sys.executable, "fusion/direct_parameter_fusion.py",
                "--results_dir", str(run_dir),
                "--gallery_dir", str(frame_gallery)
            ]
            
            fusion_success = False
            try:
                result = subprocess.run(fusion_cmd, capture_output=True, text=True, 
                                      cwd=os.getcwd(), timeout=90)
                
                if result.returncode == 0:
                    fusion_success = True
                    fusion_time = pytime.time() - start_time
                    self.timing_stats['fusion'].append(fusion_time)
                    print(f"      ⏱️ Fusion: {fusion_time:.1f}s")
                else:
                    print(f"      ❌ Fusion failed: {result.stderr}")
                    fusion_time = pytime.time() - start_time
                    self.timing_stats['fusion'].append(fusion_time)
                
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Fusion timeout")
                fusion_time = 90
                self.timing_stats['fusion'].append(fusion_time)
            
            # Step 4: Quick visualization (optional)
            print(f"      📊 Running visualization...")
            start_time = pytime.time()
            
            viz_cmd = [
                sys.executable, "fusion/enhanced_fusion_visualizer.py", 
                "--results_dir", str(run_dir),
                "--gallery_dir", str(frame_gallery)
            ]
            
            try:
                result = subprocess.run(viz_cmd, capture_output=True, text=True, 
                                      cwd=os.getcwd(), timeout=60)
                viz_time = pytime.time() - start_time
                self.timing_stats['visualization'].append(viz_time)
                print(f"      ⏱️ Visualization: {viz_time:.1f}s")
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Visualization timeout")
                viz_time = 60
                self.timing_stats['visualization'].append(viz_time)
            
            # Load fusion results
            if fusion_success:
                fusion_result = self._load_fusion_from_pipeline(run_dir, baseline_result)
            else:
                fusion_result = baseline_result
            
            return baseline_result, fusion_result
            
        except subprocess.TimeoutExpired:
            print(f"      ⏰ Pipeline timeout")
            return None, None
        except Exception as e:
            print(f"      ❌ Pipeline error: {e}")
            return None, None
        finally:
            # Clean up temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

    def _load_baseline_from_pipeline(self, run_dir: Path) -> Optional[Dict]:
        """YOUR EXACT WORKING BASELINE LOADER"""
        smplx_files = list(run_dir.glob("smplestx_results/*/person_*/smplx_params_*.json"))
        if not smplx_files:
            return None
        
        with open(smplx_files[0], 'r') as f:
            params = json.load(f)
        
        result = {
            'mesh': np.array(params['mesh']),
            'joints_3d': np.array(params['joints_3d']),
            'joints_2d': np.array(params['joints_2d']),
            'parameters': {k: np.array(v) for k, v in params.items() 
                          if k not in ['mesh', 'joints_3d', 'joints_2d']}
        }
        return result

    def _load_fusion_from_pipeline(self, run_dir: Path, baseline_result: Dict) -> Dict:
        """YOUR EXACT WORKING FUSION LOADER"""
        fusion_dir = run_dir / "fusion_results"
        if not (fusion_dir / "enhanced_mesh.npy").exists():
            return baseline_result
        
        enhanced_mesh = np.load(fusion_dir / "enhanced_mesh.npy")
        fusion_result = baseline_result.copy()
        fusion_result['mesh'] = enhanced_mesh
        fusion_result['fusion_status'] = 'success'
        
        return fusion_result

    def get_frame_data(self, frame_id: str) -> Dict:
        """YOUR EXACT WORKING FRAME DATA LOADER - with joints_2d_path added"""
        return {
            'frame_id': frame_id,
            'img_path': self.ehf_path / f"{frame_id}_img.jpg",
            'align_path': self.ehf_path / f"{frame_id}_align.ply",
            'joints_2d_path': self.ehf_path / f"{frame_id}_2Djnt.json",  # Added for metrics
            'camera_params': self.camera_params
        }
    
    def load_ground_truth_mesh(self, frame_id: str) -> np.ndarray:
        """YOUR EXACT WORKING GT MESH LOADER"""
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        mesh = trimesh.load(align_path)
        return np.array(mesh.vertices)

    def calculate_metrics(self, baseline_result: Dict, fusion_result: Dict, frame_data: Dict) -> Tuple[Dict, Dict]:
        """
        ONLY FUNCTION CHANGED: Literature metrics instead of PVE/PA-PVE
        Uses your existing joints_2d data with proper literature metrics
        """
        try:
            # Load EHF ground truth
            gt_data = self.lit_metrics.load_ehf_ground_truth(frame_data['joints_2d_path'])
            
            # Calculate literature metrics using your existing joints_2d data
            baseline_metrics = self.lit_metrics.calculate_literature_metrics(
                baseline_result['joints_2d'], gt_data
            )
            
            fusion_metrics = self.lit_metrics.calculate_literature_metrics(
                fusion_result['joints_2d'], gt_data
            )
            
            return baseline_metrics, fusion_metrics
            
        except Exception as e:
            print(f"      ⚠️ Literature metrics calculation failed: {e}")
            # Fallback metrics
            fallback_metrics = {
                'MPJPE_2D': float('inf'),
                'Hand_MPJPE': float('inf'),
                'Hand_MPJPE_Left': float('inf'),
                'Hand_MPJPE_Right': float('inf'),
                'PCK@20': 0.0,
                'PCK@50': 0.0,
                'Hand_PCK@20': 0.0,
                'Face_NME': float('inf'),
                'Face_FR': 1.0,
                'face_available': False,
                'calculation_error': str(e)
            }
            return fallback_metrics, fallback_metrics

    def print_performance_stats(self):
        """YOUR EXACT WORKING PERFORMANCE STATS"""
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        
        for step, times in self.timing_stats.items():
            if times:
                avg_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                
                print(f"   {step.replace('_', ' ').title()}:")
                print(f"      Average: {avg_time:.1f}s ± {std_time:.1f}s")
                print(f"      Range: {min_time:.1f}s - {max_time:.1f}s")
        
        # Total time breakdown
        total_avg = sum(np.mean(times) for times in self.timing_stats.values() if times)
        print(f"   Total Average per Frame: {total_avg:.1f}s")
        
        if self.timing_stats['main_pipeline']:
            main_pct = (np.mean(self.timing_stats['main_pipeline']) / total_avg) * 100
            print(f"   Main Pipeline % of Total: {main_pct:.1f}%")

    def run_compatible_evaluation(self, max_frames: Optional[int] = None) -> Dict:
        """YOUR EXACT WORKING EVALUATION LOOP"""
        frames_to_eval = self.frames[:max_frames] if max_frames else self.frames
        
        print(f"🚀 Starting compatible optimized evaluation...")
        print(f"   System: {self.get_system_info()}")
        print(f"   Total frames: {len(frames_to_eval)}")
        print(f"   Metrics: Literature-based (MPJPE-2D, Hand-MPJPE, PCK@k, Face-NME)")
        print(f"   Frames: {frames_to_eval}")
        
        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        fusion_status_summary = {'success': 0, 'failed_or_identical': 0, 'error': 0}
        
        # Process frames sequentially with optimization
        for i, frame_id in enumerate(frames_to_eval):
            print(f"\n   [{i+1}/{len(frames_to_eval)}] Frame: {frame_id}")
            frame_start_time = pytime.time()
            
            try:
                # Get frame data
                frame_data = self.get_frame_data(frame_id)
                
                # Run optimized pipeline
                baseline_result, fusion_result = self.run_optimized_single_frame_pipeline(frame_data)
                
                if baseline_result is None:
                    print(f"      ❌ Pipeline failed for frame {frame_id}")
                    fusion_status_summary['error'] += 1
                    continue
                
                # Calculate literature metrics
                baseline_metrics, fusion_metrics = self.calculate_metrics(baseline_result, fusion_result, frame_data)  
                
                # Determine fusion status
                if not np.array_equal(baseline_result['mesh'], fusion_result['mesh']):
                    fusion_status = 'success'
                    fusion_status_summary['success'] += 1
                else:
                    fusion_status = 'failed_or_identical'
                    fusion_status_summary['failed_or_identical'] += 1
                
                # Store results
                result = {
                    'frame_id': frame_id,
                    'baseline_metrics': baseline_metrics,
                    'fusion_metrics': fusion_metrics,
                    'fusion_status': fusion_status,
                    'gallery_path': str(self.central_gallery / frame_id),
                    'visualization_count': len(list((self.central_gallery / frame_id).glob("*")))
                }
                
                all_results.append(result)
                baseline_metrics_all.append(baseline_metrics)
                fusion_metrics_all.append(fusion_metrics)
                
                # Print frame summary
                frame_time = pytime.time() - frame_start_time
                print(f"      ⏱️ Frame completed in {frame_time:.1f}s")
                
                # Print key metric improvements
                if baseline_metrics.get('Hand_MPJPE', float('inf')) != float('inf') and fusion_metrics.get('Hand_MPJPE', float('inf')) != float('inf'):
                    hand_improvement = ((baseline_metrics['Hand_MPJPE'] - fusion_metrics['Hand_MPJPE']) / baseline_metrics['Hand_MPJPE']) * 100
                    print(f"      📊 Hand-MPJPE improvement: {hand_improvement:.2f}%")
                
            except Exception as e:
                print(f"      ❌ Error processing {frame_id}: {e}")
                fusion_status_summary['error'] += 1
                continue
        
        # Print performance analysis
        self.print_performance_stats()
        
        # Aggregate results
        final_results = self.aggregate_results(baseline_metrics_all, fusion_metrics_all)
        final_results['per_frame_results'] = all_results
        final_results['total_frames'] = len(all_results)
        final_results['fusion_status_summary'] = fusion_status_summary
        final_results['central_gallery_path'] = str(self.central_gallery)
        final_results['timing_stats'] = {k: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for k, v in self.timing_stats.items() if v}
        final_results['metrics_used'] = ['MPJPE_2D', 'Hand_MPJPE', 'PCK@20', 'PCK@50', 'Hand_PCK@20', 'Face_NME']
        
        # Save results with proper JSON encoding
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)
        
        with open(self.output_dir / "compatible_evaluation_results.json", 'w') as f:
            json.dump(final_results, f, indent=2, cls=NumpyEncoder)
        
        return final_results

    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        """YOUR EXACT WORKING AGGREGATION"""
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            
            keys = set()
            for m in metrics_list:
                keys.update(m.keys())
            
            averaged = {}
            for key in keys:
                values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float)) and not np.isinf(m[key])]
                if values:
                    averaged[key] = float(np.mean(values))
            return averaged
        
        baseline_avg = average_metrics(baseline_metrics)
        fusion_avg = average_metrics(fusion_metrics)
        
        improvements = {}
        for key in baseline_avg:
            if key in fusion_avg and baseline_avg[key] > 0:
                improvement = ((baseline_avg[key] - fusion_avg[key]) / baseline_avg[key]) * 100
                improvements[f"{key}_improvement_%"] = float(improvement)
        
        return {
            'baseline_average': baseline_avg,
            'fusion_average': fusion_avg,
            'improvements': improvements
        }

def main():
    parser = argparse.ArgumentParser(description='Compatible Optimized EHF Evaluation with Literature Metrics')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='EHF dataset path')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py', help='Config path')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames (0 for all)')
    
    args = parser.parse_args()
    
    max_frames = args.max_frames if args.max_frames > 0 else None
    
    print("🚀 Starting COMPATIBLE EHF Evaluation with Literature Metrics")
    print("   Using your exact working pipeline with only metrics changed")
    print("   Metrics: MPJPE-2D, Hand-MPJPE, PCK@k, Face-NME")
    start_time = pytime.time()
    
    evaluator = CompatibleOptimizedEHFEvaluator(args.ehf_path, args.config)
    results = evaluator.run_compatible_evaluation(max_frames)
    
    total_time = pytime.time() - start_time
    evaluated_frames = results.get('total_frames', 0)
    
    print(f"\n⏱️ FINAL TIMING SUMMARY:")
    print(f"   Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if evaluated_frames > 0:
        print(f"   Average per frame: {total_time/evaluated_frames:.1f}s")
    print(f"   Frames processed: {evaluated_frames}")
    
    print(f"\n✅ Compatible evaluation complete!")
    print(f"🎨 Gallery: {evaluator.central_gallery}")

if __name__ == '__main__':
    main()