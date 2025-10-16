#!/usr/bin/env python3
"""
Enhanced EHF Evaluator with V2V and PA-V2V Metrics - Based on Your Working Evaluator
ONLY replaces calculate_metrics function and related metric calculations.
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

class MeshMetricsCalculator:
    """Mesh-based metrics: Vertex-to-Vertex (V2V) and Procrustes Analysis V2V (PA-V2V)"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print(f"✅ Mesh metrics calculator initialized")
            print(f"   Using metrics: V2V (Vertex-to-Vertex) and PA-V2V (Procrustes Analysis V2V)")

    def procrustes_align(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Procrustes analysis (rigid alignment: translation + rotation)
        to align source_points to target_points.

        Args:
            source_points (np.ndarray): Nx3 array of source points (e.g., predicted mesh vertices).
            target_points (np.ndarray): Nx3 array of target points (e.g., ground truth mesh vertices).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - aligned_source (np.ndarray): Source points after optimal rotation and translation.
                - rotation_matrix (np.ndarray): 3x3 rotation matrix.
                - translation_vector (np.ndarray): 3-element translation vector.
        """
        if source_points.shape != target_points.shape or source_points.shape[1] != 3:
            raise ValueError("Input point sets must have the same Nx3 shape.")

        # 1. Center the point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        centered_source = source_points - source_centroid
        centered_target = target_points - target_centroid

        # 2. Compute optimal rotation (Kabsch algorithm via SVD)
        H = centered_source.T @ centered_target  # Covariance matrix
        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = Vt.T @ U.T

        # Handle reflection case (if the determinant is -1)
        # This typically implies a reflection, which rigid transformations don't allow.
        # If true scale is not desired (only rigid), flip the smallest singular vector.
        if np.linalg.det(rotation_matrix) < 0:
            Vt[2, :] *= -1
            rotation_matrix = Vt.T @ U.T

        # 3. Compute optimal translation
        # translation_vector = target_centroid - (rotation_matrix @ source_centroid)
        # After rotating centered_source, we need to translate it to target_centroid
        # rotated_centered_source = centered_source @ rotation_matrix.T
        # aligned_source_final = rotated_centered_source + target_centroid

        # Align centered source, then translate
        aligned_source = (rotation_matrix @ centered_source.T).T + target_centroid
        translation_vector = target_centroid - (rotation_matrix @ source_centroid)


        return aligned_source, rotation_matrix, translation_vector

    def calculate_v2v(self, pred_mesh_vertices: np.ndarray, gt_mesh_vertices: np.ndarray) -> float:
        """
        Calculates the mean Vertex-to-Vertex (V2V) distance between two meshes.
        Assumes direct correspondence and no prior alignment.

        Args:
            pred_mesh_vertices (np.ndarray): Nx3 array of predicted mesh vertices.
            gt_mesh_vertices (np.ndarray): Nx3 array of ground truth mesh vertices.

        Returns:
            float: Mean V2V distance in millimeters (assuming input is in mm or consistent units).
                  Returns NaN if inputs are invalid.
        """
        try:
            if pred_mesh_vertices.shape != gt_mesh_vertices.shape or pred_mesh_vertices.shape[1] != 3:
                if self.verbose:
                    print(f"⚠️ V2V input shape mismatch or invalid: pred {pred_mesh_vertices.shape}, gt {gt_mesh_vertices.shape}")
                return float('nan')

            distances = np.linalg.norm(pred_mesh_vertices - gt_mesh_vertices, axis=1)
            return float(np.mean(distances))
        except Exception as e:
            if self.verbose:
                print(f"⚠️ V2V calculation failed: {e}")
            return float('nan')

    def calculate_pa_v2v(self, pred_mesh_vertices: np.ndarray, gt_mesh_vertices: np.ndarray) -> float:
        """
        Calculates the mean Procrustes Analysis Vertex-to-Vertex (PA-V2V) distance.
        Aligns pred_mesh_vertices to gt_mesh_vertices using rigid Procrustes alignment
        (translation + rotation) before calculating distances.

        Args:
            pred_mesh_vertices (np.ndarray): Nx3 array of predicted mesh vertices.
            gt_mesh_vertices (np.ndarray): Nx3 array of ground truth mesh vertices.

        Returns:
            float: Mean PA-V2V distance in millimeters (assuming consistent units).
                  Returns NaN if inputs are invalid or alignment fails.
        """
        try:
            if pred_mesh_vertices.shape != gt_mesh_vertices.shape or pred_mesh_vertices.shape[1] != 3:
                if self.verbose:
                    print(f"⚠️ PA-V2V input shape mismatch or invalid: pred {pred_mesh_vertices.shape}, gt {gt_mesh_vertices.shape}")
                return float('nan')

            # Align predicted mesh to ground truth mesh
            aligned_pred_vertices, _, _ = self.procrustes_align(pred_mesh_vertices, gt_mesh_vertices)

            # Calculate V2V distance on the aligned meshes
            return self.calculate_v2v(aligned_pred_vertices, gt_mesh_vertices)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ PA-V2V calculation failed: {e}")
            return float('nan')

    def calculate_mesh_metrics(self, predicted_mesh_vertices: np.ndarray, gt_mesh_vertices: np.ndarray) -> Dict:
        """
        Calculate all mesh-based metrics.
        """
        results = {}
        results['V2V'] = self.calculate_v2v(predicted_mesh_vertices, gt_mesh_vertices)
        results['PA_V2V'] = self.calculate_pa_v2v(predicted_mesh_vertices, gt_mesh_vertices)

        # Ensure no inf values are returned
        return {k: v if not np.isinf(v) else float('nan') for k, v in results.items()}


class CompatibleOptimizedEHFEvaluator:
    """YOUR EXACT WORKING EVALUATOR with only calculate_metrics replaced"""

    def __init__(self, ehf_path: str = "data/EHF",
                 smplestx_config: str = "pretrained_models/smplest_x/config_base.py",
                 verbose_output: bool = False):
        self.ehf_path = Path(ehf_path)
        self.config = Config.load_config(smplestx_config)
        self.verbose_output = verbose_output

        # Initialize mesh metrics calculator
        self.mesh_metrics = MeshMetricsCalculator(verbose=verbose_output)

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

        if self.verbose_output:
            print(f"✅ Compatible Optimized EHF Evaluator initialized")
            print(f"   Dataset: {len(self.frames)} frames")
            print(f"   Output: {self.output_dir}")
            print(f"   Shared temp: {self.shared_temp_dir}")

    def _load_frame_list(self) -> List[str]:
        """Load list of EHF frame IDs, ensuring _align.ply exists for 3D GT"""
        frames = []
        for img_file in self.ehf_path.glob("*_img.jpg"):
            frame_id = img_file.stem.replace("_img", "")
            # Ensure ground truth mesh exists for V2V/PA-V2V
            if (self.ehf_path / f"{frame_id}_align.ply").exists():
                frames.append(frame_id)
            else:
                if self.verbose_output:
                    print(f"Skipping {frame_id}: {frame_id}_align.ply (ground truth mesh) not found.")
        frames.sort()
        return frames

    def _load_camera_params(self) -> Dict:
        """Load EHF camera parameters"""
        camera_file = self.ehf_path / "EHF_camera.txt"
        if not camera_file.exists():
            if self.verbose_output:
                print(f"⚠️ Camera file not found: {camera_file}, using default parameters.")
            # Default parameters are already in the original code, keep them here
            return {
                'focal': [1498.22426237, 1498.22426237],
                'princpt': [790.263706, 578.90334]
            }
        
        # Original code didn't actually load from file, just hardcoded.
        # If the file exists but isn't read, this behavior is unchanged.
        # If you intended to read from it, uncomment and implement file reading.
        # For now, keeping the original hardcoded values.
        return {
            'focal': [1498.22426237, 1498.22426237],
            'princpt': [790.263706, 578.90334]
        }

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
        # These are used for potential 3D metrics in the original full evaluator,
        # but for 2D literature metrics, they might not be strictly necessary
        # unless some internal calculation within SMPLest-X requires them.
        # Keeping them for compatibility with the "working pipeline intact" goal.
        self.smpl = SMPL(self.config.model.human_model_path)
        self.smplx = SMPLX(self.config.model.human_model_path)

    def get_system_info(self):
        """Get system resource info"""
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
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
                except Exception as e:
                    if self.verbose_output:
                        print(f"⚠️ Failed to remove temp dir {temp_dir}: {e}")

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
            if self.verbose_output:
                print(f"      🚀 Running main pipeline for {frame_id}...")
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
                if self.verbose_output:
                    print(f"      ❌ Main pipeline failed for {frame_id}: {result.stderr}")
                return None, None

            main_time = pytime.time() - start_time
            self.timing_stats['main_pipeline'].append(main_time)
            if self.verbose_output:
                print(f"      ⏱️ Main pipeline for {frame_id}: {main_time:.1f}s")

            # Find run directory
            run_dirs = list(temp_dir.glob("run_*"))
            if not run_dirs:
                if self.verbose_output:
                    print(f"      ❌ No run directory found for {frame_id}")
                return None, None

            run_dir = run_dirs[0]

            # Load baseline results immediately
            baseline_result = self._load_baseline_from_pipeline(run_dir)
            if baseline_result is None:
                if self.verbose_output:
                    print(f"      ❌ Failed to load baseline results for {frame_id}")
                return None, None

            # Step 2: Quick coordinate analysis
            if self.verbose_output:
                print(f"      📐 Running coordinate analysis for {frame_id}...")
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
                if self.verbose_output:
                    print(f"      ⏱️ Coordinate analysis for {frame_id}: {coord_time:.1f}s")
            except subprocess.TimeoutExpired:
                if self.verbose_output:
                    print(f"      ⏰ Coordinate analysis timeout for {frame_id}")
                coord_time = 45
                self.timing_stats['coordinate_analysis'].append(coord_time)
            except Exception as e:
                if self.verbose_output:
                    print(f"      ❌ Coordinate analysis error for {frame_id}: {e}")
                self.timing_stats['coordinate_analysis'].append(45) # Assume timeout/failure duration

            # Step 3: Fusion with timeout
            if self.verbose_output:
                print(f"      🔄 Running fusion for {frame_id}...")
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
                    if self.verbose_output:
                        print(f"      ⏱️ Fusion for {frame_id}: {fusion_time:.1f}s")
                else:
                    if self.verbose_output:
                        print(f"      ❌ Fusion failed for {frame_id}: {result.stderr}")
                    fusion_time = pytime.time() - start_time
                    self.timing_stats['fusion'].append(fusion_time)

            except subprocess.TimeoutExpired:
                if self.verbose_output:
                    print(f"      ⏰ Fusion timeout for {frame_id}")
                fusion_time = 90
                self.timing_stats['fusion'].append(fusion_time)
            except Exception as e:
                if self.verbose_output:
                    print(f"      ❌ Fusion error for {frame_id}: {e}")
                self.timing_stats['fusion'].append(90) # Assume timeout/failure duration


            # Step 4: Quick visualization (optional)
            if self.verbose_output:
                print(f"      📊 Running visualization for {frame_id}...")
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
                if self.verbose_output:
                    print(f"      ⏱️ Visualization for {frame_id}: {viz_time:.1f}s")
            except subprocess.TimeoutExpired:
                if self.verbose_output:
                    print(f"      ⏰ Visualization timeout for {frame_id}")
                viz_time = 60
                self.timing_stats['visualization'].append(viz_time)
            except Exception as e:
                if self.verbose_output:
                    print(f"      ❌ Visualization error for {frame_id}: {e}")
                self.timing_stats['visualization'].append(60) # Assume timeout/failure duration

            # Load fusion results
            if fusion_success:
                fusion_result = self._load_fusion_from_pipeline(run_dir, baseline_result)
            else:
                fusion_result = baseline_result

            return baseline_result, fusion_result

        except subprocess.TimeoutExpired:
            if self.verbose_output:
                print(f"      ⏰ Pipeline timeout for {frame_id}")
            return None, None
        except Exception as e:
            if self.verbose_output:
                print(f"      ❌ Pipeline error for {frame_id}: {e}")
            return None, None
        finally:
            # Clean up temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                if self.verbose_output:
                    print(f"⚠️ Failed to remove temp dir {temp_dir}: {e}")

    def _load_baseline_from_pipeline(self, run_dir: Path) -> Optional[Dict]:
        """YOUR EXACT WORKING BASELINE LOADER, ensuring 'mesh' is present"""
        smplx_files = list(run_dir.glob("smplestx_results/*/person_*/smplx_params_*.json"))
        if not smplx_files:
            return None

        with open(smplx_files[0], 'r') as f:
            params = json.load(f)

        # Ensure mesh vertices are loaded as numpy array
        if 'mesh' not in params or not isinstance(params['mesh'], list):
            if self.verbose_output:
                print(f"      ⚠️ 'mesh' data missing or invalid in baseline results for {run_dir}. Skipping frame.")
            return None

        result = {
            'mesh': np.array(params['mesh']),
            'joints_3d': np.array(params['joints_3d']),
            'joints_2d': np.array(params['joints_2d']),
            'parameters': {k: np.array(v) for k, v in params.items()
                          if k not in ['mesh', 'joints_3d', 'joints_2d']}
        }
        return result

    def _load_fusion_from_pipeline(self, run_dir: Path, baseline_result: Dict) -> Dict:
        """YOUR EXACT WORKING FUSION LOADER, ensuring 'mesh' is present"""
        fusion_dir = run_dir / "fusion_results"
        enhanced_mesh_path = fusion_dir / "enhanced_mesh.npy"

        if not enhanced_mesh_path.exists():
            return baseline_result # If fusion mesh not found, return baseline

        try:
            enhanced_mesh = np.load(enhanced_mesh_path)
            fusion_result = baseline_result.copy()
            fusion_result['mesh'] = enhanced_mesh
            fusion_result['fusion_status'] = 'success'
            return fusion_result
        except Exception as e:
            if self.verbose_output:
                print(f"      ⚠️ Failed to load enhanced_mesh.npy for fusion results in {run_dir}: {e}. Returning baseline.")
            return baseline_result


    def get_frame_data(self, frame_id: str) -> Dict:
        """YOUR EXACT WORKING FRAME DATA LOADER - with align_path for 3D GT"""
        return {
            'frame_id': frame_id,
            'img_path': self.ehf_path / f"{frame_id}_img.jpg",
            'align_path': self.ehf_path / f"{frame_id}_align.ply", # Used for 3D GT mesh
            'joints_2d_path': self.ehf_path / f"{frame_id}_2Djnt.json",
            'camera_params': self.camera_params
        }

    def load_ground_truth_mesh(self, frame_id: str) -> Optional[np.ndarray]:
        """YOUR EXACT WORKING GT MESH LOADER, added error handling"""
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        if not align_path.exists():
            if self.verbose_output:
                print(f"      ⚠️ Ground truth mesh not found for {frame_id}: {align_path}")
            return None
        try:
            mesh = trimesh.load(align_path)
            return np.array(mesh.vertices)
        except Exception as e:
            if self.verbose_output:
                print(f"      ⚠️ Failed to load ground truth mesh from {align_path}: {e}")
            return None

    def calculate_metrics(self, baseline_result: Dict, fusion_result: Dict, frame_data: Dict) -> Tuple[Dict, Dict]:
        """
        Calculates V2V and PA-V2V metrics for baseline and fusion results.
        """
        try:
            gt_mesh_vertices = self.load_ground_truth_mesh(frame_data['frame_id'])
            if gt_mesh_vertices is None:
                # Return NaN for all metrics if GT mesh is missing or invalid
                fallback_metrics = {'V2V': float('nan'), 'PA_V2V': float('nan')}
                return fallback_metrics, fallback_metrics

            baseline_metrics = self.mesh_metrics.calculate_mesh_metrics(
                baseline_result['mesh'], gt_mesh_vertices
            )
            fusion_metrics = self.mesh_metrics.calculate_mesh_metrics(
                fusion_result['mesh'], gt_mesh_vertices
            )

            return baseline_metrics, fusion_metrics

        except Exception as e:
            if self.verbose_output:
                print(f"      ⚠️ Mesh metrics calculation failed for frame {frame_data['frame_id']}: {e}")
            fallback_metrics = {'V2V': float('nan'), 'PA_V2V': float('nan')}
            return fallback_metrics, fallback_metrics


    def print_performance_stats(self):
        """YOUR EXACT WORKING PERFORMANCE STATS, adapted for verbosity"""
        if not self.verbose_output:
            return

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

        if self.verbose_output:
            print(f"🚀 Starting compatible optimized evaluation...")
            print(f"   System: {self.get_system_info()}")
            print(f"   Total frames: {len(frames_to_eval)}")
            print(f"   Metrics: V2V, PA-V2V")
            print(f"   Frames to process: {len(frames_to_eval)}")

        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        fusion_status_summary = {'success': 0, 'failed_or_identical': 0, 'error': 0}

        # Process frames sequentially with optimization
        for i, frame_id in enumerate(tqdm(frames_to_eval, desc="Processing Frames", disable=not self.verbose_output)):
            if self.verbose_output:
                print(f"\n   [{i+1}/{len(frames_to_eval)}] Frame: {frame_id}")
            frame_start_time = pytime.time()

            try:
                # Get frame data
                frame_data = self.get_frame_data(frame_id)

                # Run optimized pipeline
                baseline_result, fusion_result = self.run_optimized_single_frame_pipeline(frame_data)

                if baseline_result is None:
                    if self.verbose_output:
                        print(f"      ❌ Pipeline failed for frame {frame_id}")
                    fusion_status_summary['error'] += 1
                    continue

                # Calculate mesh metrics
                baseline_metrics, fusion_metrics = self.calculate_metrics(baseline_result, fusion_result, frame_data)

                # Determine fusion status
                fusion_status = 'failed_or_identical'
                if baseline_result is not None and fusion_result is not None:
                    # Check if fusion mesh is different from baseline mesh to determine 'success'
                    if 'mesh' in baseline_result and 'mesh' in fusion_result:
                        # Use allclose for floating point array comparison
                        if not np.allclose(baseline_result['mesh'], fusion_result['mesh'], atol=1e-6):
                             fusion_status = 'success'
                
                if fusion_status == 'success':
                    fusion_status_summary['success'] += 1
                else:
                    fusion_status_summary['failed_or_identical'] += 1


                # Store results
                result = {
                    'frame_id': frame_id,
                    'baseline_metrics': baseline_metrics,
                    'fusion_metrics': fusion_metrics,
                    'fusion_status': fusion_status,
                }
                if self.verbose_output:
                    result['gallery_path'] = str(self.central_gallery / frame_id)
                    result['visualization_count'] = len(list((self.central_gallery / frame_id).glob("*")))

                all_results.append(result)
                baseline_metrics_all.append(baseline_metrics)
                fusion_metrics_all.append(fusion_metrics)

                # Print frame summary
                frame_time = pytime.time() - frame_start_time
                if self.verbose_output:
                    print(f"      ⏱️ Frame {frame_id} completed in {frame_time:.1f}s")

                    # Print key metric improvements
                    # Only print if valid numbers are available for comparison
                    if not np.isnan(baseline_metrics.get('V2V', np.nan)) and \
                       not np.isnan(fusion_metrics.get('V2V', np.nan)) and \
                       baseline_metrics['V2V'] > 0: # Avoid division by zero
                        v2v_improvement = ((baseline_metrics['V2V'] - fusion_metrics['V2V']) / baseline_metrics['V2V']) * 100
                        print(f"      📊 V2V improvement: {v2v_improvement:.2f}%")
                    if not np.isnan(baseline_metrics.get('PA_V2V', np.nan)) and \
                       not np.isnan(fusion_metrics.get('PA_V2V', np.nan)) and \
                       baseline_metrics['PA_V2V'] > 0:
                        pa_v2v_improvement = ((baseline_metrics['PA_V2V'] - fusion_metrics['PA_V2V']) / baseline_metrics['PA_V2V']) * 100
                        print(f"      📊 PA-V2V improvement: {pa_v2v_improvement:.2f}%")


            except Exception as e:
                if self.verbose_output:
                    print(f"      ❌ Error processing {frame_id}: {e}")
                fusion_status_summary['error'] += 1
                continue

        # Print performance analysis
        self.print_performance_stats()

        # Aggregate results
        final_results_dict = self.aggregate_results(baseline_metrics_all, fusion_metrics_all)

        # Structure for clean JSON output
        output_data = {
            'comparison_summary': {
                'metrics_used': ['V2V', 'PA_V2V'],
                'baseline_average_metrics': final_results_dict['baseline_average'],
                'fusion_average_metrics': final_results_dict['fusion_average'],
                'improvements_percentage': final_results_dict['improvements'],
                'fusion_status_summary': fusion_status_summary
            },
            'metadata': {
                'total_evaluation_time_s': float('nan'), # Will be filled after loop
                'evaluated_frames_count': len(all_results),
                'output_directory': str(self.output_dir)
            }
        }

        if self.verbose_output:
            output_data['detailed_timing_stats'] = {k: {'mean': float(np.mean(v)) if v else float('nan'),
                                                        'std': float(np.std(v)) if v else float('nan')}
                                                    for k, v in self.timing_stats.items()}
            output_data['per_frame_detailed_results'] = all_results

        # Save results with proper JSON encoding
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    # Handle NaN directly for JSON
                    if np.isnan(obj):
                        return None # JSON null for NaN
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super(NumpyEncoder, self).default(obj)

        output_json_path = self.output_dir / "evaluation_comparison_results.json"
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)

        return output_data # Return the structured dictionary

    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        """Aggregates mesh metrics and calculates improvements."""
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}

            keys = set()
            for m in metrics_list:
                keys.update(m.keys())

            averaged = {}
            for key in keys:
                # Filter out NaNs before averaging
                values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float)) and not np.isnan(m[key])]
                if values:
                    averaged[key] = float(np.mean(values))
                else:
                    averaged[key] = float('nan') # If no valid values, average is NaN
            return averaged

        baseline_avg = average_metrics(baseline_metrics)
        fusion_avg = average_metrics(fusion_metrics)

        improvements = {}
        # Assuming V2V and PA_V2V are "lower is better" metrics
        for key in ['V2V', 'PA_V2V']:
            if key in baseline_avg and key in fusion_avg and \
               not np.isnan(baseline_avg[key]) and not np.isnan(fusion_avg[key]):
                if baseline_avg[key] > 0: # Avoid division by zero
                    reduction = ((baseline_avg[key] - fusion_avg[key]) / baseline_avg[key]) * 100
                    improvements[f"{key}_reduction_%"] = float(reduction)
                else: # Baseline was 0 or invalid, no meaningful percentage reduction
                    improvements[f"{key}_reduction_%"] = float('nan')
            else:
                improvements[f"{key}_reduction_%"] = float('nan')

        return {
            'baseline_average': baseline_avg,
            'fusion_average': fusion_avg,
            'improvements': improvements
        }

def main():
    parser = argparse.ArgumentParser(description='Compatible Optimized EHF Evaluation with V2V and PA-V2V Metrics')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='EHF dataset path')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py', help='Config path')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames (0 for all)')
    parser.add_argument('--verbose_output', action='store_true', help='Enable verbose console and JSON output')

    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames > 0 else None

    if args.verbose_output:
        print("🚀 Starting COMPATIBLE EHF Evaluation with V2V and PA-V2V Metrics")
        print("   Using your exact working pipeline with only metrics changed")
        print("   Metrics: V2V, PA-V2V")
    else:
        print("🚀 Starting COMPATIBLE EHF Evaluation (concise output)")


    start_time = pytime.time()

    evaluator = CompatibleOptimizedEHFEvaluator(args.ehf_path, args.config, verbose_output=args.verbose_output)
    results = evaluator.run_compatible_evaluation(max_frames)

    total_time = pytime.time() - start_time
    evaluated_frames = results['metadata'].get('evaluated_frames_count', 0)
    results['metadata']['total_evaluation_time_s'] = total_time # Update total time in results dict

    print(f"\n✅ Compatible evaluation complete!")
    print(f"   Results saved to: {evaluator.output_dir / 'evaluation_comparison_results.json'}")
    print(f"   Gallery: {evaluator.central_gallery}")
    if args.verbose_output:
        print(f"\n⏱️ FINAL TIMING SUMMARY:")
        print(f"   Total evaluation time: {total_time:.1f}s ({total_time/60:.1f} min)")
        if evaluated_frames > 0:
            print(f"   Average per frame: {total_time/evaluated_frames:.1f}s")
        print(f"   Frames processed: {evaluated_frames}")

    # Display the summary directly to console as well
    print("\n--- Evaluation Comparison Summary ---")
    summary = results['comparison_summary']
    print(f"Metrics Used: {', '.join(summary['metrics_used'])}")
    print("\nAverage Baseline Metrics:")
    for k, v in summary['baseline_average_metrics'].items():
        print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: N/A")
    print("\nAverage Fusion Metrics:")
    for k, v in summary['fusion_average_metrics'].items():
        print(f"  {k}: {v:.3f}" if not np.isnan(v) else f"  {k}: N/A")
    print("\nImprovements (%):")
    for k, v in summary['improvements_percentage'].items():
        print(f"  {k}: {v:.2f}%" if not np.isnan(v) else f"  {k}: N/A")
    print(f"\nFusion Status (Success/Identical/Error): {summary['fusion_status_summary']['success']}/"
          f"{summary['fusion_status_summary']['failed_or_identical']}/"
          f"{summary['fusion_status_summary']['error']}")
    print("-------------------------------------")


if __name__ == '__main__':
    main()