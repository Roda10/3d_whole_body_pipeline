#!/usr/bin/env python3
"""
Enhanced EHF Evaluator with V2V and PA-V2V Metrics
Clean version with simplified structure and improved readability.
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import trimesh
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime
from tqdm import tqdm
import psutil
from importlib import import_module

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
smplestx_path = os.path.join(project_root, 'external', 'body', 'SMPLest-X')

sys.path.insert(0, smplestx_path)
sys.path.insert(0, project_root)

# Import SMPLest-X modules
try:
    import importlib.util
    
    config_path = os.path.join(smplestx_path, 'main', 'config.py')
    spec = importlib.util.spec_from_file_location("smplestx_config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    Config = config_module.Config
    
    hm_path = os.path.join(smplestx_path, 'human_models', 'human_models.py')
    spec = importlib.util.spec_from_file_location("smplestx_human_models", hm_path)
    hm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hm_module)
    SMPL = hm_module.SMPL
    SMPLX = hm_module.SMPLX
    
    print("✓ Imports successful")
except Exception as e:
    print(f"Error importing SMPLest-X modules: {e}")
    sys.exit(1)


class MeshMetricsCalculator:
    """Calculates mesh-based metrics: Vertex-to-Vertex (V2V) and Procrustes Analysis V2V (PA-V2V)"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        if self.verbose:
            print("✅ Mesh metrics calculator initialized")

    def procrustes_align(self, source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs Procrustes analysis (rigid alignment)"""
        if source_points.shape != target_points.shape or source_points.shape[1] != 3:
            raise ValueError("Input point sets must have the same Nx3 shape")

        # Center point sets
        source_centroid = np.mean(source_points, axis=0)
        target_centroid = np.mean(target_points, axis=0)

        centered_source = source_points - source_centroid
        centered_target = target_points - target_centroid

        # Compute optimal rotation (Kabsch algorithm)
        H = centered_source.T @ centered_target
        U, S, Vt = np.linalg.svd(H)
        rotation_matrix = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(rotation_matrix) < 0:
            Vt[2, :] *= -1
            rotation_matrix = Vt.T @ U.T

        # Align and translate
        aligned_source = (rotation_matrix @ centered_source.T).T + target_centroid
        translation_vector = target_centroid - (rotation_matrix @ source_centroid)

        return aligned_source, rotation_matrix, translation_vector

    def calculate_v2v(self, pred_vertices: np.ndarray, gt_vertices: np.ndarray) -> float:
        """Calculates mean Vertex-to-Vertex distance"""
        try:
            if pred_vertices.shape != gt_vertices.shape or pred_vertices.shape[1] != 3:
                if self.verbose:
                    print(f"⚠️ V2V shape mismatch: pred {pred_vertices.shape}, gt {gt_vertices.shape}")
                return float('nan')

            distances = np.linalg.norm(pred_vertices - gt_vertices, axis=1)
            return float(np.mean(distances))
        except Exception as e:
            if self.verbose:
                print(f"⚠️ V2V calculation failed: {e}")
            return float('nan')

    def calculate_pa_v2v(self, pred_vertices: np.ndarray, gt_vertices: np.ndarray) -> float:
        """Calculates mean Procrustes Analysis V2V distance"""
        try:
            if pred_vertices.shape != gt_vertices.shape or pred_vertices.shape[1] != 3:
                if self.verbose:
                    print(f"⚠️ PA-V2V shape mismatch: pred {pred_vertices.shape}, gt {gt_vertices.shape}")
                return float('nan')

            aligned_pred, _, _ = self.procrustes_align(pred_vertices, gt_vertices)
            return self.calculate_v2v(aligned_pred, gt_vertices)
        except Exception as e:
            if self.verbose:
                print(f"⚠️ PA-V2V calculation failed: {e}")
            return float('nan')

    def calculate_mesh_metrics(self, pred_vertices: np.ndarray, gt_vertices: np.ndarray) -> Dict:
        """Calculate all mesh-based metrics"""
        results = {
            'V2V': self.calculate_v2v(pred_vertices, gt_vertices),
            'PA_V2V': self.calculate_pa_v2v(pred_vertices, gt_vertices)
        }
        return {k: v if not np.isinf(v) else float('nan') for k, v in results.items()}


class EHFEvaluator:
    """Enhanced EHF Evaluator with V2V and PA-V2V Metrics"""

    def __init__(self, ehf_path: str = "data/EHF",
                 smplestx_config: str = "pretrained_models/smplest_x/config_base.py",
                 verbose: bool = False):
        
        self.ehf_path = Path(ehf_path)
        self.config = Config.load_config(smplestx_config)
        self.verbose = verbose
        
        # Initialize components
        self.mesh_metrics = MeshMetricsCalculator(verbose=verbose)
        
        # Setup directories
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path("evaluation_results") / f"ehf_eval_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gallery_dir = self.output_dir / "gallery"
        self.gallery_dir.mkdir(exist_ok=True)
        self.temp_dir = self.output_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)

        # Load data
        self.frames = self._load_frame_list()
        self.camera_params = self._load_camera_params()
        
        # Setup models and config
        self.setup_config()
        self.setup_models()
        
        # Performance tracking
        self.timing_stats = {
            'main_pipeline': [],
            'coordinate_analysis': [],
            'fusion': [],
            'visualization': []
        }

        if self.verbose:
            print(f"✅ EHF Evaluator initialized")
            print(f"   Dataset: {len(self.frames)} frames")
            print(f"   Output: {self.output_dir}")

    def _load_frame_list(self) -> List[str]:
        """Load list of EHF frame IDs with ground truth meshes"""
        frames = []
        for img_file in self.ehf_path.glob("*_img.jpg"):
            frame_id = img_file.stem.replace("_img", "")
            if (self.ehf_path / f"{frame_id}_align.ply").exists():
                frames.append(frame_id)
            elif self.verbose:
                print(f"Skipping {frame_id}: ground truth mesh not found")
        return sorted(frames)

    def _load_camera_params(self) -> Dict:
        """Load EHF camera parameters"""
        camera_file = self.ehf_path / "EHF_camera.txt"
        if camera_file.exists():
            # Implement camera file parsing if needed
            pass
        return {
            'focal': [1498.22426237, 1498.22426237],
            'princpt': [790.263706, 578.90334]
        }

    def setup_config(self):
        """Setup configuration"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        new_config = {
            "log": {
                'exp_name': f'ehf_eval_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
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
        """Initialize models"""
        self.smpl = SMPL(self.config.model.human_model_path)
        self.smplx = SMPLX(self.config.model.human_model_path)

    def get_system_info(self) -> str:
        """Get system resource information"""
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        cpu_count = psutil.cpu_count()
        ram_gb = psutil.virtual_memory().total / 1024**3
        return f"GPU: {gpu_mem:.1f}GB, CPU: {cpu_count} cores, RAM: {ram_gb:.1f}GB"

    def cleanup_resources(self):
        """Clean up GPU memory and temporary files"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for temp_item in self.temp_dir.glob("temp_*"):
            if temp_item.is_dir():
                try:
                    import shutil
                    shutil.rmtree(temp_item)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ Failed to remove {temp_item}: {e}")

    def run_pipeline_for_frame(self, frame_data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Run the complete pipeline for a single frame"""
        frame_id = frame_data['frame_id']
        frame_gallery = self.gallery_dir / frame_id
        frame_gallery.mkdir(exist_ok=True)
        
        temp_frame_dir = self.temp_dir / f"temp_{frame_id}"
        temp_frame_dir.mkdir(exist_ok=True)

        try:
            self.cleanup_resources()

            # Step 1: Main pipeline
            if self.verbose:
                print(f"   🚀 Running main pipeline for {frame_id}")
            
            start_time = time.time()
            main_cmd = [
                sys.executable, "main.py",
                "--input_image", str(frame_data['img_path']),
                "--output_dir", str(temp_frame_dir)
            ]

            result = subprocess.run(main_cmd, capture_output=True, text=True, 
                                  cwd=os.getcwd(), timeout=300)
            
            if result.returncode != 0:
                if self.verbose:
                    print(f"   ❌ Main pipeline failed: {result.stderr}")
                return None, None

            main_time = time.time() - start_time
            self.timing_stats['main_pipeline'].append(main_time)

            # Find run directory and load baseline
            run_dirs = list(temp_frame_dir.glob("run_*"))
            if not run_dirs:
                if self.verbose:
                    print(f"   ❌ No run directory found")
                return None, None

            run_dir = run_dirs[0]
            baseline_result = self._load_baseline_results(run_dir)
            if baseline_result is None:
                return None, None

            # Step 2: Coordinate analysis
            self._run_coordinate_analysis(run_dir, frame_id)

            # Step 3: Fusion
            fusion_result = self._run_fusion(run_dir, frame_gallery, baseline_result, frame_id)

            # Step 4: Visualization
            self._run_visualization(run_dir, frame_gallery, frame_id)

            return baseline_result, fusion_result

        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"   ⏰ Pipeline timeout for {frame_id}")
            return None, None
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Pipeline error: {e}")
            return None, None
        finally:
            # Cleanup temporary directory
            try:
                import shutil
                shutil.rmtree(temp_frame_dir)
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Failed to cleanup {temp_frame_dir}: {e}")

    def _load_baseline_results(self, run_dir: Path) -> Optional[Dict]:
        """Load baseline results from pipeline output"""
        smplx_files = list(run_dir.glob("smplestx_results/*/person_*/smplx_params_*.json"))
        if not smplx_files:
            return None

        with open(smplx_files[0], 'r') as f:
            params = json.load(f)

        if 'mesh' not in params or not isinstance(params['mesh'], list):
            if self.verbose:
                print("   ⚠️ Mesh data missing in baseline results")
            return None

        return {
            'mesh': np.array(params['mesh']),
            'joints_3d': np.array(params['joints_3d']),
            'joints_2d': np.array(params['joints_2d']),
            'parameters': {k: np.array(v) for k, v in params.items()
                          if k not in ['mesh', 'joints_3d', 'joints_2d']}
        }

    def _run_coordinate_analysis(self, run_dir: Path, frame_id: str):
        """Run coordinate analysis"""
        if self.verbose:
            print(f"   📐 Running coordinate analysis for {frame_id}")
        
        start_time = time.time()
        coord_cmd = [sys.executable, "analysis_tools/coordinate_analyzer_fixed.py", str(run_dir)]
        
        try:
            subprocess.run(coord_cmd, capture_output=True, text=True, timeout=45)
            coord_time = time.time() - start_time
            self.timing_stats['coordinate_analysis'].append(coord_time)
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"   ⏰ Coordinate analysis timeout")
            self.timing_stats['coordinate_analysis'].append(45)
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Coordinate analysis error: {e}")
            self.timing_stats['coordinate_analysis'].append(45)

    def _run_fusion(self, run_dir: Path, gallery_dir: Path, baseline_result: Dict, frame_id: str) -> Dict:
        """Run fusion process"""
        if self.verbose:
            print(f"   🔄 Running fusion for {frame_id}")
        
        start_time = time.time()
        fusion_cmd = [
            sys.executable, "fusion/direct_parameter_fusion.py",
            "--results_dir", str(run_dir),
            "--gallery_dir", str(gallery_dir)
        ]

        try:
            result = subprocess.run(fusion_cmd, capture_output=True, text=True, timeout=90)
            
            if result.returncode == 0:
                fusion_time = time.time() - start_time
                self.timing_stats['fusion'].append(fusion_time)
                return self._load_fusion_results(run_dir, baseline_result)
            else:
                if self.verbose:
                    print(f"   ❌ Fusion failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"   ⏰ Fusion timeout")
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Fusion error: {e}")

        return baseline_result

    def _load_fusion_results(self, run_dir: Path, baseline_result: Dict) -> Dict:
        """Load fusion results"""
        fusion_dir = run_dir / "fusion_results"
        enhanced_mesh_path = fusion_dir / "enhanced_mesh.npy"

        if not enhanced_mesh_path.exists():
            return baseline_result

        try:
            enhanced_mesh = np.load(enhanced_mesh_path)
            fusion_result = baseline_result.copy()
            fusion_result['mesh'] = enhanced_mesh
            fusion_result['fusion_status'] = 'success'
            return fusion_result
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Failed to load fusion mesh: {e}")
            return baseline_result

    def _run_visualization(self, run_dir: Path, gallery_dir: Path, frame_id: str):
        """Run visualization"""
        if self.verbose:
            print(f"   📊 Running visualization for {frame_id}")
        
        start_time = time.time()
        viz_cmd = [
            sys.executable, "fusion/enhanced_fusion_visualizer.py",
            "--results_dir", str(run_dir),
            "--gallery_dir", str(gallery_dir)
        ]

        try:
            subprocess.run(viz_cmd, capture_output=True, text=True, timeout=60)
            viz_time = time.time() - start_time
            self.timing_stats['visualization'].append(viz_time)
        except subprocess.TimeoutExpired:
            if self.verbose:
                print(f"   ⏰ Visualization timeout")
            self.timing_stats['visualization'].append(60)
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Visualization error: {e}")
            self.timing_stats['visualization'].append(60)

    def get_frame_data(self, frame_id: str) -> Dict:
        """Get frame data including paths"""
        return {
            'frame_id': frame_id,
            'img_path': self.ehf_path / f"{frame_id}_img.jpg",
            'align_path': self.ehf_path / f"{frame_id}_align.ply",
            'joints_2d_path': self.ehf_path / f"{frame_id}_2Djnt.json",
            'camera_params': self.camera_params
        }

    def load_ground_truth_mesh(self, frame_id: str) -> Optional[np.ndarray]:
        """Load ground truth mesh vertices"""
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        if not align_path.exists():
            if self.verbose:
                print(f"   ⚠️ Ground truth mesh not found: {align_path}")
            return None
        
        try:
            mesh = trimesh.load(align_path)
            return np.array(mesh.vertices)
        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Failed to load ground truth mesh: {e}")
            return None

    def calculate_metrics(self, baseline_result: Dict, fusion_result: Dict, frame_data: Dict) -> Tuple[Dict, Dict]:
        """Calculate V2V and PA-V2V metrics"""
        try:
            gt_vertices = self.load_ground_truth_mesh(frame_data['frame_id'])
            if gt_vertices is None:
                fallback = {'V2V': float('nan'), 'PA_V2V': float('nan')}
                return fallback, fallback

            baseline_metrics = self.mesh_metrics.calculate_mesh_metrics(
                baseline_result['mesh'], gt_vertices
            )
            fusion_metrics = self.mesh_metrics.calculate_mesh_metrics(
                fusion_result['mesh'], gt_vertices
            )

            return baseline_metrics, fusion_metrics

        except Exception as e:
            if self.verbose:
                print(f"   ⚠️ Metrics calculation failed: {e}")
            fallback = {'V2V': float('nan'), 'PA_V2V': float('nan')}
            return fallback, fallback

    def print_performance_stats(self):
        """Print performance statistics"""
        if not self.verbose:
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

    def run_evaluation(self, max_frames: Optional[int] = None) -> Dict:
        """Run complete evaluation"""
        frames_to_eval = self.frames[:max_frames] if max_frames else self.frames

        if self.verbose:
            print(f"🚀 Starting EHF Evaluation")
            print(f"   System: {self.get_system_info()}")
            print(f"   Frames: {len(frames_to_eval)}")
            print(f"   Metrics: V2V, PA-V2V")

        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        fusion_status = {'success': 0, 'failed': 0, 'error': 0}

        for i, frame_id in enumerate(tqdm(frames_to_eval, desc="Processing", disable=not self.verbose)):
            if self.verbose:
                print(f"\n   [{i+1}/{len(frames_to_eval)}] Frame: {frame_id}")
            
            frame_start = time.time()
            
            try:
                frame_data = self.get_frame_data(frame_id)
                baseline_result, fusion_result = self.run_pipeline_for_frame(frame_data)

                if baseline_result is None:
                    fusion_status['error'] += 1
                    continue

                # Calculate metrics
                baseline_metrics, fusion_metrics = self.calculate_metrics(
                    baseline_result, fusion_result, frame_data
                )

                # Determine fusion status
                status = 'failed'
                if (baseline_result is not None and fusion_result is not None and
                    'mesh' in baseline_result and 'mesh' in fusion_result):
                    if not np.allclose(baseline_result['mesh'], fusion_result['mesh'], atol=1e-6):
                        status = 'success'
                        fusion_status['success'] += 1
                    else:
                        fusion_status['failed'] += 1

                # Store results
                result = {
                    'frame_id': frame_id,
                    'baseline_metrics': baseline_metrics,
                    'fusion_metrics': fusion_metrics,
                    'fusion_status': status,
                }
                
                if self.verbose:
                    result['gallery_path'] = str(self.gallery_dir / frame_id)

                all_results.append(result)
                baseline_metrics_all.append(baseline_metrics)
                fusion_metrics_all.append(fusion_metrics)

                frame_time = time.time() - frame_start
                if self.verbose:
                    print(f"   ⏱️ Frame completed in {frame_time:.1f}s")

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Error processing {frame_id}: {e}")
                fusion_status['error'] += 1

        # Print performance analysis
        self.print_performance_stats()

        # Aggregate and save results
        final_results = self.aggregate_results(baseline_metrics_all, fusion_metrics_all)
        self.save_results(final_results, fusion_status, all_results)

        return final_results

    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all frames"""
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            
            averaged = {}
            for key in ['V2V', 'PA_V2V']:
                values = [m[key] for m in metrics_list if key in m and 
                         isinstance(m[key], (int, float)) and not np.isnan(m[key])]
                averaged[key] = float(np.mean(values)) if values else float('nan')
            return averaged

        baseline_avg = average_metrics(baseline_metrics)
        fusion_avg = average_metrics(fusion_metrics)

        improvements = {}
        for key in ['V2V', 'PA_V2V']:
            if (key in baseline_avg and key in fusion_avg and
                not np.isnan(baseline_avg[key]) and not np.isnan(fusion_avg[key]) and
                baseline_avg[key] > 0):
                
                reduction = ((baseline_avg[key] - fusion_avg[key]) / baseline_avg[key]) * 100
                improvements[f"{key}_reduction_%"] = float(reduction)
            else:
                improvements[f"{key}_reduction_%"] = float('nan')

        return {
            'baseline_average': baseline_avg,
            'fusion_average': fusion_avg,
            'improvements': improvements
        }

    def save_results(self, results: Dict, fusion_status: Dict, all_results: List[Dict]):
        """Save evaluation results to JSON"""
        output_data = {
            'comparison_summary': {
                'metrics_used': ['V2V', 'PA_V2V'],
                'baseline_average_metrics': results['baseline_average'],
                'fusion_average_metrics': results['fusion_average'],
                'improvements_percentage': results['improvements'],
                'fusion_status_summary': fusion_status
            },
            'metadata': {
                'evaluated_frames_count': len(all_results),
                'output_directory': str(self.output_dir)
            }
        }

        if self.verbose:
            output_data['detailed_timing_stats'] = {
                k: {'mean': float(np.mean(v)) if v else float('nan'),
                    'std': float(np.std(v)) if v else float('nan')}
                for k, v in self.timing_stats.items()
            }
            output_data['per_frame_results'] = all_results

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj) if not np.isnan(obj) else None
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                return super().default(obj)

        output_path = self.output_dir / "evaluation_results.json"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)

        if self.verbose:
            print(f"✅ Results saved to: {output_path}")

    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        summary = results['comparison_summary']
        
        print("\n--- Evaluation Summary ---")
        print(f"Metrics: {', '.join(summary['metrics_used'])}")
        
        print("\nAverage Baseline Metrics:")
        for metric, value in summary['baseline_average_metrics'].items():
            print(f"  {metric}: {value:.3f}" if not np.isnan(value) else f"  {metric}: N/A")
        
        print("\nAverage Fusion Metrics:")
        for metric, value in summary['fusion_average_metrics'].items():
            print(f"  {metric}: {value:.3f}" if not np.isnan(value) else f"  {metric}: N/A")
        
        print("\nImprovements (%):")
        for metric, value in summary['improvements_percentage'].items():
            print(f"  {metric}: {value:.2f}%" if not np.isnan(value) else f"  {metric}: N/A")
        
        status = summary['fusion_status_summary']
        print(f"\nFusion Status: Success: {status['success']}, Failed: {status['failed']}, Errors: {status['error']}")
        print("--------------------------")


def main():
    parser = argparse.ArgumentParser(description='EHF Evaluation with V2V and PA-V2V Metrics')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='EHF dataset path')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py', 
                       help='SMPLest-X config path')
    parser.add_argument('--max_frames', type=int, default=10, help='Maximum frames to process')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    max_frames = args.max_frames if args.max_frames > 0 else None
    
    if args.verbose:
        print("🚀 Starting EHF Evaluation with V2V and PA-V2V Metrics")
    else:
        print("🚀 Starting EHF Evaluation")

    start_time = time.time()
    
    evaluator = EHFEvaluator(args.ehf_path, args.config, args.verbose)
    results = evaluator.run_evaluation(max_frames)
    
    total_time = time.time() - start_time
    frame_count = results['metadata']['evaluated_frames_count']
    
    print(f"\n✅ Evaluation completed!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   Frames processed: {frame_count}")
    if frame_count > 0:
        print(f"   Average per frame: {total_time/frame_count:.1f}s")
    
    evaluator.print_summary(results)


if __name__ == '__main__':
    main()