#!/usr/bin/env python3
"""
Compatible Optimized EHF Evaluator
Works with existing single-image main.py but applies optimizations
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

class CompatibleOptimizedEHFEvaluator:
    """Optimized evaluator compatible with existing single-image main.py"""
    
    def __init__(self, ehf_path: str = "data/EHF", 
                 smplestx_config: str = "pretrained_models/smplest_x/config_base.py"):
        self.ehf_path = Path(ehf_path)
        self.config = Config.load_config(smplestx_config)
        
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
        """Run optimized pipeline for single frame with performance monitoring"""
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
        """Load baseline SMPLest-X results"""
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
        """Load fusion results"""
        fusion_dir = run_dir / "fusion_results"
        if not (fusion_dir / "enhanced_mesh.npy").exists():
            return baseline_result
        
        enhanced_mesh = np.load(fusion_dir / "enhanced_mesh.npy")
        fusion_result = baseline_result.copy()
        fusion_result['mesh'] = enhanced_mesh
        fusion_result['fusion_status'] = 'success'
        
        return fusion_result

    def get_frame_data(self, frame_id: str) -> Dict:
        """Get frame data"""
        return {
            'frame_id': frame_id,
            'img_path': self.ehf_path / f"{frame_id}_img.jpg",
            'align_path': self.ehf_path / f"{frame_id}_align.ply",
            'camera_params': self.camera_params
        }
    
    def load_ground_truth_mesh(self, frame_id: str) -> np.ndarray:
        """Load ground truth mesh"""
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        mesh = trimesh.load(align_path)
        return np.array(mesh.vertices)

    def calculate_metrics(self, predicted_mesh: np.ndarray, gt_mesh: np.ndarray) -> Dict:
        """Calculate evaluation metrics quickly"""
        if predicted_mesh is None or gt_mesh is None:
            return {}
        
        # Per Vertex Error
        vertex_errors = np.linalg.norm(predicted_mesh - gt_mesh, axis=1)
        metrics = {'PVE': float(np.mean(vertex_errors) * 1000)}
        
        # Quick Procrustes alignment
        pred_centered = predicted_mesh - predicted_mesh.mean(axis=0)
        gt_centered = gt_mesh - gt_mesh.mean(axis=0)
        
        pred_scale = np.sqrt(np.sum(pred_centered**2))
        gt_scale = np.sqrt(np.sum(gt_centered**2))
        
        if pred_scale > 0 and gt_scale > 0:
            pred_normalized = pred_centered / pred_scale
            gt_normalized = gt_centered / gt_scale
            
            H = pred_normalized.T @ gt_normalized
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            pred_aligned = (pred_normalized @ R) * gt_scale + gt_mesh.mean(axis=0)
            vertex_errors_aligned = np.linalg.norm(pred_aligned - gt_mesh, axis=1)
            metrics['PA-PVE'] = float(np.mean(vertex_errors_aligned) * 1000)
        
        return metrics

    def print_performance_stats(self):
        """Print detailed performance statistics"""
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
        """Run evaluation compatible with existing main.py"""
        frames_to_eval = self.frames[:max_frames] if max_frames else self.frames
        
        print(f"🚀 Starting compatible optimized evaluation...")
        print(f"   System: {self.get_system_info()}")
        print(f"   Total frames: {len(frames_to_eval)}")
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
                gt_mesh = self.load_ground_truth_mesh(frame_id)
                
                # Run optimized pipeline
                baseline_result, fusion_result = self.run_optimized_single_frame_pipeline(frame_data)
                
                if baseline_result is None:
                    print(f"      ❌ Pipeline failed for frame {frame_id}")
                    fusion_status_summary['error'] += 1
                    continue
                
                # Calculate metrics
                baseline_metrics = self.calculate_metrics(baseline_result['mesh'], gt_mesh)
                fusion_metrics = self.calculate_metrics(fusion_result['mesh'], gt_mesh)
                
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
                
                if 'PVE' in baseline_metrics and 'PVE' in fusion_metrics:
                    improvement = ((baseline_metrics['PVE'] - fusion_metrics['PVE']) / baseline_metrics['PVE']) * 100
                    print(f"      📊 PVE improvement: {improvement:.2f}%")
                
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
        
        # Save results
        with open(self.output_dir / "compatible_evaluation_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results

    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across frames"""
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            
            keys = set()
            for m in metrics_list:
                keys.update(m.keys())
            
            averaged = {}
            for key in keys:
                values = [m[key] for m in metrics_list if key in m]
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
    parser = argparse.ArgumentParser(description='Compatible Optimized EHF Evaluation')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='EHF dataset path')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py', help='Config path')
    parser.add_argument('--max_frames', type=int, default=10, help='Max frames (0 for all)')
    
    args = parser.parse_args()
    
    max_frames = args.max_frames if args.max_frames > 0 else None
    
    print("🚀 Starting COMPATIBLE OPTIMIZED EHF Evaluation")
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