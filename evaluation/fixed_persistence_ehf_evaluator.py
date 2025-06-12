#!/usr/bin/env python3
"""
Fixed Persistence EHF Evaluator - Matches working path patterns from slow version
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

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from main.config import Config
    from human_models.human_models import SMPL, SMPLX
except ImportError as e:
    print(f"Error importing SMPLest-X modules: {e}")
    sys.exit(1)

class FixedPersistenceEHFEvaluator:
    """Fixed EHF Evaluator with proper path handling like the working slow version"""
    
    def __init__(self, ehf_path: str = "data/EHF"):
        self.ehf_path = Path(ehf_path)
        
        # Setup output directory - MATCH slow version pattern
        self.output_dir = Path("evaluation_results") / f"ehf_persistence_fixed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directories - MATCH slow version
        self.central_gallery = self.output_dir / "gallery"
        self.central_gallery.mkdir(exist_ok=True)
        self.shared_temp_dir = self.output_dir / "shared_temp"
        self.shared_temp_dir.mkdir(exist_ok=True)
        
        # Load frame list
        self.frames = self._load_frame_list()
        
        # Load camera parameters
        self.camera_params = {
            'focal': [1498.22426237, 1498.22426237],
            'princpt': [790.263706, 578.90334]
        }
        
        # Setup models for metrics
        config = Config.load_config("pretrained_models/smplest_x/config_base.py")
        self.smpl = SMPL(config.model.human_model_path)
        self.smplx = SMPLX(config.model.human_model_path)
        
        print(f"✅ Fixed Persistence EHF Evaluator initialized")
        print(f"   Dataset: {len(self.frames)} frames")
        print(f"   Output: {self.output_dir}")
        print(f"   Central gallery: {self.central_gallery}")
        print(f"   Shared temp: {self.shared_temp_dir}")

    def _load_frame_list(self) -> List[str]:
        """Load list of EHF frame IDs"""
        frames = []
        for img_file in self.ehf_path.glob("*_img.jpg"):
            frame_id = img_file.stem.replace("_img", "")
            frames.append(frame_id)
        frames.sort()
        return frames

    def check_services_status(self):
        """Check if persistence services are running"""
        try:
            import requests
            services = ['http://localhost:8001/health', 'http://localhost:8002/health', 'http://localhost:8003/health']
            all_healthy = True
            
            for service_url in services:
                try:
                    response = requests.get(service_url, timeout=5)
                    if response.status_code != 200:
                        all_healthy = False
                        break
                except:
                    all_healthy = False
                    break
            
            return all_healthy
        except:
            return False

    def cleanup_previous_runs(self):
        """Clean up GPU memory and temp files - MATCH slow version"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for temp_dir in self.shared_temp_dir.glob("temp_*"):
            if temp_dir.is_dir():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass

    def run_persistence_pipeline_with_gallery(self, frame_data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Run persistence pipeline with proper gallery handling - MATCH slow version pattern"""
        frame_id = frame_data['frame_id']
        img_path = frame_data['img_path']
        
        # Create frame-specific gallery - MATCH slow version
        frame_gallery = self.central_gallery / frame_id
        frame_gallery.mkdir(exist_ok=True)
        
        # Use shared temp directory - MATCH slow version
        temp_dir = self.shared_temp_dir / f"temp_{frame_id}"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Clean up before starting - MATCH slow version
            self.cleanup_previous_runs()
            
            print(f"      🚀 Running persistence pipeline...")
            start_time = pytime.time()
            
            # Use the optimized main_persistence.py with temp_dir
            main_cmd = [
                sys.executable, "main_persistence.py",
                "--input_image", str(img_path),
                "--output_dir", str(temp_dir)
            ]
            
            result = subprocess.run(main_cmd, capture_output=True, text=True, 
                                  cwd=os.getcwd(), timeout=180)
            
            if result.returncode != 0:
                print(f"      ❌ Persistence pipeline failed: {result.stderr}")
                return None, None
            
            pipeline_time = pytime.time() - start_time
            print(f"      ⏱️ Persistence pipeline: {pipeline_time:.1f}s")
            
            # Find the latest run directory - MATCH slow version pattern
            run_dirs = sorted(list(temp_dir.glob("run_*")), 
                            key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not run_dirs:
                print(f"      ❌ No run directory found")
                return None, None
            
            run_dir = run_dirs[0]
            
            # Load baseline results immediately - MATCH slow version
            baseline_result = self._load_baseline_from_pipeline(run_dir)
            if baseline_result is None:
                print(f"      ❌ Failed to load baseline results")
                return None, None
            
            # Step 2: Coordinate analysis with timeout - MATCH slow version
            print(f"      📐 Running coordinate analysis...")
            coord_start_time = pytime.time()
            coord_cmd = [
                sys.executable, "analysis_tools/persistence_coordinate_analyzer.py",
                "--results_dir", str(run_dir)
            ]
            
            try:
                result = subprocess.run(coord_cmd, capture_output=True, text=True, 
                                      cwd=os.getcwd(), timeout=45)
                coord_time = pytime.time() - coord_start_time
                print(f"      ⏱️ Coordinate analysis: {coord_time:.1f}s")
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Coordinate analysis timeout")
                coord_time = 45
            
            # Step 3: Fusion with gallery parameter - MATCH slow version
            print(f"      🔄 Running fusion...")
            fusion_start_time = pytime.time()
            fusion_cmd = [
                sys.executable, "fusion/direct_parameter_fusion.py",
                "--results_dir", str(run_dir),
                "--gallery_dir", str(frame_gallery)  # KEY FIX: Add gallery parameter
            ]
            
            fusion_success = False
            try:
                result = subprocess.run(fusion_cmd, capture_output=True, text=True, 
                                      cwd=os.getcwd(), timeout=90)
                
                if result.returncode == 0:
                    fusion_success = True
                    fusion_time = pytime.time() - fusion_start_time
                    print(f"      ⏱️ Fusion: {fusion_time:.1f}s")
                else:
                    print(f"      ❌ Fusion failed: {result.stderr}")
                    fusion_time = pytime.time() - fusion_start_time
                
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Fusion timeout")
                fusion_time = 90
            
            # Step 4: Visualization with gallery - MATCH slow version
            print(f"      📊 Running visualization...")
            viz_start_time = pytime.time()
            viz_cmd = [
                sys.executable, "fusion/enhanced_fusion_visualizer.py", 
                "--results_dir", str(run_dir),
                "--gallery_dir", str(frame_gallery)  # KEY FIX: Add gallery parameter
            ]
            
            try:
                result = subprocess.run(viz_cmd, capture_output=True, text=True, 
                                      cwd=os.getcwd(), timeout=60)
                viz_time = pytime.time() - viz_start_time
                print(f"      ⏱️ Visualization: {viz_time:.1f}s")
            except subprocess.TimeoutExpired:
                print(f"      ⏰ Visualization timeout")
                viz_time = 60
            
            # Load fusion results - IMPROVED pattern matching
            if fusion_success:
                fusion_result = self._load_fusion_from_pipeline(run_dir, baseline_result)
            else:
                fusion_result = baseline_result
            
            total_time = pipeline_time + coord_time + fusion_time + viz_time
            print(f"      ⏱️ Total time: {total_time:.1f}s")
            
            return baseline_result, fusion_result
            
        except Exception as e:
            print(f"      ❌ Pipeline error: {e}")
            return None, None
        finally:
            # Clean up temp directory - MATCH slow version
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass

    def _load_baseline_from_pipeline(self, run_dir: Path) -> Optional[Dict]:
        """Load baseline SMPLest-X results - IMPROVED pattern matching"""
        # More comprehensive pattern matching
        patterns = [
            "smplestx_results/*/person_*/smplx_params_person_*.json",
            "smplestx_results/inference_output_*/person_*/smplx_params_person_*.json",
            "smplestx_results/person_*/smplx_params_person_*.json"
        ]
        
        smplx_files = []
        for pattern in patterns:
            found_files = list(run_dir.glob(pattern))
            smplx_files.extend(found_files)
            if found_files:
                print(f"         📄 Found SMPLest-X files with pattern: {pattern}")
                break
        
        if not smplx_files:
            print(f"         ❌ No SMPLest-X parameter files found in {run_dir}")
            return None
        
        # Use the first file found
        param_file = smplx_files[0]
        print(f"         ✅ Loading: {param_file.relative_to(run_dir)}")
        
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        result = {
            'mesh': np.array(params['mesh']),
            'joints_3d': np.array(params['joints_3d']),
            'joints_2d': np.array(params['joints_2d']),
            'parameters': {k: np.array(v) for k, v in params.items() 
                          if k not in ['mesh', 'joints_3d', 'joints_2d']}
        }
        print(f"         ✅ Loaded mesh with {result['mesh'].shape[0]} vertices")
        return result

    def _load_fusion_from_pipeline(self, run_dir: Path, baseline_result: Dict) -> Dict:
        """Load fusion results - IMPROVED pattern matching"""
        fusion_dir = run_dir / "fusion_results"
        enhanced_mesh_file = fusion_dir / "enhanced_mesh.npy"
        
        print(f"         🔍 Looking for fusion results in: {fusion_dir}")
        
        if not enhanced_mesh_file.exists():
            print(f"         ⚠️ Enhanced mesh not found, using baseline")
            return baseline_result
        
        try:
            enhanced_mesh = np.load(enhanced_mesh_file)
            fusion_result = baseline_result.copy()
            fusion_result['mesh'] = enhanced_mesh
            fusion_result['fusion_status'] = 'success'
            
            print(f"         ✅ Loaded enhanced mesh with {enhanced_mesh.shape[0]} vertices")
            
            # Check if mesh actually changed
            mesh_diff = np.linalg.norm(enhanced_mesh - baseline_result['mesh'])
            print(f"         📊 Mesh difference magnitude: {mesh_diff:.6f}")
            
            return fusion_result
            
        except Exception as e:
            print(f"         ❌ Error loading fusion results: {e}")
            return baseline_result

    def get_frame_data(self, frame_id: str) -> Dict:
        """Get frame data"""
        return {
            'frame_id': frame_id,
            'img_path': self.ehf_path / f"{frame_id}_img.jpg",
            'align_path': self.ehf_path / f"{frame_id}_align.ply",
            'camera_params': self.camera_params,
            'gallery_path': self.central_gallery / frame_id  # Add gallery path
        }
    
    def load_ground_truth_mesh(self, frame_id: str) -> np.ndarray:
        """Load ground truth mesh"""
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        mesh = trimesh.load(align_path)
        return np.array(mesh.vertices)

    def calculate_metrics(self, predicted_mesh: np.ndarray, gt_mesh: np.ndarray) -> Dict:
        """Calculate evaluation metrics"""
        if predicted_mesh is None or gt_mesh is None:
            return {}
        
        # Per Vertex Error in millimeters
        vertex_errors = np.linalg.norm(predicted_mesh - gt_mesh, axis=1)
        metrics = {'PVE': float(np.mean(vertex_errors) * 1000)}
        
        return metrics

    def run_evaluation(self, max_frames: Optional[int] = None) -> Dict:
        """Run fixed persistence-based evaluation"""
        frames_to_eval = self.frames[:max_frames] if max_frames else self.frames
        
        print(f"🚀 Starting FIXED PERSISTENCE EHF Evaluation")
        print(f"   Total frames: {len(frames_to_eval)}")
        print(f"   Gallery system: Enabled")
        print(f"   Path fixes: Applied")
        
        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        fusion_status_summary = {'success': 0, 'failed_or_identical': 0, 'error': 0}
        
        total_start_time = pytime.time()
        
        for i, frame_id in enumerate(frames_to_eval):
            print(f"\n   [{i+1}/{len(frames_to_eval)}] Frame: {frame_id}")
            frame_start_time = pytime.time()
            
            try:
                # Get frame data
                frame_data = self.get_frame_data(frame_id)
                gt_mesh = self.load_ground_truth_mesh(frame_id)
                
                # Run persistence pipeline with gallery
                baseline_result, fusion_result = self.run_persistence_pipeline_with_gallery(frame_data)
                
                if baseline_result is None:
                    print(f"      ❌ Pipeline failed for frame {frame_id}")
                    fusion_status_summary['error'] += 1
                    continue
                
                # Calculate metrics
                baseline_metrics = self.calculate_metrics(baseline_result['mesh'], gt_mesh)
                fusion_metrics = self.calculate_metrics(fusion_result['mesh'], gt_mesh)
                
                # Determine fusion status - IMPROVED logic
                mesh_diff = np.linalg.norm(baseline_result['mesh'] - fusion_result['mesh'])
                if mesh_diff > 1e-6:  # More sensitive threshold
                    fusion_status = 'success'
                    fusion_status_summary['success'] += 1
                    print(f"      ✅ Fusion successful - mesh changed by {mesh_diff:.6f}")
                else:
                    fusion_status = 'failed_or_identical'
                    fusion_status_summary['failed_or_identical'] += 1
                    print(f"      ⚠️ Fusion identical - no mesh change detected")
                
                # Store results
                result = {
                    'frame_id': frame_id,
                    'baseline_metrics': baseline_metrics,
                    'fusion_metrics': fusion_metrics,
                    'fusion_status': fusion_status,
                    'gallery_path': str(frame_data['gallery_path']),
                    'mesh_difference_magnitude': float(mesh_diff)
                }
                
                all_results.append(result)
                baseline_metrics_all.append(baseline_metrics)
                fusion_metrics_all.append(fusion_metrics)
                
                frame_time = pytime.time() - frame_start_time
                print(f"      ⏱️ Frame completed in {frame_time:.1f}s")
                
                if 'PVE' in baseline_metrics and 'PVE' in fusion_metrics:
                    improvement = ((baseline_metrics['PVE'] - fusion_metrics['PVE']) / baseline_metrics['PVE']) * 100
                    print(f"      📊 PVE improvement: {improvement:.2f}%")
                
            except Exception as e:
                print(f"      ❌ Error processing {frame_id}: {e}")
                fusion_status_summary['error'] += 1
                continue
        
        total_time = pytime.time() - total_start_time
        
        # Aggregate results
        final_results = {
            'total_frames': len(all_results),
            'total_time_seconds': total_time,
            'average_time_per_frame': total_time / len(all_results) if all_results else 0,
            'baseline_average': self._average_metrics(baseline_metrics_all),
            'fusion_average': self._average_metrics(fusion_metrics_all),
            'fusion_status_summary': fusion_status_summary,
            'central_gallery_path': str(self.central_gallery),
            'per_frame_results': all_results
        }
        
        # Calculate improvements
        baseline_avg = final_results['baseline_average']
        fusion_avg = final_results['fusion_average']
        improvements = {}
        for key in baseline_avg:
            if key in fusion_avg and baseline_avg[key] > 0:
                improvement = ((baseline_avg[key] - fusion_avg[key]) / baseline_avg[key]) * 100
                improvements[f"{key}_improvement_%"] = float(improvement)
        final_results['improvements'] = improvements
        
        # Save results
        with open(self.output_dir / "fixed_persistence_evaluation_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        return final_results

    def _average_metrics(self, metrics_list):
        """Average metrics across frames"""
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

def main():
    parser = argparse.ArgumentParser(description='Fixed Persistence EHF Evaluation')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='EHF dataset path')
    parser.add_argument('--max_frames', type=int, default=3, help='Max frames (0 for all)')
    
    args = parser.parse_args()
    
    max_frames = args.max_frames if args.max_frames > 0 else None
    
    start_time = pytime.time()
    
    evaluator = FixedPersistenceEHFEvaluator(args.ehf_path)
    results = evaluator.run_evaluation(max_frames)
    
    total_time = pytime.time() - start_time
    evaluated_frames = results.get('total_frames', 0)
    fusion_summary = results.get('fusion_status_summary', {})
    
    print(f"\n⚡ FIXED PERSISTENCE EVALUATION RESULTS:")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    if evaluated_frames > 0:
        print(f"   Average per frame: {total_time/evaluated_frames:.1f}s")
    print(f"   Frames processed: {evaluated_frames}")
    print(f"   Fusion success: {fusion_summary.get('success', 0)}")
    print(f"   Fusion failed/identical: {fusion_summary.get('failed_or_identical', 0)}")
    
    if 'improvements' in results:
        for metric, improvement in results['improvements'].items():
            print(f"   {metric}: {improvement:.2f}%")
    
    print(f"\n🎨 Gallery: {results.get('central_gallery_path', 'N/A')}")
    print(f"✅ Fixed persistence evaluation complete!")

if __name__ == '__main__':
    main()
