#!/usr/bin/env python3
"""
EHF Fusion Evaluator
Compares baseline SMPLest-X vs Fusion (SMPLest-X + WiLoR + EMOCA) on EHF dataset
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import trimesh
import subprocess
import time as pytime
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime
from tqdm import tqdm
### MODIFICATION ###
# Import the ProcessPoolExecutor for parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from main.config import Config
    from human_models.human_models import SMPL, SMPLX
except ImportError as e:
    print(f"Error importing SMPLest-X modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class EHFDataset:
    """EHF Dataset loader for evaluation"""
    
    def __init__(self, ehf_path: str = "data/EHF"):
        self.ehf_path = Path(ehf_path)
        self.frames = self._load_frame_list()
        self.camera_params = self._load_camera_params()
        
    def _load_frame_list(self) -> List[str]:
        """Load list of EHF frame IDs"""
        frames = []
        for img_file in self.ehf_path.glob("*_img.jpg"):
            frame_id = img_file.stem.replace("_img", "")
            frames.append(frame_id)
        frames.sort()
        print(f"Found {len(frames)} EHF frames")
        return frames
    
    def _load_camera_params(self) -> Dict:
        """Load EHF camera parameters"""
        camera_file = self.ehf_path / "EHF_camera.txt"
        if not camera_file.exists():
            raise FileNotFoundError(f"Camera file not found: {camera_file}")
        
        # Parse the camera file (simplified parsing)
        camera_params = {
            'focal': [1498.22426237, 1498.22426237],
            'princpt': [790.263706, 578.90334]
        }
        return camera_params
    
    def get_frame_data(self, frame_id: str) -> Dict:
        """Get all data for a specific frame"""
        frame_data = {
            'frame_id': frame_id,
            'img_path': self.ehf_path / f"{frame_id}_img.jpg",
            'align_path': self.ehf_path / f"{frame_id}_align.ply",
            'joints_2d_path': self.ehf_path / f"{frame_id}_2Djnt.json",
            'camera_params': self.camera_params
        }
        
        # Verify all files exist
        for key, path in frame_data.items():
            if key.endswith('_path') and not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
        
        return frame_data
    
    def load_ground_truth_mesh(self, frame_id: str) -> np.ndarray:
        """Load ground truth mesh from PLY file"""
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        mesh = trimesh.load(align_path)
        return np.array(mesh.vertices)
    
    def load_2d_joints(self, frame_id: str) -> np.ndarray:
        """Load 2D joints from OpenPose JSON"""
        joints_path = self.ehf_path / f"{frame_id}_2Djnt.json"
        with open(joints_path, 'r') as f:
            data = json.load(f)
        
        # Extract pose keypoints (assuming single person)
        if data['people']:
            pose_kpts = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
            return pose_kpts
        else:
            return np.zeros((25, 3))  # Default empty pose

class EHFFusionEvaluator:
    """Main evaluator class for comparing baseline vs fusion on EHF"""
    
    def __init__(self, ehf_path: str = "data/EHF", 
                 smplestx_config: str = "pretrained_models/smplest_x/config_base.py"):
        self.ehf_dataset = EHFDataset(ehf_path)
        self.config = Config.load_config(smplestx_config)
        self.output_dir = Path("evaluation_results") / f"ehf_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup config for SMPLest-X compatibility
        self.setup_config()
        
        # Setup models
        self.setup_models()
        
        print(f"EHF Evaluator initialized")
        print(f"Dataset: {len(self.ehf_dataset.frames)} frames")
        print(f"Output: {self.output_dir}")
    
    def setup_config(self):
        """Setup config with required paths for SMPLest-X"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        new_config = {
            "log": {
                'exp_name': f'ehf_evaluation_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'output_dir': str(self.output_dir),
                'model_dir': str(self.output_dir / 'model_dump'),
                'log_dir': str(log_dir),
                'result_dir': str(self.output_dir / 'result'),
            },
            "model": { "pretrained_model_path": "pretrained_models/smplest_x/smplest_x_h.pth.tar" }
        }
        self.config.update_config(new_config)
        if not hasattr(self.config, 'log'): raise AttributeError("Config missing 'log' section")
        if not hasattr(self.config.log, 'log_dir'): raise AttributeError("Config missing 'log.log_dir'")
        print(f"✅ Config setup complete, log dir: {log_dir}")
    
    def setup_models(self):
        """Initialize SMPL-X models"""
        self.smpl = SMPL(self.config.model.human_model_path)
        self.smplx = SMPLX(self.config.model.human_model_path)
        print("✅ Models initialized (using complete pipeline for inference)")
    
    def run_complete_pipeline(self, frame_data: Dict) -> Tuple[Dict, Dict]:
        """Run complete pipeline (main.py + coordinate analysis + fusion) on EHF frame"""
        frame_id = frame_data['frame_id']
        img_path = frame_data['img_path']
        temp_dir = self.output_dir / "temp_pipeline" / frame_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Run main.py
            main_cmd = [sys.executable, "main.py", "--input_image", str(img_path), "--output_dir", str(temp_dir)]
            result = subprocess.run(main_cmd, capture_output=True, text=True, cwd=os.getcwd(), check=True)
            
            run_dirs = sorted(list(temp_dir.glob("run_*")), key=os.path.getmtime, reverse=True)
            if not run_dirs: return None, None
            run_dir = run_dirs[0]
            
            baseline_result = self._load_baseline_from_pipeline(run_dir)
            if baseline_result is None: return None, None
            
            # Step 2: Run coordinate analysis
            coord_cmd = [sys.executable, "analysis_tools/coordinate_analyzer_fixed.py", str(run_dir)]
            subprocess.run(coord_cmd, capture_output=True, text=True, cwd=os.getcwd())
            
            # Step 3: Run fusion
            fusion_cmd = [sys.executable, "fusion/direct_parameter_fusion.py", "--results_dir", str(run_dir)]
            subprocess.run(fusion_cmd, capture_output=True, text=True, cwd=os.getcwd(), check=True)

            fusion_result = self._load_fusion_from_pipeline(run_dir, baseline_result)
            if fusion_result is None: return baseline_result, baseline_result
            
            return baseline_result, fusion_result
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Pipeline subprocess failed for frame {frame_id}: {e.stderr}")
            return None, None
        except Exception as e:
            print(f"   ❌ Pipeline error for frame {frame_id}: {e}")
            return None, None

    def _load_baseline_from_pipeline(self, run_dir: Path) -> Optional[Dict]:
        """Load baseline SMPLest-X results from pipeline output"""
        smplx_files = list(run_dir.glob("smplestx_results/*/person_*/smplx_params_*.json"))
        if not smplx_files: return None
        with open(smplx_files[0], 'r') as f: params = json.load(f)
        return {'mesh': np.array(params['mesh']), 'joints_3d': np.array(params['joints_3d']), 'joints_2d': np.array(params['joints_2d']),
                'parameters': {'betas': np.array(params['betas']), 'body_pose': np.array(params['body_pose']), 'left_hand_pose': np.array(params['left_hand_pose']),
                               'right_hand_pose': np.array(params['right_hand_pose']), 'expression': np.array(params['expression']),
                               'root_pose': np.array(params['root_pose']), 'jaw_pose': np.array(params['jaw_pose']), 'translation': np.array(params['translation'])}}

    def _load_fusion_from_pipeline(self, run_dir: Path, baseline_result: Dict) -> Optional[Dict]:
        """Load fusion results from pipeline output"""
        fusion_dir = run_dir / "fusion_results"
        if not (fusion_dir / "enhanced_mesh.npy").exists(): return None
        enhanced_mesh = np.load(fusion_dir / "enhanced_mesh.npy")
        fused_params = baseline_result['parameters'].copy()
        if (fusion_dir / "fused_parameters.json").exists():
            with open(fusion_dir / "fused_parameters.json", 'r') as f: fused_params_loaded = json.load(f)
            for key, value in fused_params_loaded.items():
                if isinstance(value, list) and key != 'fusion_metadata': fused_params[key] = np.array(value)
        return {'mesh': enhanced_mesh, 'joints_3d': baseline_result['joints_3d'].copy(), 'joints_2d': baseline_result['joints_2d'].copy(),
                'parameters': fused_params, 'fusion_status': 'success'}

    def calculate_metrics(self, predicted_mesh: np.ndarray, gt_mesh: np.ndarray, 
                         predicted_joints: np.ndarray = None, gt_joints: np.ndarray = None) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        if predicted_mesh is not None and gt_mesh is not None:
            vertex_errors = np.linalg.norm(predicted_mesh - gt_mesh, axis=1)
            metrics['PVE'] = float(np.mean(vertex_errors) * 1000)
            metrics['PVE_std'] = float(np.std(vertex_errors) * 1000)
            pred_aligned = self.procrustes_align(predicted_mesh, gt_mesh)
            vertex_errors_aligned = np.linalg.norm(pred_aligned - gt_mesh, axis=1)
            metrics['PA-PVE'] = float(np.mean(vertex_errors_aligned) * 1000)
        return metrics
    
    def procrustes_align(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Procrustes alignment for fair comparison"""
        pred_centered = pred - pred.mean(axis=0)
        gt_centered = gt - gt.mean(axis=0)
        pred_scale = np.sqrt(np.sum(pred_centered**2))
        if pred_scale == 0: return pred
        pred_normalized = pred_centered / pred_scale
        gt_normalized = gt - gt.mean(axis=0)
        H = pred_normalized.T @ gt_normalized
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        return pred_normalized @ R * pred_scale + gt.mean(axis=0)

    def evaluate_single_frame(self, frame_id: str) -> Dict:
        """Evaluate both baseline and fusion on a single frame using complete pipeline"""
        # This function no longer prints, to keep parallel output clean.
        frame_data = self.ehf_dataset.get_frame_data(frame_id)
        gt_mesh = self.ehf_dataset.load_ground_truth_mesh(frame_id)
        
        baseline_result, fusion_result = self.run_complete_pipeline(frame_data)
        
        if baseline_result is None:
            return {'frame_id': frame_id, 'status': 'pipeline_error'}
        
        if fusion_result is not None and not np.array_equal(baseline_result['mesh'], fusion_result['mesh']):
            fusion_status = 'success'
        else:
            fusion_status = 'failed_or_identical'
            fusion_result = baseline_result
        
        baseline_metrics = self.calculate_metrics(baseline_result['mesh'], gt_mesh)
        fusion_metrics = self.calculate_metrics(fusion_result['mesh'], gt_mesh)
        
        return {
            'frame_id': frame_id,
            'status': 'success',
            'baseline_metrics': baseline_metrics,
            'fusion_metrics': fusion_metrics,
            'fusion_status': fusion_status
        }
    
    ### MODIFICATION ###
    # Create a wrapper function to handle errors gracefully in parallel execution.
    # This allows one process to fail without crashing the whole evaluation.
    def _evaluate_frame_wrapper(self, frame_id: str) -> Optional[Dict]:
        """Wrapper for evaluate_single_frame to catch all exceptions."""
        try:
            # Note: We need a new instance of the evaluator for each process
            # if we were to parallelize the __init__. But since we parallelize
            # a method call, `self` is correctly pickled and sent to the worker.
            return self.evaluate_single_frame(frame_id)
        except Exception as e:
            # This will catch file not found errors, etc.
            print(f"--- Critical error processing frame {frame_id}: {e} ---")
            return {'frame_id': frame_id, 'status': 'critical_error'}

    ### MODIFICATION ###
    # The run_full_evaluation method is completely rewritten to use ProcessPoolExecutor.
    def run_full_evaluation(self, max_frames: Optional[int] = 10, num_workers: int = 4) -> Dict:
        """Run evaluation on EHF frames in parallel."""
        frames_to_eval = self.ehf_dataset.frames
        if max_frames:
            frames_to_eval = frames_to_eval[:max_frames]
        
        print(f"Starting evaluation on {len(frames_to_eval)} frames using {num_workers} parallel workers...")
        
        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        fusion_status_summary = {'success': 0, 'failed_or_identical': 0, 'pipeline_error': 0, 'critical_error': 0}

        # Use ProcessPoolExecutor to run evaluations in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Create a future for each frame
            futures = {executor.submit(self._evaluate_frame_wrapper, frame_id): frame_id for frame_id in frames_to_eval}
            
            # Use tqdm to show a progress bar as futures complete
            for future in tqdm(as_completed(futures), total=len(frames_to_eval), desc="Processing Frames"):
                result = future.result()
                
                if result is None:
                    # Should not happen with the wrapper, but as a safeguard
                    continue

                if result['status'] == 'success':
                    all_results.append(result)
                    baseline_metrics_all.append(result['baseline_metrics'])
                    fusion_metrics_all.append(result['fusion_metrics'])
                    fusion_status_summary[result['fusion_status']] += 1
                elif result['status'] == 'pipeline_error':
                    fusion_status_summary['pipeline_error'] += 1
                    print(f"Pipeline error for frame: {result['frame_id']}")
                elif result['status'] == 'critical_error':
                    fusion_status_summary['critical_error'] += 1
                    print(f"Critical error for frame: {result['frame_id']}")

        if not all_results:
            print(f"\n❌ No frames were successfully evaluated!")
            return {'total_frames': 0, 'fusion_status_summary': fusion_status_summary, 'error': 'No successful evaluations'}
        
        # Aggregate and save results (this part remains the same)
        final_results = self.aggregate_results(baseline_metrics_all, fusion_metrics_all)
        final_results['per_frame_results'] = all_results
        final_results['total_frames'] = len(all_results)
        final_results['fusion_status_summary'] = fusion_status_summary
        
        self.save_results(final_results)
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Total frames submitted: {len(frames_to_eval)}")
        print(f"   Successfully evaluated: {len(all_results)}")
        print(f"   Fusion success count: {fusion_status_summary['success']}")
        if fusion_status_summary['failed_or_identical'] > 0:
            print(f"   Fusion failed/identical: {fusion_status_summary['failed_or_identical']}")
        if fusion_status_summary['pipeline_error'] > 0:
            print(f"   Pipeline errors: {fusion_status_summary['pipeline_error']}")
        if fusion_status_summary['critical_error'] > 0:
            print(f"   Critical processing errors: {fusion_status_summary['critical_error']}")
        
        return final_results
    
    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all frames"""
        def average_metrics(metrics_list):
            if not metrics_list: return {}
            keys = metrics_list[0].keys()
            averaged = {}
            for key in keys:
                values = [m[key] for m in metrics_list if key in m]
                if values:
                    averaged[key] = float(np.mean(values))
                    averaged[f"{key}_std"] = float(np.std(values))
            return averaged
        
        baseline_avg = average_metrics(baseline_metrics)
        fusion_avg = average_metrics(fusion_metrics)
        improvements = {}
        for key in baseline_avg:
            if key.endswith('_std'): continue
            if key in fusion_avg and baseline_avg[key] > 0:
                improvements[f"{key}_improvement_%"] = float(((baseline_avg[key] - fusion_avg[key]) / baseline_avg[key]) * 100)
        
        return {'baseline_average': baseline_avg, 'fusion_average': fusion_avg, 'improvements': improvements}
    
    def save_results(self, results: Dict):
        """Save evaluation results"""
        with open(self.output_dir / "ehf_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder) # Use a robust JSON encoder
        self.create_comparison_table(results)
        print(f"✅ Results saved to: {self.output_dir}")

    def create_comparison_table(self, results: Dict):
        """Create a simple comparison table"""
        baseline = results.get('baseline_average', {})
        fusion = results.get('fusion_average', {})
        improvements = results.get('improvements', {})
        fusion_summary = results.get('fusion_status_summary', {})
        
        table_content = "EHF EVALUATION RESULTS\n" + "=" * 50 + "\n\n"
        table_content += f"Total Frames Evaluated: {results.get('total_frames', 0)}\n"
        if fusion_summary:
            table_content += f"Fusion Success Rate: {fusion_summary.get('success', 0)}/{results.get('total_frames', 0)}\n"
            if fusion_summary.get('failed_or_identical', 0) > 0: table_content += f"Fusion Failed/Identical: {fusion_summary.get('failed_or_identical', 0)}\n"
            if fusion_summary.get('pipeline_error', 0) > 0: table_content += f"Pipeline Errors: {fusion_summary.get('pipeline_error', 0)}\n"
        table_content += "\nMETRIC COMPARISON\n" + "-" * 30 + "\n"
        table_content += f"{'Metric':<20} {'Baseline':<12} {'Fusion':<12} {'Improvement':<12}\n" + "-" * 56 + "\n"
        
        key_metrics = ['PVE', 'PA-PVE']
        for metric in key_metrics:
            if metric in baseline and metric in fusion:
                table_content += f"{metric:<20} {baseline.get(metric, 0):<12.2f} {fusion.get(metric, 0):<12.2f} {improvements.get(f'{metric}_improvement_%', 0):<12.2f}%\n"
        
        table_content += "\nNOTE: Lower values are better for all metrics (errors in mm)\n"
        
        with open(self.output_dir / "comparison_table.txt", 'w') as f: f.write(table_content)
        print("\n" + table_content)

# Helper for saving numpy arrays in JSON
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    parser = argparse.ArgumentParser(description='EHF Fusion Evaluation')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='Path to EHF dataset')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py', help='Path to SMPLest-X config')
    parser.add_argument('--max_frames', type=int, default=10, help='Maximum number of frames to evaluate (0 for all)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers for processing frames. Recommended: 2-4.')
    
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames > 0 else None
    
    print("--- Initializing Evaluator ---")
    init_start_time = pytime.time()
    evaluator = EHFFusionEvaluator(args.ehf_path, args.config)
    init_end_time = pytime.time()
    print(f"--- Initialization complete in {init_end_time - init_start_time:.2f} seconds ---\n")

    
    print("--- Starting Full Evaluation ---")
    eval_start_time = pytime.time()
    results = evaluator.run_full_evaluation(max_frames, args.workers)
    eval_end_time = pytime.time()
    
    total_eval_time = eval_end_time - eval_start_time
    evaluated_frames = results.get('total_frames', 0)

    print(f"\n--- Evaluation Timing Summary ---")
    print(f"Total evaluation wall-clock time: {total_eval_time:.2f} seconds")
    
    # Convert to minutes and seconds for readability if it's long
    minutes, seconds = divmod(total_eval_time, 60)
    print(f"Equivalent to: {int(minutes)} minutes and {seconds:.2f} seconds")

    if evaluated_frames > 0:
        avg_time_per_frame = total_eval_time / evaluated_frames
        print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")

    print(f"\n✅ Evaluation complete! Results saved to: {evaluator.output_dir}")


if __name__ == '__main__':
    main()