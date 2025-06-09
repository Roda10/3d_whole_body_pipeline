#!/usr/bin/env python3
"""
EHF Fusion Evaluator
Compares baseline SMPLest-X vs Fusion (SMPLest-X + WiLoR + EMOCA) on EHF dataset
Leverages parallel processing for speed.
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import trimesh
import subprocess
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime
from tqdm import tqdm
import time as pytime
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
        for img_file in sorted(self.ehf_path.glob("*_img.jpg")):
            frame_id = img_file.stem.replace("_img", "")
            frames.append(frame_id)
        print(f"Found {len(frames)} EHF frames in dataset directory")
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
        
        for key, path in frame_data.items():
            if key.endswith('_path') and not path.exists():
                raise FileNotFoundError(f"Missing file for frame {frame_id}: {path}")
        
        return frame_data
    
    def load_ground_truth_mesh(self, frame_id: str) -> np.ndarray:
        align_path = self.ehf_path / f"{frame_id}_align.ply"
        mesh = trimesh.load(align_path, process=False)
        return np.array(mesh.vertices)

class EHFFusionEvaluator:
    """Main evaluator class for comparing baseline vs fusion on EHF"""
    
    def __init__(self, ehf_path: str = "data/EHF", 
                 smplestx_config: str = "pretrained_models/smplest_x/config_base.py"):
        self.ehf_dataset = EHFDataset(ehf_path)
        self.config = Config.load_config(smplestx_config)
        self.output_dir = Path("evaluation_results") / f"ehf_evaluation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_config()
        self.setup_models()
        
        print(f"EHF Evaluator initialized")
        print(f"Outputting results to: {self.output_dir}")
    
    def setup_config(self):
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
    
    def setup_models(self):
        self.smpl = SMPL(self.config.model.human_model_path)
        self.smplx = SMPLX(self.config.model.human_model_path)
        print("✅ Models initialized for metric calculations.")
    
    def run_complete_pipeline(self, frame_data: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        frame_id = frame_data['frame_id']
        img_path = frame_data['img_path']
        temp_dir = self.output_dir / "temp_pipeline" / frame_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Run main.py
            main_cmd = [sys.executable, "main.py", "--input_image", str(img_path), "--output_dir", str(temp_dir)]
            subprocess.run(main_cmd, capture_output=True, text=True, cwd=os.getcwd(), check=True)
            
            run_dirs = sorted(list(temp_dir.glob("run_*")), key=os.path.getmtime, reverse=True)
            if not run_dirs: return None, None
            run_dir = run_dirs[0]
            
            baseline_result = self._load_baseline_from_pipeline(run_dir)
            if baseline_result is None: return None, None
            
            # Step 2: Run coordinate analysis (with strict checking)
            coord_cmd = [sys.executable, "analysis_tools/coordinate_analyzer_fixed.py", str(run_dir)]
            subprocess.run(coord_cmd, capture_output=True, text=True, cwd=os.getcwd(), check=True)
            
            # Step 3: Run fusion (with strict checking)
            fusion_cmd = [sys.executable, "fusion/direct_parameter_fusion.py", "--results_dir", str(run_dir)]
            subprocess.run(fusion_cmd, capture_output=True, text=True, cwd=os.getcwd(), check=True)

            fusion_result = self._load_fusion_from_pipeline(run_dir, baseline_result)
            if fusion_result is None: return baseline_result, baseline_result
            
            return baseline_result, fusion_result
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Pipeline subprocess failed for frame {frame_id}. Stderr:\n{e.stderr}")
            return None, None
        except Exception as e:
            print(f"   ❌ Pipeline error for frame {frame_id}: {e}")
            return None, None

    def _load_baseline_from_pipeline(self, run_dir: Path) -> Optional[Dict]:
        smplx_files = list(run_dir.glob("smplestx_results/*/person_*/smplx_params_*.json"))
        if not smplx_files: return None
        with open(smplx_files[0], 'r') as f: params = json.load(f)
        return {'mesh': np.array(params['mesh']), 'joints_3d': np.array(params['joints_3d']), 'joints_2d': np.array(params['joints_2d']),
                'parameters': {'betas': np.array(params['betas']), 'body_pose': np.array(params['body_pose']), 'left_hand_pose': np.array(params['left_hand_pose']),
                               'right_hand_pose': np.array(params['right_hand_pose']), 'expression': np.array(params['expression']),
                               'root_pose': np.array(params['root_pose']), 'jaw_pose': np.array(params['jaw_pose']), 'translation': np.array(params['translation'])}}

    def _load_fusion_from_pipeline(self, run_dir: Path, baseline_result: Dict) -> Optional[Dict]:
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

    def calculate_metrics(self, predicted_mesh: np.ndarray, gt_mesh: np.ndarray) -> Dict:
        metrics = {}
        if predicted_mesh is not None and gt_mesh is not None:
            vertex_errors = np.linalg.norm(predicted_mesh - gt_mesh, axis=1)
            metrics['PVE'] = float(np.mean(vertex_errors) * 1000)
        return metrics
    
    def evaluate_single_frame(self, frame_id: str) -> Dict:
        frame_data = self.ehf_dataset.get_frame_data(frame_id)
        gt_mesh = self.ehf_dataset.load_ground_truth_mesh(frame_id)
        
        baseline_result, fusion_result = self.run_complete_pipeline(frame_data)
        
        if baseline_result is None:
            return {'frame_id': frame_id, 'status': 'pipeline_error'}
        
        fusion_status = 'failed_or_identical'
        if fusion_result is not None and not np.array_equal(baseline_result['mesh'], fusion_result['mesh']):
            fusion_status = 'success'
        else:
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
    
    def _evaluate_frame_wrapper(self, frame_id: str) -> Optional[Dict]:
        try:
            return self.evaluate_single_frame(frame_id)
        except Exception as e:
            print(f"--- Critical error processing frame {frame_id}: {e} ---")
            return {'frame_id': frame_id, 'status': 'critical_error'}

    def run_full_evaluation(self, max_frames: Optional[int] = None, num_workers: int = 4, frame_list_path: Optional[str] = None) -> Dict:
        """Run evaluation on EHF frames in parallel."""
        
        if frame_list_path:
            print(f"Loading frames to evaluate from file: {frame_list_path}")
            try:
                with open(frame_list_path, 'r') as f:
                    frames_to_eval = [line.strip() for line in f if line.strip()]
                print(f"Found {len(frames_to_eval)} frames in the list.")
            except FileNotFoundError:
                print(f"Error: Frame list file not found at {frame_list_path}")
                return {}
        else:
            print("No frame list provided. Using all frames from the dataset directory.")
            frames_to_eval = self.ehf_dataset.frames
            if max_frames:
                frames_to_eval = frames_to_eval[:max_frames]
        
        if not frames_to_eval:
            print("No frames to evaluate. Exiting.")
            return {}
            
        print(f"Starting evaluation on {len(frames_to_eval)} frames using {num_workers} parallel workers...")
        
        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        status_summary = {'success': 0, 'failed_or_identical': 0, 'pipeline_error': 0, 'critical_error': 0}

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._evaluate_frame_wrapper, frame_id): frame_id for frame_id in frames_to_eval}
            
            for future in tqdm(as_completed(futures), total=len(frames_to_eval), desc="Processing Frames"):
                result = future.result()
                
                if result is None: continue

                if result['status'] == 'success':
                    all_results.append(result)
                    baseline_metrics_all.append(result['baseline_metrics'])
                    fusion_metrics_all.append(result['fusion_metrics'])
                    status_summary[result['fusion_status']] += 1
                else:
                    status_summary[result['status']] += 1
                    print(f"Frame {result['frame_id']} failed with status: {result['status']}")

        if not all_results:
            print(f"\n❌ No frames were successfully evaluated!")
            return {'total_frames': 0, 'status_summary': status_summary, 'error': 'No successful evaluations'}
        
        final_results = self.aggregate_results(baseline_metrics_all, fusion_metrics_all)
        final_results['per_frame_results'] = all_results
        final_results['total_frames_evaluated'] = len(all_results)
        final_results['total_frames_submitted'] = len(frames_to_eval)
        final_results['status_summary'] = status_summary
        
        self.save_results(final_results)
        
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Total frames submitted: {len(frames_to_eval)}")
        print(f"   Successfully evaluated: {len(all_results)}")
        print(f"   Fusion success count: {status_summary['success']}")
        for status, count in status_summary.items():
            if status != 'success' and count > 0:
                print(f"   {status.replace('_', ' ').title()}: {count}")
        
        return final_results
    
    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        def average_metrics(metrics_list):
            if not metrics_list: return {}
            keys = metrics_list[0].keys()
            averaged = {}
            for key in keys:
                values = [m[key] for m in metrics_list if key in m and m[key] is not None]
                if values:
                    averaged[key] = float(np.mean(values))
            return averaged
        
        baseline_avg = average_metrics(baseline_metrics)
        fusion_avg = average_metrics(fusion_metrics)
        improvements = {}
        for key in baseline_avg:
            if key in fusion_avg and baseline_avg[key] > 0:
                improvements[f"{key}_improvement_%"] = float(((baseline_avg[key] - fusion_avg[key]) / baseline_avg[key]) * 100)
        
        return {'baseline_average': baseline_avg, 'fusion_average': fusion_avg, 'improvements': improvements}
    
    def save_results(self, results: Dict):
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        with open(self.output_dir / "ehf_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        self.create_comparison_table(results)
        print(f"✅ Results saved to: {self.output_dir}")

    def create_comparison_table(self, results: Dict):
        baseline = results.get('baseline_average', {})
        fusion = results.get('fusion_average', {})
        
        table_content = f"EHF EVALUATION RESULTS - {self.output_dir.name}\n" + "=" * 50 + "\n\n"
        table_content += f"Total Frames Evaluated: {results.get('total_frames_evaluated', 0)} / {results.get('total_frames_submitted', 0)}\n"
        
        table_content += "\nMETRIC COMPARISON (PVE in mm, lower is better)\n" + "-" * 50 + "\n"
        table_content += f"{'Metric':<15} {'Baseline':<12} {'Fusion':<12} {'Improvement':<12}\n" + "-" * 50 + "\n"
        
        pve_b = baseline.get('PVE', 'N/A')
        pve_f = fusion.get('PVE', 'N/A')
        imp = results.get('improvements', {}).get('PVE_improvement_%', 'N/A')
        
        pve_b_str = f"{pve_b:.2f}" if isinstance(pve_b, float) else pve_b
        pve_f_str = f"{pve_f:.2f}" if isinstance(pve_f, float) else pve_f
        imp_str = f"{imp:.2f}%" if isinstance(imp, float) else imp
        
        table_content += f"{'PVE':<15} {pve_b_str:<12} {pve_f_str:<12} {imp_str:<12}\n"
        
        with open(self.output_dir / "comparison_summary.txt", 'w') as f: f.write(table_content)
        print("\n" + table_content)

def main():
    parser = argparse.ArgumentParser(description='EHF Fusion Evaluation (Parallel)')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='Path to EHF dataset')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py', help='Path to SMPLest-X config')
    parser.add_argument('--max_frames', type=int, default=0, help='Max frames to run if no frame_list is provided (0 for all)')
    parser.add_argument('--workers', type=int, default=3, help='Number of parallel workers. Recommended: 2-4 per L4 GPU.')
    parser.add_argument('--frame_list', type=str, default=None, help='Path to a text file containing frame IDs to process, one per line.')
    
    args = parser.parse_args()

    max_frames = args.max_frames if args.max_frames > 0 else None
    
    init_start_time = pytime.time()
    evaluator = EHFFusionEvaluator(args.ehf_path, args.config)
    init_end_time = pytime.time()
    print(f"--- Initialization complete in {init_end_time - init_start_time:.2f} seconds ---\n")
    
    eval_start_time = pytime.time()
    results = evaluator.run_full_evaluation(
        max_frames=max_frames, 
        num_workers=args.workers, 
        frame_list_path=args.frame_list
    )
    eval_end_time = pytime.time()
    
    total_eval_time = eval_end_time - eval_start_time
    evaluated_frames = results.get('total_frames_evaluated', 0)

    print(f"\n--- Evaluation Timing Summary ---")
    print(f"Total evaluation wall-clock time: {total_eval_time:.2f} seconds")
    minutes, seconds = divmod(total_eval_time, 60)
    print(f"Equivalent to: {int(minutes)} minutes and {seconds:.2f} seconds")

    if evaluated_frames > 0:
        avg_time_per_frame = total_eval_time / evaluated_frames
        print(f"Average time per frame: {avg_time_per_frame:.2f} seconds")

    print(f"\n✅ Evaluation complete!")

if __name__ == '__main__':
    main()