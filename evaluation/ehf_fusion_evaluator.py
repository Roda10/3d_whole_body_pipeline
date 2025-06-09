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
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import datetime
from tqdm import tqdm

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
        # Ensure log directories exist
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Update config with required paths
        new_config = {
            "log": {
                'exp_name': f'ehf_evaluation_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}',
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
        
        # Ensure the config has all required attributes
        if not hasattr(self.config, 'log'):
            raise AttributeError("Config missing 'log' section")
        if not hasattr(self.config.log, 'log_dir'):
            raise AttributeError("Config missing 'log.log_dir'")
            
        print(f"✅ Config setup complete, log dir: {log_dir}")
    
    def setup_models(self):
        """Initialize SMPL-X models (no longer need baseline tester since we use complete pipeline)"""
        # Initialize human models for metric calculations
        self.smpl = SMPL(self.config.model.human_model_path)
        self.smplx = SMPLX(self.config.model.human_model_path)
        
        print("✅ Models initialized (using complete pipeline for inference)")
    
    def run_complete_pipeline(self, frame_data: Dict) -> Tuple[Dict, Dict]:
        """Run complete pipeline (main.py + coordinate analysis + fusion) on EHF frame"""
        import subprocess
        import tempfile
        import glob
        
        frame_id = frame_data['frame_id']
        img_path = frame_data['img_path']
        
        # Create temporary directory for this frame's pipeline
        temp_dir = self.output_dir / "temp_pipeline" / frame_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Run main.py (complete pipeline: SMPLest-X + WiLoR + EMOCA)
            print(f"   🚀 Running complete pipeline...")
            main_cmd = [
                sys.executable, "main.py",
                "--input_image", str(img_path),
                "--output_dir", str(temp_dir)
            ]
            
            result = subprocess.run(main_cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                print(f"   ❌ Pipeline failed: {result.stderr}")
                return None, None
            
            # Find the run directory (main.py creates run_TIMESTAMP)
            run_dirs = list(temp_dir.glob("run_*"))
            if not run_dirs:
                print(f"   ❌ No run directory found")
                return None, None
            
            run_dir = run_dirs[0]  # Take the most recent
            print(f"   📁 Pipeline output: {run_dir.name}")
            
            # Load baseline results (SMPLest-X)
            baseline_result = self._load_baseline_from_pipeline(run_dir)
            if baseline_result is None:
                print(f"   ❌ Failed to load baseline results")
                return None, None
            
            # Step 2: Run coordinate analysis
            print(f"   📐 Running coordinate analysis...")
            coord_cmd = [
                sys.executable, "analysis_tools/coordinate_analyzer_fixed.py",
                str(run_dir)
            ]
            
            result = subprocess.run(coord_cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                print(f"   ⚠️ Coordinate analysis failed: {result.stderr}")
                # Continue without coordinate analysis (fusion may still work)
            
            # Step 3: Run fusion
            print(f"   🔄 Running fusion...")
            fusion_cmd = [
                sys.executable, "fusion/direct_parameter_fusion.py",
                "--results_dir", str(run_dir)
            ]
            
            result = subprocess.run(fusion_cmd, capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                print(f"   ❌ Fusion failed: {result.stderr}")
                # Return baseline only
                return baseline_result, baseline_result
            
            # Load fusion results
            fusion_result = self._load_fusion_from_pipeline(run_dir, baseline_result)
            if fusion_result is None:
                print(f"   ⚠️ Fusion results not found, using baseline")
                return baseline_result, baseline_result
            
            print(f"   ✅ Complete pipeline successful")
            return baseline_result, fusion_result
            
        except Exception as e:
            print(f"   ❌ Pipeline error: {e}")
            return None, None
    
    def _load_baseline_from_pipeline(self, run_dir: Path) -> Optional[Dict]:
        """Load baseline SMPLest-X results from pipeline output"""
        # Look for SMPLest-X results
        smplx_pattern = run_dir / "smplestx_results" / "*" / "person_*" / "smplx_params_*.json"
        smplx_files = list(run_dir.glob("smplestx_results/*/person_*/smplx_params_*.json"))
        
        if not smplx_files:
            return None
        
        # Load the first person's parameters
        with open(smplx_files[0], 'r') as f:
            params = json.load(f)
        
        # Convert lists back to numpy arrays
        result = {
            'mesh': np.array(params['mesh']),
            'joints_3d': np.array(params['joints_3d']),
            'joints_2d': np.array(params['joints_2d']),
            'parameters': {
                'betas': np.array(params['betas']),
                'body_pose': np.array(params['body_pose']),
                'left_hand_pose': np.array(params['left_hand_pose']),
                'right_hand_pose': np.array(params['right_hand_pose']),
                'expression': np.array(params['expression']),
                'root_pose': np.array(params['root_pose']),
                'jaw_pose': np.array(params['jaw_pose']),
                'translation': np.array(params['translation'])
            }
        }
        
        return result
    
    def _load_fusion_from_pipeline(self, run_dir: Path, baseline_result: Dict) -> Optional[Dict]:
        """Load fusion results from pipeline output"""
        fusion_dir = run_dir / "fusion_results"
        
        # Check if fusion was successful
        if not (fusion_dir / "enhanced_mesh.npy").exists():
            return None
        
        # Load enhanced mesh
        enhanced_mesh = np.load(fusion_dir / "enhanced_mesh.npy")
        
        # Load fused parameters if available
        fused_params = baseline_result['parameters'].copy()
        if (fusion_dir / "fused_parameters.json").exists():
            with open(fusion_dir / "fused_parameters.json", 'r') as f:
                fused_params_loaded = json.load(f)
            
            # Convert back to numpy arrays
            for key, value in fused_params_loaded.items():
                if isinstance(value, list) and key != 'fusion_metadata':
                    fused_params[key] = np.array(value)
        
        fusion_result = {
            'mesh': enhanced_mesh,
            'joints_3d': baseline_result['joints_3d'].copy(),  # Keep baseline joints
            'joints_2d': baseline_result['joints_2d'].copy(),
            'parameters': fused_params,
            'fusion_status': 'success'
        }
        
        return fusion_result
    
    def calculate_metrics(self, predicted_mesh: np.ndarray, gt_mesh: np.ndarray, 
                         predicted_joints: np.ndarray = None, gt_joints: np.ndarray = None) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # Mesh-level metrics
        if predicted_mesh is not None and gt_mesh is not None:
            # Per Vertex Error (PVE)
            vertex_errors = np.linalg.norm(predicted_mesh - gt_mesh, axis=1)
            metrics['PVE'] = float(np.mean(vertex_errors) * 1000)  # Convert to mm
            metrics['PVE_std'] = float(np.std(vertex_errors) * 1000)
            
            # Procrustes-aligned PVE
            pred_aligned = self.procrustes_align(predicted_mesh, gt_mesh)
            vertex_errors_aligned = np.linalg.norm(pred_aligned - gt_mesh, axis=1)
            metrics['PA-PVE'] = float(np.mean(vertex_errors_aligned) * 1000)
        
        # Joint-level metrics (if available)
        if predicted_joints is not None and gt_joints is not None:
            # Mean Per Joint Position Error (MPJPE)
            joint_errors = np.linalg.norm(predicted_joints - gt_joints, axis=1)
            metrics['MPJPE'] = float(np.mean(joint_errors) * 1000)
            
            # Procrustes-aligned MPJPE  
            joints_aligned = self.procrustes_align(predicted_joints, gt_joints)
            joint_errors_aligned = np.linalg.norm(joints_aligned - gt_joints, axis=1)
            metrics['PA-MPJPE'] = float(np.mean(joint_errors_aligned) * 1000)
            
            # Hand-specific metrics (assuming SMPL-X joint order)
            # Left hand joints (indices 25-39 in SMPL-X)
            if len(predicted_joints) > 39:
                left_hand_errors = joint_errors[25:40]
                metrics['Left_Hand_MPJPE'] = float(np.mean(left_hand_errors) * 1000)
                
                # Right hand joints (indices 40-54 in SMPL-X)
                right_hand_errors = joint_errors[40:55]
                metrics['Right_Hand_MPJPE'] = float(np.mean(right_hand_errors) * 1000)
                
                # Face joints (indices 55+ in SMPL-X)
                if len(predicted_joints) > 55:
                    face_errors = joint_errors[55:]
                    metrics['Face_MPJPE'] = float(np.mean(face_errors) * 1000)
        
        return metrics
    
    def procrustes_align(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Procrustes alignment for fair comparison"""
        # Center the point sets
        pred_centered = pred - pred.mean(axis=0)
        gt_centered = gt - gt.mean(axis=0)
        
        # Scale to unit variance
        pred_scale = np.sqrt(np.sum(pred_centered**2))
        gt_scale = np.sqrt(np.sum(gt_centered**2))
        
        if pred_scale > 0 and gt_scale > 0:
            pred_normalized = pred_centered / pred_scale
            gt_normalized = gt_centered / gt_scale
            
            # Find optimal rotation using SVD
            H = pred_normalized.T @ gt_normalized
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Apply transformation
            pred_aligned = (pred_normalized @ R) * gt_scale + gt.mean(axis=0)
            return pred_aligned
        else:
            return pred
    
    def evaluate_single_frame(self, frame_id: str) -> Dict:
        """Evaluate both baseline and fusion on a single frame using complete pipeline"""
        print(f"   📸 Loading frame data...")
        
        # Load frame data
        frame_data = self.ehf_dataset.get_frame_data(frame_id)
        gt_mesh = self.ehf_dataset.load_ground_truth_mesh(frame_id)
        print(f"   📐 Ground truth mesh: {gt_mesh.shape[0]} vertices")
        
        # Run complete pipeline (SMPLest-X + WiLoR + EMOCA + coordinate analysis + fusion)
        baseline_result, fusion_result = self.run_complete_pipeline(frame_data)
        
        if baseline_result is None:
            print(f"   ❌ Pipeline failed for frame {frame_id}")
            return None
        
        # Determine fusion status
        if fusion_result is not None and not np.array_equal(baseline_result['mesh'], fusion_result['mesh']):
            fusion_status = 'success'
            print(f"   ✅ Fusion successful - mesh differs from baseline")
        else:
            fusion_status = 'failed_or_identical'
            fusion_result = baseline_result  # Use baseline as fallback
            print(f"   ⚠️ Fusion failed or identical to baseline")
        
        print(f"   📊 Baseline mesh: {baseline_result['mesh'].shape[0]} vertices")
        print(f"   📈 Fusion status: {fusion_status}")
        
        # Calculate metrics
        print(f"   📏 Calculating metrics...")
        baseline_metrics = self.calculate_metrics(
            baseline_result['mesh'], gt_mesh,
            baseline_result['joints_3d'], None  # No GT joints available in EHF
        )
        
        fusion_metrics = self.calculate_metrics(
            fusion_result['mesh'], gt_mesh,
            fusion_result['joints_3d'], None
        )
        
        # Quick comparison
        if 'PVE' in baseline_metrics and 'PVE' in fusion_metrics:
            improvement = ((baseline_metrics['PVE'] - fusion_metrics['PVE']) / baseline_metrics['PVE']) * 100
            print(f"   📊 PVE improvement: {improvement:.2f}%")
        
        return {
            'frame_id': frame_id,
            'baseline_metrics': baseline_metrics,
            'fusion_metrics': fusion_metrics,
            'fusion_status': fusion_status
        }
    
    def run_full_evaluation(self, max_frames: Optional[int] = 10) -> Dict:
        """Run evaluation on EHF frames (default limited to 10 for testing)"""
        frames_to_eval = self.ehf_dataset.frames
        if max_frames:
            frames_to_eval = frames_to_eval[:max_frames]
        
        print(f"Starting evaluation on {len(frames_to_eval)} frames...")
        print(f"Frames to evaluate: {frames_to_eval}")
        
        all_results = []
        baseline_metrics_all = []
        fusion_metrics_all = []
        fusion_status_summary = {'success': 0, 'failed_or_identical': 0, 'error': 0}
        
        for i, frame_id in enumerate(frames_to_eval):
            print(f"\n[{i+1}/{len(frames_to_eval)}] Processing frame: {frame_id}")
            try:
                result = self.evaluate_single_frame(frame_id)
                
                if result is None:
                    print(f"   ❌ Frame {frame_id} evaluation failed")
                    fusion_status_summary['error'] += 1
                    continue
                
                all_results.append(result)
                baseline_metrics_all.append(result['baseline_metrics'])
                fusion_metrics_all.append(result['fusion_metrics'])
                
                # Track fusion status
                status = result['fusion_status']
                if status == 'success':
                    fusion_status_summary['success'] += 1
                elif status == 'failed_or_identical':
                    fusion_status_summary['failed_or_identical'] += 1
                else:
                    fusion_status_summary['error'] += 1
                
                print(f"   ✅ Frame {frame_id} completed")
                
            except Exception as e:
                print(f"   ❌ Error evaluating frame {frame_id}: {e}")
                fusion_status_summary['error'] += 1
                continue
        
        if len(all_results) == 0:
            print(f"\n❌ No frames were successfully evaluated!")
            return {
                'total_frames': 0,
                'fusion_status_summary': fusion_status_summary,
                'error': 'No successful evaluations'
            }
        
        # Aggregate results
        final_results = self.aggregate_results(baseline_metrics_all, fusion_metrics_all)
        final_results['per_frame_results'] = all_results
        final_results['total_frames'] = len(all_results)
        final_results['fusion_status_summary'] = fusion_status_summary
        
        # Save results
        self.save_results(final_results)
        
        # Print summary
        print(f"\n📊 EVALUATION SUMMARY:")
        print(f"   Total frames evaluated: {len(all_results)}")
        print(f"   Fusion success rate: {fusion_status_summary['success']}/{len(all_results)}")
        if fusion_status_summary['failed_or_identical'] > 0:
            print(f"   Fusion failed/identical: {fusion_status_summary['failed_or_identical']}")
        if fusion_status_summary['error'] > 0:
            print(f"   Pipeline errors: {fusion_status_summary['error']}")
        
        return final_results
    
    def aggregate_results(self, baseline_metrics: List[Dict], fusion_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across all frames"""
        def average_metrics(metrics_list):
            if not metrics_list:
                return {}
            
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
        
        # Calculate improvements
        improvements = {}
        for key in baseline_avg:
            if key.endswith('_std'):
                continue
            if key in fusion_avg:
                baseline_val = baseline_avg[key]
                fusion_val = fusion_avg[key]
                if baseline_val > 0:
                    improvement = ((baseline_val - fusion_val) / baseline_val) * 100
                    improvements[f"{key}_improvement_%"] = float(improvement)
        
        return {
            'baseline_average': baseline_avg,
            'fusion_average': fusion_avg,
            'improvements': improvements
        }
    
    def save_results(self, results: Dict):
        """Save evaluation results"""
        # Save full results as JSON
        with open(self.output_dir / "ehf_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create comparison table
        self.create_comparison_table(results)
        
        print(f"✅ Results saved to: {self.output_dir}")
    
    def create_comparison_table(self, results: Dict):
        """Create a simple comparison table"""
        baseline = results['baseline_average']
        fusion = results['fusion_average']
        improvements = results['improvements']
        fusion_summary = results.get('fusion_status_summary', {})
        
        table_content = "EHF EVALUATION RESULTS\n"
        table_content += "=" * 50 + "\n\n"
        
        table_content += f"Total Frames Evaluated: {results['total_frames']}\n"
        if fusion_summary:
            table_content += f"Fusion Success Rate: {fusion_summary.get('success', 0)}/{results['total_frames']}\n"
            if fusion_summary.get('failed_or_identical', 0) > 0:
                table_content += f"Fusion Failed/Identical: {fusion_summary.get('failed_or_identical', 0)}\n"
            if fusion_summary.get('error', 0) > 0:
                table_content += f"Pipeline Errors: {fusion_summary.get('error', 0)}\n"
        table_content += "\n"
        
        table_content += "METRIC COMPARISON\n"
        table_content += "-" * 30 + "\n"
        table_content += f"{'Metric':<20} {'Baseline':<12} {'Fusion':<12} {'Improvement':<12}\n"
        table_content += "-" * 56 + "\n"
        
        key_metrics = ['PVE', 'PA-PVE', 'MPJPE', 'PA-MPJPE', 'Left_Hand_MPJPE', 'Right_Hand_MPJPE', 'Face_MPJPE']
        
        for metric in key_metrics:
            if metric in baseline and metric in fusion:
                baseline_val = baseline[metric]
                fusion_val = fusion[metric]
                improvement_key = f"{metric}_improvement_%"
                improvement = improvements.get(improvement_key, 0)
                
                table_content += f"{metric:<20} {baseline_val:<12.2f} {fusion_val:<12.2f} {improvement:<12.2f}%\n"
        
        table_content += "\nNOTE: Lower values are better for all metrics (errors in mm)\n"
        table_content += "Positive improvement % means fusion is better than baseline\n"
        
        if fusion_summary.get('success', 0) < results['total_frames']:
            table_content += f"\n⚠️  Some frames used baseline fallback due to fusion issues\n"
        
        # Save table
        with open(self.output_dir / "comparison_table.txt", 'w') as f:
            f.write(table_content)
        
        print("\n" + table_content)

def main():
    parser = argparse.ArgumentParser(description='EHF Fusion Evaluation')
    parser.add_argument('--ehf_path', type=str, default='data/EHF',
                       help='Path to EHF dataset')
    parser.add_argument('--config', type=str, default='pretrained_models/smplest_x/config_base.py',
                       help='Path to SMPLest-X config')
    parser.add_argument('--max_frames', type=int, default=10,
                       help='Maximum number of frames to evaluate (default: 10 for testing)')
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = EHFFusionEvaluator(args.ehf_path, args.config)
    results = evaluator.run_full_evaluation(args.max_frames)
    
    print(f"\n✅ Evaluation complete! Results saved to: {evaluator.output_dir}")

if __name__ == '__main__':
    main()