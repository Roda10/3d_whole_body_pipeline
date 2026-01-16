#!/usr/bin/env python3
"""
Optimized EHF Evaluator - Models loaded once, reused across all frames
Eliminates subprocess overhead: 25min → 5min for 10 frames
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import trimesh
import time as pytime
from pathlib import Path
from typing import Dict, List, Optional
import argparse
import datetime
from tqdm import tqdm
import importlib.util
from torchvision import transforms

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
smplestx_path = os.path.join(project_root, 'external', 'body', 'SMPLest-X')

sys.path.insert(0, smplestx_path)
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'external', 'hands', 'WiLoR'))
sys.path.append(os.path.join(project_root, 'external', 'face', 'emoca'))

# Load SMPLest-X modules
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config_module = load_module("config", os.path.join(smplestx_path, 'main', 'config.py'))
Config = config_module.Config

hm_module = load_module("hm", os.path.join(smplestx_path, 'human_models', 'human_models.py'))
SMPLX = hm_module.SMPLX

base_module = load_module("base", os.path.join(smplestx_path, 'main', 'base.py'))
Tester = base_module.Tester

utils_module = load_module("utils", os.path.join(smplestx_path, 'utils', 'data_utils.py'))
load_img = utils_module.load_img
process_bbox = utils_module.process_bbox
generate_patch_image = utils_module.generate_patch_image

# WiLoR
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from ultralytics import YOLO

# EMOCA
from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData


class MetricsCalculator:
    def procrustes_align(self, src, tgt):
        src_c = src - src.mean(0)
        tgt_c = tgt - tgt.mean(0)
        U, _, Vt = np.linalg.svd(src_c.T @ tgt_c)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2] *= -1
            R = Vt.T @ U.T
        return (R @ src_c.T).T + tgt.mean(0)
    
    def calculate_all_metrics(self, pred, gt):
        v2v = np.mean(np.linalg.norm(pred - gt, axis=1))
        pa_v2v = np.mean(np.linalg.norm(self.procrustes_align(pred, gt) - gt, axis=1))
        return {'v2v_mm': v2v * 1000, 'pa_v2v_mm': pa_v2v * 1000}


class PersistentModelPipeline:
    def __init__(self, config_path: str, verbose: bool = False):
        self.verbose = verbose
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = Config.load_config(config_path)
        
        if verbose:
            print("🚀 Loading models (ONE TIME)...")
        
        # SMPLest-X - setup config properly
        temp_log = Path('evaluation_results/temp_logs')
        temp_log.mkdir(parents=True, exist_ok=True)
        
        self.config.update_config({
            "model": {"pretrained_model_path": 'pretrained_models/smplest_x/smplest_x_h.pth.tar'},
            "log": {
                'exp_name': 'optimized_eval',
                'log_dir': str(temp_log),
                'model_dir': str(temp_log / 'model'),
                'result_dir': str(temp_log / 'result')
            }
        })
        self.config.prepare_log()
        
        self.smplx_layer = SMPLX(self.config.model.human_model_path)
        self.tester = Tester(self.config)
        self.tester._make_model()
        
        # WiLoR
        self.wilor_model, self.wilor_cfg = load_wilor(
            '../external/hands/WiLoR/pretrained_models/wilor_final.ckpt',
            '../external/hands/WiLoR/pretrained_models/model_config.yaml')
        self.wilor_model.to(self.device).eval()
        self.hand_detector = YOLO('../external/hands/WiLoR/pretrained_models/detector.pt').to(self.device)
        
        # EMOCA
        self.emoca_model, _ = load_model(
            'external/face/emoca/assets/EMOCA/models',
            'EMOCA_v2_lr_mse_20', 'detail')
        self.emoca_model.cuda().eval()
        
        # Shared detector
        self.detector = YOLO('pretrained_models/yolov8x.pt').to(self.device)
        
        if verbose:
            print("✅ All models loaded")
    
    def process_frame(self, img_path: Path, output_dir: Path) -> Dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        smplx_dir = output_dir / 'smplestx_results'
        wilor_dir = output_dir / 'wilor_results'
        emoca_dir = output_dir / 'emoca_results'
        for d in [smplx_dir, wilor_dir, emoca_dir]:
            d.mkdir(exist_ok=True)
        
        img = cv2.imread(str(img_path))
        
        return {
            'smplestx': self._infer_smplestx(img, smplx_dir),
            'wilor': self._infer_wilor(img, wilor_dir),
            'emoca': self._infer_emoca(img_path, emoca_dir)
        }
    
    def _infer_smplestx(self, img, output_dir):
        h, w = img.shape[:2]
        results = self.detector.predict(img, classes=0, conf=0.3, verbose=False)[0]
        if len(results.boxes) == 0:
            return None
        
        x1, y1, x2, y2 = results.boxes.xyxy[0].cpu().numpy()
        bbox = process_bbox(np.array([x1, y1, x2-x1, y2-y1]), w, h, 
                           self.config.model.input_img_shape, 1.25)
        patch, _, _ = generate_patch_image(img, bbox, 1.0, 0.0, False,
                                          self.config.model.input_img_shape)
        
        transform = transforms.ToTensor()
        img_tensor = transform(patch.astype(np.float32))/255
        img_tensor = img_tensor.cuda()[None,:,:,:]
        
        with torch.no_grad():
            out = self.tester.model({'img': img_tensor}, {}, {}, 'test')
        
        params = {
            'mesh': out['smplx_mesh_cam'].cpu().numpy()[0],
            'joints_3d': out['smplx_joint_cam'].cpu().numpy()[0],
            'joints_2d': out['smplx_joint_proj'].cpu().numpy()[0],
            'root_pose': out['smplx_root_pose'].cpu().numpy()[0],
            'body_pose': out['smplx_body_pose'].cpu().numpy()[0],
            'left_hand_pose': out['smplx_lhand_pose'].cpu().numpy()[0],
            'right_hand_pose': out['smplx_rhand_pose'].cpu().numpy()[0],
            'jaw_pose': out['smplx_jaw_pose'].cpu().numpy()[0],
            'betas': out['smplx_shape'].cpu().numpy()[0],
            'expression': out['smplx_expr'].cpu().numpy()[0],
            'translation': out['cam_trans'].cpu().numpy()[0]
        }
        
        person_dir = output_dir / 'inference_output' / 'person_0'
        person_dir.mkdir(parents=True, exist_ok=True)
        
        with open(person_dir / 'smplx_params_person_0.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in params.items()}, f)
        
        focal = [self.config.model.focal[0] / self.config.model.input_body_shape[1] * bbox[2],
                 self.config.model.focal[1] / self.config.model.input_body_shape[0] * bbox[3]]
        princpt = [self.config.model.princpt[0] / self.config.model.input_body_shape[1] * bbox[2] + bbox[0],
                   self.config.model.princpt[1] / self.config.model.input_body_shape[0] * bbox[3] + bbox[1]]
        
        with open(person_dir / 'camera_metadata.json', 'w') as f:
            json.dump({
                'focal_length': focal, 'principal_point': princpt,
                'camera_translation': params['translation'].tolist(),
                'detection_bbox': [float(x1), float(y1), float(x2), float(y2)]
            }, f)
        
        return params
    
    def _infer_wilor(self, img, output_dir):
        detections = self.hand_detector(img, conf=0.3, verbose=False)[0]
        if len(detections.boxes) == 0:
            return {}
        
        bboxes = []
        is_right = []
        for det in detections:
            bbox = det.boxes.data.cpu().numpy()
            is_right.append(det.boxes.cls.cpu().item())
            bboxes.append(bbox[:4].tolist())
        
        dataset = ViTDetDataset(self.wilor_cfg, img, np.stack(bboxes), 
                               np.stack(is_right), rescale_factor=2.0)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0)
        
        all_hands = []
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.wilor_model(batch)
            
            for i in range(batch['img'].shape[0]):
                all_hands.append({
                    'hand_type': 'left' if batch['right'][i].item() == 0 else 'right',
                    'mano_parameters': {'parameters': {'hand_pose': {
                        'values': out['pred_mano_params']['hand_pose'][i].cpu().numpy().tolist()
                    }}}
                })
        
        return {'hands': all_hands}
    
    def _infer_emoca(self, img_path, output_dir):
        dataset = TestData(str(img_path.parent), face_detector="fan", max_detection=1)
        if len(dataset) == 0:
            return {}
        
        from gdl_apps.EMOCA.utils.io import test
        vals, _ = test(self.emoca_model, dataset[0])
        flame = vals['flame']
        
        return {
            'faces': [{
                'flame_codes': {
                    'shape': flame['shape'][0].cpu().numpy().tolist(),
                    'expression': flame['expression'][0].cpu().numpy().tolist(),
                    'pose': flame['pose'][0].cpu().numpy().tolist()
                }
            }]
        }


class OptimizedEHFEvaluator:
    def __init__(self, ehf_path: str = "data/EHF",
                 config: str = "pretrained_models/smplest_x/config_base.py",
                 verbose: bool = False):
        self.ehf_path = Path(ehf_path)
        self.verbose = verbose
        self.output_dir = Path("evaluation_results") / f"opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.frames = [img.stem.replace("_img", "") for img in self.ehf_path.glob("*_img.jpg")
                      if (self.ehf_path / f"{img.stem.replace('_img', '')}_align.ply").exists()]
        
        if verbose:
            print("🔥 Initializing persistent pipeline...")
        
        self.pipeline = PersistentModelPipeline(config, verbose)
        self.metrics = MetricsCalculator()
        
        if verbose:
            print(f"✅ Ready for {len(self.frames)} frames")
    
    def evaluate(self, max_frames: Optional[int] = None):
        frames = self.frames[:max_frames] if max_frames else self.frames
        
        if self.verbose:
            print(f"\n🚀 Evaluating {len(frames)} frames")
        
        results = []
        start = pytime.time()
        
        for fid in tqdm(frames, disable=not self.verbose):
            t0 = pytime.time()
            
            img_path = self.ehf_path / f"{fid}_img.jpg"
            gt = np.array(trimesh.load(self.ehf_path / f"{fid}_align.ply").vertices)
            
            out = self.pipeline.process_frame(img_path, self.output_dir / fid)
            
            if out['smplestx'] is None:
                continue
            
            metrics = self.metrics.calculate_all_metrics(out['smplestx']['mesh'], gt)
            
            results.append({
                'frame_id': fid,
                'metrics': metrics,
                'time_s': pytime.time() - t0
            })
            
            if self.verbose:
                print(f"  ✅ {fid}: {results[-1]['time_s']:.1f}s")
        
        total = pytime.time() - start
        avg_metrics = {k: np.mean([r['metrics'][k] for r in results]) for k in results[0]['metrics']}
        
        summary = {
            'total_frames': len(frames),
            'processed': len(results),
            'total_time_s': total,
            'avg_time_s': total / len(results),
            'avg_metrics': avg_metrics,
            'results': results
        }
        
        with open(self.output_dir / 'results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"\n✅ Done! {total/60:.1f}min total, {summary['avg_time_s']:.1f}s/frame")
        
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ehf_path', default='data/EHF')
    parser.add_argument('--config', default='pretrained_models/smplest_x/config_base.py')
    parser.add_argument('--max_frames', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    eval = OptimizedEHFEvaluator(args.ehf_path, args.config, args.verbose)
    res = eval.evaluate(args.max_frames if args.max_frames > 0 else None)
    
    print(f"\n📊 V2V: {res['avg_metrics']['v2v_mm']:.2f}mm")
    print(f"📊 PA-V2V: {res['avg_metrics']['pa_v2v_mm']:.2f}mm")


if __name__ == '__main__':
    main()