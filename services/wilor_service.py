#!/usr/bin/env python3
"""
Fixed WiLoR Persistence Service
Matches exactly the working wilor_adapter.py structure and working directory requirements
"""

import os
import sys
import cv2
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Critical: Change to adapters directory (WiLoR MUST run from adapters/)
service_file = Path(__file__).resolve()
adapters_dir = service_file.parent.parent / 'adapters'  # Go up from services/ to project/, then to adapters/
os.chdir(adapters_dir)
print(f"🔧 Changed working directory to: {os.getcwd()}")

# Add project paths (exactly like working wilor_adapter.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'WiLoR'))

from scripts.wilor_output_extractor import WiLoRParameterExtractor, save_wilor_parameters_json
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO

# Constants
LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

# Initialize FastAPI app
app = FastAPI(title="WiLoR Persistence Service", version="1.0.0")

# Global model instances (loaded once at startup)
model = None
model_cfg = None
detector = None
renderer = None
extractor = None
device = None

# Request/Response models
class PredictRequest(BaseModel):
    image_path: str
    output_dir: str
    rescale_factor: float = 2.0
    save_mesh: bool = False

class PredictResponse(BaseModel):
    success: bool
    message: str
    processing_time: float
    num_hands: int
    output_files: List[str] = []

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cuda_available: bool
    gpu_memory_gb: float

def load_models():
    """Load all WiLoR models - exact paths from working adapter"""
    global model, model_cfg, detector, renderer, extractor, device
    
    print("🚀 Loading WiLoR models...")
    start_time = time.time()
    
    try:
        # Set device
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"✅ Using device: {device}")
        
        # Load WiLoR model (exact paths from working adapter)
        checkpoint_path = '../external/WiLoR/pretrained_models/wilor_final.ckpt'
        cfg_path = '../external/WiLoR/pretrained_models/model_config.yaml'
        
        model, model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path)
        model = model.to(device)
        model.eval()
        print("✅ WiLoR model loaded")
        
        # Load YOLO detector (exact path from working adapter)
        detector_path = '../external/WiLoR/pretrained_models/detector.pt'
        detector = YOLO(detector_path)
        detector = detector.to(device)
        print("✅ YOLO hand detector loaded")
        
        # Setup renderer (exactly like working adapter)
        renderer = Renderer(model_cfg, faces=model.mano.faces)
        print("✅ WiLoR renderer initialized")
        
        # Create parameter extractor (exactly like working adapter)
        extractor = WiLoRParameterExtractor(model_cfg)
        print("✅ Parameter extractor ready")
        
        load_time = time.time() - start_time
        print(f"🎉 All WiLoR models loaded successfully in {load_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def project_full_img(points, cam_trans, focal_length, img_res):
    """Project 3D points to 2D image coordinates - exactly like working adapter"""
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3)
    K[0, 0] = focal_length
    K[1, 1] = focal_length
    K[0, 2] = camera_center[0]
    K[1, 2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:]
    V_2d = (K @ points.T).T
    return V_2d[..., :-1]

def process_single_image(image_path: str, output_dir: str, rescale_factor: float, save_mesh: bool) -> Dict[str, Any]:
    """Process single image - EXACT copy of working adapter logic"""
    global model, model_cfg, detector, renderer, extractor, device
    
    if not all([model, model_cfg, detector, renderer, extractor]):
        raise RuntimeError("Models not loaded. Call load_models() first.")
    
    start_time = time.time()
    
    # Create output directory
    output_folder = Path(output_dir)
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")
    
    # Load image (exactly like working adapter)
    img_cv2 = cv2.imread(str(image_path))
    if img_cv2 is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Hand detection with YOLO (exactly like working adapter)
    detections = detector(img_cv2, conf=0.3, verbose=False)[0]
    bboxes = []
    is_right = []
    
    for det in detections:
        bbox = det.boxes.data.cpu().detach().squeeze().numpy()
        is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
        bboxes.append(bbox[:4].tolist())
    
    if len(bboxes) == 0:
        # No hands detected
        processing_time = time.time() - start_time
        return {
            "success": True,
            "message": "No hands detected in image",
            "processing_time": processing_time,
            "num_hands": 0,
            "output_files": []
        }
    
    # Prepare data for WiLoR inference (exactly like working adapter)
    boxes = np.stack(bboxes)
    right = np.stack(is_right)
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Storage for results (exactly like working adapter)
    all_verts = []
    all_cam_t = []
    all_right = []
    all_joints = []
    all_kpts = []
    
    # Store batch and output data for JSON export (exactly like working adapter)
    export_batches = []
    export_outputs = []
    
    # Process batches (exactly like working adapter)
    for batch in dataloader:
        batch = recursive_to(batch, device)
        
        with torch.no_grad():
            out = model(batch)
        
        multiplier = (2 * batch['right'] - 1)
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
        
        # Store for JSON export (exactly like working adapter)
        export_batches.append({
            'img': batch['img'],
            'right': batch['right'],
            'box_center': batch['box_center'],
            'box_size': batch['box_size'],
            'img_size': batch['img_size']
        })
        
        export_outputs.append({
            'pred_vertices': out['pred_vertices'],
            'pred_keypoints_3d': out['pred_keypoints_3d'],
            'pred_cam': out['pred_cam'],
            'pred_cam_t_full': pred_cam_t_full,
            'pred_mano_params': out.get('pred_mano_params', {})
        })
        
        # Process each hand in the batch (exactly like working adapter)
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            verts = out['pred_vertices'][n].detach().cpu().numpy()
            joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
            
            is_right_hand = batch['right'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right_hand - 1) * verts[:, 0]
            joints[:, 0] = (2 * is_right_hand - 1) * joints[:, 0]
            cam_t = pred_cam_t_full[n]
            kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
            
            all_verts.append(verts)
            all_cam_t.append(cam_t)
            all_right.append(is_right_hand)
            all_joints.append(joints)
            all_kpts.append(kpts_2d)
            
            # Save mesh if requested (exactly like working adapter)
            if save_mesh:
                img_fn = Path(image_path).stem
                camera_translation = cam_t.copy()
                tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right_hand)
                mesh_path = output_folder / f'{img_fn}_{n}.obj'
                tmesh.export(str(mesh_path))
    
    output_files = []
    
    # Export comprehensive JSON parameters (exactly like working adapter)
    if len(all_verts) > 0:
        # Combine all batches for comprehensive export (exactly like working adapter)
        combined_batch = {
            'img': torch.cat([b['img'] for b in export_batches], dim=0),
            'right': torch.cat([b['right'] for b in export_batches], dim=0),
            'box_center': torch.cat([b['box_center'] for b in export_batches], dim=0),
            'box_size': torch.cat([b['box_size'] for b in export_batches], dim=0),
            'img_size': torch.cat([b['img_size'] for b in export_batches], dim=0)
        }
        
        # Combine outputs with MANO parameters (exactly like working adapter)
        combined_output = {
            'pred_vertices': torch.cat([o['pred_vertices'] for o in export_outputs], dim=0),
            'pred_keypoints_3d': torch.cat([o['pred_keypoints_3d'] for o in export_outputs], dim=0),
            'pred_cam': torch.cat([o['pred_cam'] for o in export_outputs], dim=0)
        }
        
        # Add MANO parameters if available (exactly like working adapter)
        if export_outputs and 'pred_mano_params' in export_outputs[0]:
            mano_params = {}
            for key in export_outputs[0]['pred_mano_params'].keys():
                mano_params[key] = torch.cat([o['pred_mano_params'][key] for o in export_outputs], dim=0)
            combined_output['pred_mano_params'] = mano_params
        
        combined_cam_t = np.concatenate([o['pred_cam_t_full'] for o in export_outputs], axis=0)
        
        # Get filename from image path (exactly like working adapter)
        img_fn = Path(image_path).stem
        
        # Save comprehensive parameters JSON (exactly like working adapter)
        json_path = output_folder / f'{img_fn}_parameters.json'
        save_wilor_parameters_json(extractor, combined_batch, combined_output,
                                 combined_cam_t, image_path, scaled_focal_length, str(json_path))
        output_files.append(str(json_path))
        
        # Render and save visualization (exactly like working adapter)
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length,
        )
        
        img_size_for_render = combined_batch['img_size'][0]  # Use first image size
        cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, 
                                                render_res=img_size_for_render, 
                                                is_right=all_right, **misc_args)
        
        # Overlay on input image (exactly like working adapter)
        input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
        input_img = np.concatenate([input_img, np.ones_like(input_img[:, :, :1])], axis=2)
        input_img_overlay = input_img[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
        
        # Save rendered result (exactly like working adapter)
        render_path = output_folder / f'{img_fn}.jpg'
        cv2.imwrite(str(render_path), 255 * input_img_overlay[:, :, ::-1])
        output_files.append(str(render_path))
    
    processing_time = time.time() - start_time
    num_hands = len(all_verts)
    
    return {
        "success": True,
        "message": f"Successfully processed {num_hands} hand(s)",
        "processing_time": processing_time,
        "num_hands": num_hands,
        "output_files": output_files
    }

@app.on_event("startup")
async def startup_event():
    """Load models when service starts"""
    try:
        success = load_models()
        if success:
            print("🎉 WiLoR Service ready!")
        else:
            print("❌ WiLoR Service failed to load models")
            raise RuntimeError("Model loading failed")
    except Exception as e:
        print(f"❌ Failed to start WiLoR service: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = all([model, model_cfg, detector, renderer, extractor])
    cuda_available = torch.cuda.is_available()
    gpu_memory_gb = 0.0
    
    if cuda_available:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        cuda_available=cuda_available,
        gpu_memory_gb=gpu_memory_gb
    )

@app.get("/status")
async def get_status():
    """Get detailed service status"""
    return {
        "service": "WiLoR Persistence Service",
        "version": "1.0.0",
        "models_loaded": all([model, model_cfg, detector, renderer, extractor]),
        "cuda_available": torch.cuda.is_available(),
        "working_directory": str(Path.cwd()),
        "model_details": {
            "wilor_loaded": model is not None,
            "detector_loaded": detector is not None,
            "renderer_loaded": renderer is not None,
            "extractor_loaded": extractor is not None
        }
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Main inference endpoint"""
    try:
        result = process_single_image(
            image_path=request.image_path,
            output_dir=request.output_dir,
            rescale_factor=request.rescale_factor,
            save_mesh=request.save_mesh
        )
        
        return PredictResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting WiLoR Persistence Service on port 8002...")
    uvicorn.run(app, host="0.0.0.0", port=8002)