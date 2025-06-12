#!/usr/bin/env python3
"""
Fixed SMPLest-X Persistence Service
Matches exactly the working smplestx_adapter.py structure
"""

import os
import sys
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import json
import cv2
import datetime
import time
from pathlib import Path
from typing import Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Change to project root directory (services/ is subfolder)
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Add SMPLest-X to path for imports (exactly like working adapter)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))

from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression

# Initialize FastAPI app
app = FastAPI(title="SMPLest-X Persistence Service", version="1.0.0")

# Global model instances (loaded once at startup)
smpl_x = None
demoer = None
detector = None
cfg = None
transform = None

# Request/Response models
class PredictRequest(BaseModel):
    image_path: str
    output_dir: str
    multi_person: bool = True
    cfg_path: str = "pretrained_models/smplest_x/config_base.py"

class PredictResponse(BaseModel):
    success: bool
    message: str
    processing_time: float
    num_persons: int
    output_files: list = []

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cuda_available: bool
    gpu_memory_gb: float

def numpy_to_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: numpy_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_serializable(item) for item in obj]
    else:
        return obj

def load_models(cfg_path: str = "pretrained_models/smplest_x/config_base.py"):
    """Load all models once at service startup - matches working adapter exactly"""
    global smpl_x, demoer, detector, cfg, transform
    
    print("🚀 Loading SMPLest-X models...")
    start_time = time.time()
    
    try:
        cudnn.benchmark = True
        
        # Load config (exactly like working adapter)
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        cfg = Config.load_config(cfg_path)
        
        # Use exact checkpoint path from working adapter
        checkpoint_path = '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/pretrained_models/smplest_x/smplest_x_h.pth.tar'
        
        # Output folder for temp logging (services don't need persistent logs)
        temp_output_dir = Path(f'./temp_service_logs/smplestx_{time_str}')
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        
        new_config = {
            "model": {
                "pretrained_model_path": checkpoint_path,
            },
            "log": {
                'exp_name': f'smplestx_service_{time_str}',
                'log_dir': str(temp_output_dir / 'log'),
            }
        }
        cfg.update_config(new_config)
        cfg.prepare_log()
        
        # Load human models (exactly like working adapter)
        smpl_x = SMPLX(cfg.model.human_model_path) 
        print("✅ SMPL-X model loaded")
        
        # Load SMPLest-X tester (exactly like working adapter)
        demoer = Tester(cfg)
        demoer._make_model()
        print("✅ SMPLest-X tester loaded")
        
        # Load detector (exactly like working adapter)
        bbox_model = getattr(cfg.inference.detection, "model_path", 
                            './pretrained_models/yolov8x.pt')
        detector = YOLO(bbox_model)
        print("✅ YOLO detector loaded")
        
        # Setup transform (exactly like working adapter)
        transform = transforms.ToTensor()
        print("✅ Transform initialized")
        
        load_time = time.time() - start_time
        print(f"🎉 All models loaded successfully in {load_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_single_image(image_path: str, output_dir: str, multi_person: bool) -> Dict[str, Any]:
    """Process single image - EXACT copy of working adapter logic"""
    global smpl_x, demoer, detector, cfg, transform
    
    if not all([smpl_x, demoer, detector, cfg, transform]):
        raise RuntimeError("Models not loaded. Call load_models() first.")
    
    start_time = time.time()
    
    # Create timestamped output folder (exactly like working adapter)
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = Path(output_dir) / f'inference_output_{time_str}'
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")
    
    # EXACT working adapter logic from here:
    original_img = load_img(image_path)
    vis_img = original_img.copy()
    original_img_height, original_img_width = original_img.shape[:2]
    
    # detection, xyxy (exactly like working adapter)
    yolo_results = detector.predict(original_img, 
                                device='cuda', 
                                classes=0, # 'person' class in COCO
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0] # Get the first (and only) result object

    # Extract bounding boxes and confidence scores (exactly like working adapter)
    yolo_bbox_xyxy = yolo_results.boxes.xyxy.detach().cpu().numpy()
    yolo_conf = yolo_results.boxes.conf.detach().cpu().numpy()

    # Combine xyxy and confidence scores for NMS (exactly like working adapter)
    yolo_bbox = np.concatenate((yolo_bbox_xyxy, yolo_conf[:, None]), axis=1)
    num_bbox = len(yolo_bbox)

    if num_bbox < 1:
        # Save original image with no detections (exactly like working adapter)
        output_filename = f"no_bbox_{os.path.basename(image_path)}"
        cv2.imwrite(os.path.join(output_folder, output_filename), vis_img[:, :, ::-1])
        
        processing_time = time.time() - start_time
        return {
            "success": True,
            "message": f"No persons detected. Original image saved.",
            "processing_time": processing_time,
            "num_persons": 0,
            "output_files": [str(output_folder / output_filename)]
        }

    # Handle multi-person vs single person (exactly like working adapter)
    if not multi_person:
        yolo_bbox = yolo_bbox[:1] 
        num_bbox = 1
    else:
        yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
        num_bbox = len(yolo_bbox)

    output_files = []
    
    # Loop all detected bboxes (exactly like working adapter)
    for bbox_id in range(num_bbox):
        if bbox_id >= len(yolo_bbox):
            continue
            
        # Get the bbox in [x1, y1, x2, y2, confidence] format (exactly like working adapter)
        current_yolo_bbox_with_conf = yolo_bbox[bbox_id]
        x1, y1, x2, y2 = current_yolo_bbox_with_conf[:4]
        yolo_bbox_xywh = np.array([x1, y1, x2 - x1, y2 - y1]) # Convert to xywh for process_bbox

        # Process bbox (exactly like working adapter)
        bbox = process_bbox(bbox=yolo_bbox_xywh, 
                            img_width=original_img_width, 
                            img_height=original_img_height, 
                            input_img_shape=cfg.model.input_img_shape, 
                            ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
        img, _, _ = generate_patch_image(cvimg=original_img, 
                                            bbox=bbox, 
                                            scale=1.0, 
                                            rot=0.0, 
                                            do_flip=False, 
                                            out_shape=cfg.model.input_img_shape)
            
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]
        inputs = {'img': img}
        targets = {}
        meta_info = {}

        # mesh recovery (exactly like working adapter)
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

        # Extract SMPL-X parameters (exactly like working adapter)
        smplx_params = {
            # Joints
            'joints_3d': out['smplx_joint_cam'].detach().cpu().numpy()[0],  # 3D joints (camera coordinates)
            'joints_2d': out['smplx_joint_proj'].detach().cpu().numpy()[0],    # 2D projected joints

            # Pose parameters (axis-angle format)
            'root_pose': out['smplx_root_pose'].detach().cpu().numpy()[0],     # Global orientation
            'body_pose': out['smplx_body_pose'].detach().cpu().numpy()[0],     # Body pose (21 joints)
            'left_hand_pose': out['smplx_lhand_pose'].detach().cpu().numpy()[0],  # Left hand pose (15 joints)
            'right_hand_pose': out['smplx_rhand_pose'].detach().cpu().numpy()[0], # Right hand pose (15 joints)
            'jaw_pose': out['smplx_jaw_pose'].detach().cpu().numpy()[0],       # Jaw rotation

            # Shape and expression
            'betas': out['smplx_shape'].detach().cpu().numpy()[0],             # Body shape parameters (10-dim)
            'expression': out['smplx_expr'].detach().cpu().numpy()[0],         # Facial expression (10-dim)

            # Camera
            'translation': out['cam_trans'].detach().cpu().numpy()[0],         # Mesh translation (x,y,z)
            
            # The actual mesh vertices!
            'mesh': out['smplx_mesh_cam'].detach().cpu().numpy()[0],           # Full body mesh vertices (10475 vertices)
        }
        
        # Create person-specific output folder (exactly like working adapter)
        person_output_folder = output_folder / f'person_{bbox_id}'
        os.makedirs(person_output_folder, exist_ok=True)

        # Save parameters as JSON (exactly like working adapter)
        serializable_params = numpy_to_serializable(smplx_params)
        params_filename = person_output_folder / f'smplx_params_person_{bbox_id}.json'
        with open(params_filename, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        output_files.append(str(params_filename))

        # Save camera metadata (exactly like working adapter)
        focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
        princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
        
        camera_meta = {
            'focal_length': focal,
            'principal_point': princpt,
            'camera_translation': out['cam_trans'].detach().cpu().numpy()[0].tolist(),
            'detection_bbox': [float(x1), float(y1), float(x2), float(y2)]
        }
        
        meta_path = person_output_folder / "camera_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(camera_meta, f, indent=2)
        output_files.append(str(meta_path))

        # Draw bbox and render mesh (exactly like working adapter)
        vis_img = cv2.rectangle(vis_img, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (0, 255, 0), 1)
        
        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, 
                            {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

    # Save rendered image (exactly like working adapter)
    output_filename = f"rendered_{os.path.basename(image_path)}"
    rendered_path = output_folder / output_filename
    cv2.imwrite(str(rendered_path), vis_img[:, :, ::-1])
    output_files.append(str(rendered_path))
    
    processing_time = time.time() - start_time
    
    return {
        "success": True,
        "message": f"Successfully processed {num_bbox} person(s)",
        "processing_time": processing_time,
        "num_persons": num_bbox,
        "output_files": output_files
    }

@app.on_event("startup")
async def startup_event():
    """Load models when service starts"""
    try:
        success = load_models()
        if success:
            print("🎉 SMPLest-X Service ready!")
        else:
            print("❌ SMPLest-X Service failed to load models")
            raise RuntimeError("Model loading failed")
    except Exception as e:
        print(f"❌ Failed to start SMPLest-X service: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = all([smpl_x, demoer, detector, cfg, transform])
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
        "service": "SMPLest-X Persistence Service",
        "version": "1.0.0",
        "models_loaded": all([smpl_x, demoer, detector, cfg, transform]),
        "cuda_available": torch.cuda.is_available(),
        "working_directory": str(Path.cwd()),
        "model_details": {
            "smplx_loaded": smpl_x is not None,
            "tester_loaded": demoer is not None,
            "detector_loaded": detector is not None,
            "config_loaded": cfg is not None,
            "transform_ready": transform is not None
        }
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Main inference endpoint"""
    try:
        result = process_single_image(
            image_path=request.image_path,
            output_dir=request.output_dir,
            multi_person=request.multi_person
        )
        
        return PredictResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting SMPLest-X Persistence Service on port 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)