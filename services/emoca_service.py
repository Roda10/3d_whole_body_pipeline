#!/usr/bin/env python3
"""
Fixed EMOCA Persistence Service
Resolves the save_images naming conflict
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add EMOCA paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'emoca'))

from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
# FIX: Rename save_images import to avoid conflict with parameter
from gdl_apps.EMOCA.utils.io import save_obj, save_codes, test
from gdl_apps.EMOCA.utils.io import save_images as save_emoca_images

# Initialize FastAPI app
app = FastAPI(title="EMOCA Persistence Service", version="1.0.0")

# Global model instances (loaded once at startup)
emoca = None
conf = None
device = None

# Request/Response models
class PredictRequest(BaseModel):
    image_path: str
    output_dir: str
    model_name: str = 'EMOCA_v2_lr_mse_20'
    mode: str = 'detail'
    save_images: bool = True
    save_codes: bool = True
    save_mesh: bool = False

class PredictResponse(BaseModel):
    success: bool
    message: str
    processing_time: float
    num_faces: int
    output_files: List[str] = []

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    cuda_available: bool
    gpu_memory_gb: float

def load_models(model_name: str = 'EMOCA_v2_lr_mse_20', mode: str = 'detail'):
    """Load EMOCA model once at service startup"""
    global emoca, conf, device
    
    print("🚀 Loading EMOCA models...")
    start_time = time.time()
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"✅ Using device: {device}")
    
    # Set path to models
    path_to_models = str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models")
    
    # Load EMOCA model
    emoca, conf = load_model(path_to_models, model_name, mode)
    emoca.cuda()
    emoca.eval()
    print(f"✅ EMOCA model '{model_name}' loaded in '{mode}' mode")
    
    load_time = time.time() - start_time
    print(f"🎉 EMOCA model loaded successfully in {load_time:.2f} seconds")
    return True

def process_single_image(image_path: str, output_dir: str, should_save_images: bool, 
                        should_save_codes: bool, should_save_mesh: bool) -> Dict[str, Any]:
    """Process single image using loaded EMOCA model"""
    global emoca, conf, device
    
    if emoca is None:
        raise RuntimeError("EMOCA model not loaded. Call load_models() first.")
    
    start_time = time.time()
    
    # Create output directory
    output_folder = Path(output_dir)
    os.makedirs(output_folder, exist_ok=True)
    
    # Check if input image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found at {image_path}")
    
    # Create temporary folder for EMOCA input (it expects a folder)
    temp_input_folder = output_folder / "temp_input"
    temp_input_folder.mkdir(exist_ok=True)
    
    try:
        # Copy image to temp folder
        temp_image_path = temp_input_folder / Path(image_path).name
        shutil.copy2(image_path, temp_image_path)
        
        # Create dataset for EMOCA
        dataset = TestData(str(temp_input_folder), face_detector="fan", max_detection=20)
        
        if len(dataset) == 0:
            # No faces detected
            processing_time = time.time() - start_time
            return {
                "success": True,
                "message": "No faces detected in image",
                "processing_time": processing_time,
                "num_faces": 0,
                "output_files": []
            }
        
        output_files = []
        total_faces_processed = 0
        
        # Process all detections in the dataset
        for i in range(len(dataset)):
            batch = dataset[i]
            vals, visdict = test(emoca, batch)
            
            current_bs = batch["image"].shape[0]
            total_faces_processed += current_bs
            
            for j in range(current_bs):
                name = batch["image_name"][j]
                
                sample_output_folder = output_folder / name
                sample_output_folder.mkdir(parents=True, exist_ok=True)
                
                # Save mesh if requested
                if should_save_mesh:
                    mesh_path = sample_output_folder / "mesh_coarse.obj"
                    save_obj(emoca, str(mesh_path), vals, j)
                    output_files.append(str(mesh_path))
                
                # FIXED: Use renamed function to avoid conflict
                if should_save_images:
                    save_emoca_images(str(output_folder), name, visdict, with_detection=True, i=j)
                    # Add image output paths (EMOCA saves multiple image outputs)
                    for img_type in ['inputs', 'detail', 'coarse']:
                        potential_img = output_folder / f"{name}_{img_type}.jpg"
                        if potential_img.exists():
                            output_files.append(str(potential_img))
                
                # Save codes (FLAME parameters) if requested
                if should_save_codes:
                    save_codes(output_folder, name, vals, i=j)
                    
                    # Also save as more accessible JSON format
                    codes_json = {}
                    
                    # Extract FLAME parameters from vals
                    if 'expcode' in vals:
                        codes_json['expression'] = vals['expcode'][j].detach().cpu().numpy().tolist()
                    if 'shapecode' in vals:
                        codes_json['shape'] = vals['shapecode'][j].detach().cpu().numpy().tolist()
                    if 'posecode' in vals:
                        codes_json['pose'] = vals['posecode'][j].detach().cpu().numpy().tolist()
                    if 'cam' in vals:
                        codes_json['camera'] = vals['cam'][j].detach().cpu().numpy().tolist()
                    if 'lightcode' in vals:
                        codes_json['lighting'] = vals['lightcode'][j].detach().cpu().numpy().tolist()
                    if 'texcode' in vals:
                        codes_json['texture'] = vals['texcode'][j].detach().cpu().numpy().tolist()
                    if 'detailcode' in vals:
                        codes_json['detail'] = vals['detailcode'][j].detach().cpu().numpy().tolist()
                    
                    # Save comprehensive codes as JSON
                    codes_json_path = sample_output_folder / "codes.json"
                    with open(codes_json_path, 'w') as f:
                        json.dump(codes_json, f, indent=2)
                    output_files.append(str(codes_json_path))
                    
                    # Also save individual parameter files for easier access
                    for param_name, param_data in codes_json.items():
                        param_file = sample_output_folder / f"{param_name}.json"
                        with open(param_file, 'w') as f:
                            json.dump({param_name: param_data}, f, indent=2)
                        output_files.append(str(param_file))
        
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "message": f"Successfully processed {total_faces_processed} face(s)",
            "processing_time": processing_time,
            "num_faces": total_faces_processed,
            "output_files": output_files
        }
        
    finally:
        # Clean up temporary folder
        if temp_input_folder.exists():
            shutil.rmtree(temp_input_folder)

@app.on_event("startup")
async def startup_event():
    """Load models when service starts"""
    try:
        load_models()
        print("🎉 EMOCA Service ready!")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_loaded = emoca is not None
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
        "service": "EMOCA Persistence Service",
        "version": "1.0.0",
        "models_loaded": emoca is not None,
        "cuda_available": torch.cuda.is_available(),
        "model_details": {
            "emoca_loaded": emoca is not None,
            "config_loaded": conf is not None,
        }
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Main inference endpoint"""
    try:
        # Re-load model if different parameters requested
        current_model_name = getattr(emoca, 'model_name', 'unknown') if emoca else None
        if emoca is None or current_model_name != request.model_name:
            print(f"Loading different EMOCA model: {request.model_name}")
            load_models(request.model_name, request.mode)
        
        # FIXED: Pass renamed parameters to avoid conflict
        result = process_single_image(
            image_path=request.image_path,
            output_dir=request.output_dir,
            should_save_images=request.save_images,
            should_save_codes=request.save_codes,
            should_save_mesh=request.save_mesh
        )
        
        return PredictResponse(**result)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/reload")
async def reload_model(model_name: str = 'EMOCA_v2_lr_mse_20', mode: str = 'detail'):
    """Reload EMOCA model with different parameters"""
    try:
        load_models(model_name, mode)
        return {"success": True, "message": f"Reloaded model '{model_name}' in '{mode}' mode"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting EMOCA Persistence Service on port 8003...")
    uvicorn.run(app, host="0.0.0.0", port=8003)