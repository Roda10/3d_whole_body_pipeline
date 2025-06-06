import os
import sys
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import json
import cv2
import datetime
from pathlib import Path

# Add SMPLest-X to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'SMPLest-X'))

from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--ckpt_name', type=str, default='model_dump', help='Name of the checkpoint folder under pretrained_models')
    parser.add_argument('--multi_person', action='store_true', help='Enable multi-person inference')

    # --- NEW ARGUMENTS FOR SINGLE IMAGE INFERENCE ---
    parser.add_argument('--cfg_path', type=str, required=False, help='Path to the configuration file')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the single input image file')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save output images')
    # -----------------------------------------------

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config_path = args.cfg_path
    cfg = Config.load_config(config_path)
    
    checkpoint_path = '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/pretrained_models/smplest_x/smplest_x_h.pth.tar'

    # --- Output folder logic ---
    output_folder = Path(args.output_dir) / f'inference_output_{time_str}' # Create a unique subfolder for each run
    os.makedirs(output_folder, exist_ok=True)
    # ---------------------------
    
    # exp_name is now based on timestamp and checkpoint name
    exp_name = f'inference_single_image_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(output_folder, 'log'), # Log within the new output folder
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path) 

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using {args.num_gpus} GPU(s).")
    demoer.logger.info(f'Inference on single image [{args.input_image}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    # --- Single image processing ---
    img_path = args.input_image

    if not os.path.exists(img_path):
        demoer.logger.error(f"Error: Input image not found at {img_path}. Exiting.")
        return # Exit if image not found

    transform = transforms.ToTensor()
    original_img = load_img(img_path)
    vis_img = original_img.copy()
    original_img_height, original_img_width = original_img.shape[:2]
    
    # detection, xyxy
    yolo_results = detector.predict(original_img, 
                                device='cuda', 
                                classes=0, # 'person' class in COCO
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0] # Get the first (and only) result object

    # Extract bounding boxes and confidence scores
    yolo_bbox_xyxy = yolo_results.boxes.xyxy.detach().cpu().numpy()
    yolo_conf = yolo_results.boxes.conf.detach().cpu().numpy()

    # Combine xyxy and confidence scores for NMS
    # Each bbox will now be [x1, y1, x2, y2, confidence]
    yolo_bbox = np.concatenate((yolo_bbox_xyxy, yolo_conf[:, None]), axis=1)

    num_bbox = len(yolo_bbox)

    if num_bbox < 1:
        demoer.logger.info(f"No bounding boxes detected for {os.path.basename(img_path)}. Saving original image.")
        cv2.imwrite(os.path.join(output_folder, f"no_bbox_{os.path.basename(img_path)}"), vis_img[:, :, ::-1])
        return # Exit as no people detected

    if not args.multi_person:
        # If not multi-person, still take the top one (by confidence, which is now included)
        yolo_bbox = yolo_bbox[:1] 
        num_bbox = 1
    else:
        # Now yolo_bbox has 5 elements, so NMS will work
        yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
        num_bbox = len(yolo_bbox)


    if num_bbox < 1:
        demoer.logger.info(f"No bounding boxes detected for {os.path.basename(img_path)}. Saving original image.")
        cv2.imwrite(os.path.join(output_folder, f"no_bbox_{os.path.basename(img_path)}"), vis_img[:, :, ::-1])
        return # Exit as no people detected

    if not args.multi_person:
        yolo_bbox = yolo_bbox[:1] # Take only the first (largest/highest confidence) bbox
        num_bbox = 1
    else:
        yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
        num_bbox = len(yolo_bbox)

    # loop all detected bboxes
    for bbox_id in range(num_bbox):
        if bbox_id >= len(yolo_bbox):
            continue
            
        # Get the bbox in [x1, y1, x2, y2, confidence] format
        current_yolo_bbox_with_conf = yolo_bbox[bbox_id]
        
        # Extract just the [x1, y1, x2, y2] part
        x1, y1, x2, y2 = current_yolo_bbox_with_conf[:4]

        yolo_bbox_xywh = np.array([x1, y1, x2 - x1, y2 - y1]) # Convert to xywh for process_bbox

        # xywh
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

        # mesh recovery
        with torch.no_grad():
            out = demoer.model(inputs, targets, meta_info, 'test')

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
            
            # ADD THIS LINE - The actual mesh vertices!
            'mesh': out['smplx_mesh_cam'].detach().cpu().numpy()[0],           # Full body mesh vertices (10475 vertices)
        }
        # Create unique output subfolder for this inference run
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_output_folder = Path(args.output_dir) / f'inference_output_{time_str}_{Path(args.input_image).stem}'
        os.makedirs(run_output_folder, exist_ok=True)
        demoer.logger.info(f"Saving outputs to: {run_output_folder}")

        # # --- Save Parameters and Mesh ---
        person_output_folder = run_output_folder / f'person_{bbox_id}'
        os.makedirs(person_output_folder, exist_ok=True)

        # Create parameter shape summary
        param_shapes = {}
        for param_name, param_value in smplx_params.items():
            if isinstance(param_value, np.ndarray):
                param_shapes[param_name] = {
                    'shape': list(param_value.shape),
                    'size': int(param_value.size)
                }
        
        # Print parameter shapes
        demoer.logger.info(f"\n=== SMPL-X Parameter Shapes for Person {bbox_id} ===")
        for param_name, shape_info in param_shapes.items():
            demoer.logger.info(f"{param_name}: shape={shape_info['shape']}, "
                             f"size={shape_info['size']}")
        
        # Convert numpy arrays to serializable format for JSON
        serializable_params = numpy_to_serializable(smplx_params)
        
        # Save full parameters as .json
        params_filename = person_output_folder / f'smplx_params_person_{bbox_id}.json'
        with open(params_filename, 'w') as f:
            json.dump(serializable_params, f, indent=2)
        demoer.logger.info(f"Saved SMPL-X parameters for person {bbox_id} to: {params_filename}")

        # Save parameter shapes summary
        summary_filename = person_output_folder / f'smplx_shapes_person_{bbox_id}.json'
        summary = {
            'bbox_id': bbox_id,
            'bbox_xyxy': [float(x1), float(y1), float(x2), float(y2)],
            'parameter_shapes': param_shapes,
            'total_parameters': sum(shape_info['size'] for shape_info in param_shapes.values()),
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        demoer.logger.info(f"Saved parameter shapes summary to: {summary_filename}")

        mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

        # render mesh
        focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
        princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
        
        # --- SAVE CAMERA AND DETECTION METADATA ---
        camera_meta = {
            'focal_length': focal,
            'principal_point': princpt,
            'camera_translation': out['cam_trans'].detach().cpu().numpy()[0].tolist(),
            'detection_bbox': [float(x1), float(y1), float(x2), float(y2)]
        }

        # Save to a file inside the specific person's output folder
        meta_path = person_output_folder / "camera_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(camera_meta, f, indent=2)
        demoer.logger.info(f"✓ Camera metadata saved to {meta_path}")
        # --- END OF NEW CODE BLOCK ---

        # draw the bbox on img
        # Use the original xyxy for drawing the rectangle
        vis_img = cv2.rectangle(vis_img, (int(x1), int(y1)), 
                                (int(x2), int(y2)), (0, 255, 0), 1)
        # draw mesh
        vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=False)

    # save rendered image - Single image output
    output_filename = f"rendered_{os.path.basename(img_path)}"
    cv2.imwrite(os.path.join(output_folder, output_filename), vis_img[:, :, ::-1])
    demoer.logger.info(f"Processed image saved to: {os.path.join(output_folder, output_filename)}")

if __name__ == "__main__":
    main()