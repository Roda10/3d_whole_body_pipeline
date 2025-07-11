from pathlib import Path
import torch
import argparse
import os
import sys
import cv2
import numpy as np
import json
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.wilor_output_extractor import WiLoRParameterExtractor, save_wilor_parameters_json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'external', 'WiLoR'))

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

def main():

    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save rendered results')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = '../external/WiLoR/pretrained_models/wilor_final.ckpt' , cfg_path= '../external/WiLoR/pretrained_models/model_config.yaml')
    detector = YOLO('../external/WiLoR/pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    
    # Iterate over all images in folder
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf = 0.3, verbose=False)[0]
        bboxes    = []
        is_right  = []
        for det in detections: 
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            continue
            
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        
        # Store batch and output data for JSON export
        export_batches = []
        export_outputs = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Store for JSON export with correct MANO parameters
            export_batches.append({
                'img': batch['img'],
                'right': batch['right'],
                'box_center': batch['box_center'], 
                'box_size': batch['box_size'],
                'img_size': batch['img_size']
            })
            
            # Fix: Extract MANO parameters from the correct location
            export_outputs.append({
                'pred_vertices': out['pred_vertices'],
                'pred_keypoints_3d': out['pred_keypoints_3d'],
                'pred_cam': out['pred_cam'],
                'pred_cam_t_full': pred_cam_t_full,
                # Add MANO parameters - these are inside pred_mano_params
                'pred_mano_params': out.get('pred_mano_params', {})
            })
            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right    = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right-1)*verts[:,0]
                joints[:,0] = (2*is_right-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                # Save mesh if requested
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # Export comprehensive JSON parameters
        if len(all_verts) > 0:
            # Create parameter extractor
            extractor = WiLoRParameterExtractor(model_cfg)
            
            # Combine all batches for comprehensive export
            combined_batch = {
                'img': torch.cat([b['img'] for b in export_batches], dim=0),
                'right': torch.cat([b['right'] for b in export_batches], dim=0),
                'box_center': torch.cat([b['box_center'] for b in export_batches], dim=0),
                'box_size': torch.cat([b['box_size'] for b in export_batches], dim=0),
                'img_size': torch.cat([b['img_size'] for b in export_batches], dim=0)
            }
            
            # Combine outputs with MANO parameters
            combined_output = {
                'pred_vertices': torch.cat([o['pred_vertices'] for o in export_outputs], dim=0),
                'pred_keypoints_3d': torch.cat([o['pred_keypoints_3d'] for o in export_outputs], dim=0),
                'pred_cam': torch.cat([o['pred_cam'] for o in export_outputs], dim=0)
            }
            
            # Add MANO parameters if available
            if export_outputs and 'pred_mano_params' in export_outputs[0]:
                mano_params = {}
                for key in export_outputs[0]['pred_mano_params'].keys():
                    mano_params[key] = torch.cat([o['pred_mano_params'][key] for o in export_outputs], dim=0)
                combined_output['pred_mano_params'] = mano_params
            
            combined_cam_t = np.concatenate([o['pred_cam_t_full'] for o in export_outputs], axis=0)
            
            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            
            # Save comprehensive parameters JSON
            json_path = os.path.join(args.out_folder, f'{img_fn}_parameters.json')
            save_wilor_parameters_json(extractor, combined_batch, combined_output, 
                                     combined_cam_t, img_path, scaled_focal_length, json_path)
            
            print(f"Saved parameters to: {json_path}")
            
            # Render front view (your existing rendering code)
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()