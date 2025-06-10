#!/usr/bin/env python3
"""
Find and extract MANO mean poses from various sources
"""

import os
import numpy as np
import pickle
from pathlib import Path

def find_mano_mean_poses():
    """Search for MANO mean poses in common locations"""
    
    # Common paths to check
    search_paths = [
        '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/external/WiLoR/mano_data/mano_mean_params.npz',
        '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/external/WiLoR/mano_data/MANO_RIGHT.pkl',
        '/home/rodeo_aims_ac_za/3d_whole_body_pipeline/external/WiLoR/mano_data/LEFT.pkl',
    ]
    
    means_found = {
        'left': None,
        'right': None
    }
    
    print("🔍 Searching for MANO mean poses...\n")
    
    # Check NPZ files first
    for path in search_paths:
        if path.endswith('.npz') and os.path.exists(path):
            print(f"Found NPZ: {path}")
            data = np.load(path)
            print(f"  Keys: {list(data.keys())}")
            
            # Different files use different key names
            for key in ['hands_mean', 'pose_mean', 'mean_pose', 'pose']:
                if key in data:
                    mean_data = data[key]
                    print(f"  Found '{key}' with shape: {mean_data.shape}")
                    
                    if mean_data.size >= 45:
                        means_found['left'] = mean_data.flatten()[:45]
                        means_found['right'] = mean_data.flatten()[:45]
                        print("  ✅ Extracted hand means from NPZ")
                        break
    
    # Check PKL files for separate left/right
    for path in search_paths:
        if path.endswith('.pkl') and os.path.exists(path):
            print(f"\nFound PKL: {path}")
            try:
                with open(path, 'rb') as f:
                    mano_data = pickle.load(f, encoding='latin1')
                
                print(f"  Keys: {list(mano_data.keys())}")
                
                if 'hands_mean' in mano_data:
                    mean_pose = mano_data['hands_mean']
                    print(f"  Found 'hands_mean' with shape: {mean_pose.shape}")
                    
                    if 'RIGHT' in path.upper():
                        means_found['right'] = mean_pose.flatten()[:45]
                        print("  ✅ Extracted RIGHT hand mean")
                    elif 'LEFT' in path.upper():
                        means_found['left'] = mean_pose.flatten()[:45]
                        print("  ✅ Extracted LEFT hand mean")
                
            except Exception as e:
                print(f"  ❌ Error loading: {e}")
    
    # Save extracted means
    if means_found['left'] is not None and means_found['right'] is not None:
        output_path = 'pretrained_models/mano_mean_params.npz'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.savez(output_path,
                 hands_mean_left=means_found['left'],
                 hands_mean_right=means_found['right'])
        
        print(f"\n✅ Saved MANO mean poses to: {output_path}")
        print(f"   Left hand mean range: [{means_found['left'].min():.3f}, {means_found['left'].max():.3f}]")
        print(f"   Right hand mean range: [{means_found['right'].min():.3f}, {means_found['right'].max():.3f}]")
        
        return output_path
    else:
        print("\n❌ Could not find MANO mean poses!")
        print("\nTo fix this:")
        print("1. Download MANO models from https://mano.is.tue.mpg.de/")
        print("2. Place MANO_LEFT.pkl and MANO_RIGHT.pkl in pretrained_models/mano/")
        print("3. Run this script again")
        
        return None

if __name__ == '__main__':
    find_mano_mean_poses()