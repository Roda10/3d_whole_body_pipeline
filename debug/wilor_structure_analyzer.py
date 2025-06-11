#!/usr/bin/env python3
"""
Analyze the structure of WiLoR's 01_img_parameters.json file
"""

import json
from pathlib import Path
import numpy as np
import pprint

def analyze_wilor_structure(results_dir: str):
    """Analyze the exact structure of WiLoR output"""
    
    results_path = Path(results_dir)
    wilor_file = results_path / 'wilor_results' / '01_img_parameters.json'
    
    print(f"📊 ANALYZING WILOR DATA STRUCTURE")
    print(f"File: {wilor_file}")
    print("="*60)
    
    if not wilor_file.exists():
        print(f"❌ File not found: {wilor_file}")
        return
        
    with open(wilor_file, 'r') as f:
        data = json.load(f)
    
    # Top-level keys
    print("\n🔑 TOP-LEVEL KEYS:")
    for key in data.keys():
        value_type = type(data[key]).__name__
        if isinstance(data[key], list):
            print(f"  - {key}: {value_type} with {len(data[key])} items")
        elif isinstance(data[key], dict):
            print(f"  - {key}: {value_type} with {len(data[key])} keys")
        else:
            print(f"  - {key}: {value_type}")
    
    # Analyze 'hands' structure
    if 'hands' in data:
        print(f"\n🖐️ ANALYZING 'hands' STRUCTURE:")
        hands = data['hands']
        print(f"Number of hand entries: {len(hands)}")
        
        for idx, hand_data in enumerate(hands):
            print(f"\n📍 Hand entry {idx}:")
            
            if isinstance(hand_data, dict):
                # Show all keys
                print(f"   Keys: {list(hand_data.keys())}")
                
                # Check for hand type identifier
                if 'hand_type' in hand_data:
                    print(f"   Hand type: {hand_data['hand_type']}")
                elif 'type' in hand_data:
                    print(f"   Type: {hand_data['type']}")
                    
                # Check for MANO parameters
                if 'mano_parameters' in hand_data:
                    mano = hand_data['mano_parameters']
                    print(f"   MANO parameters found:")
                    for key, value in mano.items():
                        if isinstance(value, list):
                            arr = np.array(value)
                            print(f"     - {key}: shape {arr.shape}, range [{arr.min():.3f}, {arr.max():.3f}]")
                        else:
                            print(f"     - {key}: {type(value).__name__}")
                            
                # Check for 3D joints
                if 'cam_3d_joints' in hand_data:
                    joints = np.array(hand_data['cam_3d_joints'])
                    print(f"   3D joints: shape {joints.shape}")
                    
                # Check for mesh vertices
                if 'mesh_vertices' in hand_data:
                    verts = np.array(hand_data['mesh_vertices'])
                    print(f"   Mesh vertices: shape {verts.shape}")
                    
                # Show first few keys if many
                if len(hand_data) > 10:
                    print(f"   ... and {len(hand_data) - 10} more keys")
                else:
                    # Show all other keys
                    other_keys = [k for k in hand_data.keys() 
                                  if k not in ['mano_parameters', 'cam_3d_joints', 
                                             'mesh_vertices', 'hand_type', 'type']]
                    if other_keys:
                        print(f"   Other keys: {other_keys}")
                        
    # Check for any other relevant structures
    other_keys = [k for k in data.keys() if k != 'hands']
    if other_keys:
        print(f"\n📋 OTHER DATA:")
        for key in other_keys:
            if isinstance(data[key], (list, np.ndarray)):
                print(f"  - {key}: array/list with shape {np.array(data[key]).shape}")
            elif isinstance(data[key], dict):
                print(f"  - {key}: dict with keys {list(data[key].keys())[:5]}...")
            else:
                print(f"  - {key}: {data[key]}")
                
    # Save detailed structure to file
    output_file = results_path / 'wilor_results' / 'wilor_structure_analysis.txt'
    with open(output_file, 'w') as f:
        f.write("WILOR DATA STRUCTURE ANALYSIS\n")
        f.write("="*60 + "\n\n")
        f.write("Full data structure:\n")
        f.write(pprint.pformat(data, indent=2, width=100, depth=4))
        
    print(f"\n💾 Detailed structure saved to: {output_file}")
    
    # Determine how to access left and right hands
    print(f"\n🎯 HOW TO ACCESS HANDS:")
    if len(hands) == 2:
        print("  Likely: hands[0] = left, hands[1] = right")
    elif len(hands) == 3:
        print("  Likely: hands[0] = left, hands[1] = right, hands[2] = both/metadata")
        print("  Check 'hand_type' or 'type' field to confirm")
        
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    analyze_wilor_structure(args.results_dir)