#!/usr/bin/env python3
"""
Find where WiLoR data is actually saved in the results directory
"""

import json
from pathlib import Path
import pprint

def find_wilor_files(results_dir: str):
    """Search for all WiLoR-related files in the results directory"""
    
    results_path = Path(results_dir)
    print(f"🔍 Searching for WiLoR files in: {results_path}")
    print("="*60)
    
    # Pattern searches
    patterns = [
        '**/wilor*',
        '**/*hand*',
        '**/*mano*',
        '**/*parameters.json',
        '**/*3d_joints*',
        '**/*mesh*'
    ]
    
    found_files = {}
    
    for pattern in patterns:
        files = list(results_path.glob(pattern))
        if files:
            found_files[pattern] = files
    
    # Display findings
    print("\n📁 FOUND FILES:")
    print("-"*60)
    
    all_files = []
    for pattern, files in found_files.items():
        print(f"\nPattern '{pattern}':")
        for f in files:
            relative_path = f.relative_to(results_path)
            print(f"  📄 {relative_path}")
            all_files.append(f)
    
    # Check JSON files for hand-related data
    print("\n📊 ANALYZING JSON FILES:")
    print("-"*60)
    
    json_files = [f for f in all_files if f.suffix == '.json']
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            relative_path = json_file.relative_to(results_path)
            print(f"\n📄 {relative_path}:")
            
            # Check for hand-related keys
            hand_keys = [k for k in data.keys() if any(term in k.lower() for term in ['hand', 'mano', 'wilor'])]
            
            if hand_keys:
                print(f"  Found hand-related keys: {hand_keys}")
                
                # Show structure of hand data
                for key in hand_keys:
                    if isinstance(data[key], dict):
                        print(f"  '{key}' structure:")
                        for k, v in data[key].items():
                            if isinstance(v, (list, dict)):
                                print(f"    - {k}: {type(v).__name__} with {len(v)} items")
                            else:
                                print(f"    - {k}: {v}")
                    elif isinstance(data[key], list):
                        print(f"  '{key}': list with {len(data[key])} items")
            
            # Also check for specific expected structures
            if 'mano_parameters' in data:
                print("  ✅ Contains 'mano_parameters'")
                mano_keys = data['mano_parameters'].keys()
                print(f"     Keys: {list(mano_keys)}")
                
            if 'cam_3d_joints' in data:
                print("  ✅ Contains 'cam_3d_joints'")
                if isinstance(data['cam_3d_joints'], list):
                    print(f"     Shape: {len(data['cam_3d_joints'])} joints")
                    
        except Exception as e:
            print(f"  ⚠️  Error reading file: {e}")
    
    # Look specifically in expected locations
    print("\n🎯 CHECKING EXPECTED LOCATIONS:")
    print("-"*60)
    
    expected_paths = [
        'wilor_results/left_hand_parameters.json',
        'wilor_results/right_hand_parameters.json',
        'wilor_results/left_mano_params.json',
        'wilor_results/right_mano_params.json',
        'wilor_results/results.json',
        'wilor_output/left_hand_parameters.json',
        'wilor_output/right_hand_parameters.json',
    ]
    
    for exp_path in expected_paths:
        full_path = results_path / exp_path
        if full_path.exists():
            print(f"  ✅ Found: {exp_path}")
        else:
            print(f"  ❌ Missing: {exp_path}")
    
    # Check directory structure
    print("\n📂 DIRECTORY STRUCTURE:")
    print("-"*60)
    
    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        """Print directory tree"""
        if current_depth >= max_depth:
            return
            
        contents = sorted(directory.iterdir())
        dirs = [d for d in contents if d.is_dir()]
        files = [f for f in contents if f.is_file()]
        
        # Print directories first
        for i, path in enumerate(dirs):
            is_last_dir = (i == len(dirs) - 1) and len(files) == 0
            print(f"{prefix}{'└── ' if is_last_dir else '├── '}{path.name}/")
            
            extension = "    " if is_last_dir else "│   "
            print_tree(path, prefix + extension, max_depth, current_depth + 1)
        
        # Then files
        for i, path in enumerate(files):
            is_last = i == len(files) - 1
            print(f"{prefix}{'└── ' if is_last else '├── '}{path.name}")
    
    print_tree(results_path, max_depth=3)
    
    # Summary
    print("\n📊 SUMMARY:")
    print("-"*60)
    print(f"Total files found: {len(all_files)}")
    print(f"JSON files found: {len(json_files)}")
    
    # Suggest where WiLoR data might be
    print("\n💡 SUGGESTIONS:")
    if not any('wilor' in str(f).lower() for f in all_files):
        print("  ⚠️  No 'wilor' named files found - check if WiLoR adapter ran successfully")
    else:
        print("  ✅ Found WiLoR-related files")
        
    return all_files


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    find_wilor_files(args.results_dir)