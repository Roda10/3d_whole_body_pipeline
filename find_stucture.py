#!/usr/bin/env python3
"""
Quick script to find and display the structure of your pipeline results
"""

import os
from pathlib import Path
import json

def find_pipeline_structure(base_dir):
    """Find and display the structure of pipeline results"""
    base_path = Path(base_dir)
    
    print(f"\n📁 Analyzing directory: {base_path}")
    print("=" * 60)
    
    # Find all JSON files
    json_files = list(base_path.rglob("*.json"))
    print(f"\n📄 Found {len(json_files)} JSON files:")
    
    # Categorize files
    smplestx_files = []
    wilor_files = []
    emoca_files = []
    other_files = []
    
    for jf in json_files:
        rel_path = jf.relative_to(base_path)
        print(f"   - {rel_path}")
        
        # Try to categorize based on path or content
        if 'smplx' in str(jf).lower() or 'smplest' in str(jf).lower():
            smplestx_files.append(jf)
        elif 'wilor' in str(jf).lower() or 'hand' in str(jf).lower():
            wilor_files.append(jf)
        elif 'emoca' in str(jf).lower() or 'expression' in str(jf).lower():
            emoca_files.append(jf)
        else:
            # Try to peek inside to categorize
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        if 'joints_3d' in keys or 'body_pose' in keys:
                            smplestx_files.append(jf)
                        elif 'hands' in keys or 'vertices_3d' in keys:
                            wilor_files.append(jf)
                        elif 'expcode' in keys or 'shapecode' in keys:
                            emoca_files.append(jf)
                        else:
                            other_files.append(jf)
            except:
                other_files.append(jf)
    
    # Display categorization
    print("\n🎯 Categorized files:")
    print(f"\nSMPLest-X files ({len(smplestx_files)}):")
    for f in smplestx_files:
        print(f"   - {f.relative_to(base_path)}")
    
    print(f"\nWiLoR files ({len(wilor_files)}):")
    for f in wilor_files:
        print(f"   - {f.relative_to(base_path)}")
    
    print(f"\nEMOCA files ({len(emoca_files)}):")
    for f in emoca_files:
        print(f"   - {f.relative_to(base_path)}")
    
    print(f"\nOther files ({len(other_files)}):")
    for f in other_files:
        print(f"   - {f.relative_to(base_path)}")
    
    # Find subdirectories
    print("\n📂 Directory structure:")
    for root, dirs, files in os.walk(base_path):
        level = root.replace(str(base_path), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")
    
    # Try to peek at one file from each category
    print("\n🔍 Sample file contents:")
    
    if smplestx_files:
        print("\nSample SMPLest-X file:")
        try:
            with open(smplestx_files[0], 'r') as f:
                data = json.load(f)
                print(f"   Keys: {list(data.keys())[:10]}")
                if 'joints_3d' in data and hasattr(data['joints_3d'], 'shape'):
                    print(f"   joints_3d shape: {data['joints_3d'].shape}")
        except Exception as e:
            print(f"   Error reading: {e}")
    
    if wilor_files:
        print("\nSample WiLoR file:")
        try:
            with open(wilor_files[0], 'r') as f:
                data = json.load(f)
                print(f"   Keys: {list(data.keys())[:10]}")
        except Exception as e:
            print(f"   Error reading: {e}")
    
    if emoca_files:
        print("\nSample EMOCA file:")
        try:
            with open(emoca_files[0], 'r') as f:
                data = json.load(f)
                print(f"   Keys: {list(data.keys())[:10]}")
        except Exception as e:
            print(f"   Error reading: {e}")
    
    return {
        'smplestx_files': smplestx_files,
        'wilor_files': wilor_files,
        'emoca_files': emoca_files,
        'other_files': other_files
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python find_structure.py /path/to/pipeline_results")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    find_pipeline_structure(results_dir)