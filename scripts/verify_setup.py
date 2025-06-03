#!/usr/bin/env python3
"""
Setup verification script for SMPLest-X adapter
Run this to verify your environment is correctly configured
"""

import sys
from pathlib import Path
import importlib.util

def check_file_exists(path, description):
    """Check if a file exists and print status"""
    if Path(path).exists():
        print(f"✓ {description}: {path}")
        return True
    else:
        print(f"✗ {description}: {path} (NOT FOUND)")
        return False

def check_python_module(module_path, module_name):
    """Check if a Python module can be imported from a specific path"""
    sys.path.insert(0, str(module_path))
    try:
        __import__(module_name)
        print(f"✓ Can import {module_name} from {module_path}")
        return True
    except ImportError as e:
        print(f"✗ Cannot import {module_name} from {module_path}: {e}")
        return False
    finally:
        if str(module_path) in sys.path:
            sys.path.remove(str(module_path))

def main():
    print("SMPLest-X Adapter Setup Verification")
    print("=" * 50)
    
    # Get project root (assuming this script is in project root)
    project_root = Path(__file__).parent
    print(f"Project root: {project_root.absolute()}")
    print()
    
    # Check project structure
    print("1. Checking project structure...")
    structure_ok = True
    
    required_dirs = [
        ("adapters", "adapters"),
        ("external", "external"),
        ("external/SMPLest-X", "external/SMPLest-X"),
        ("data", "data"),
        ("data/full_images", "data/full_images"),
        ("pretrained_models", "pretrained_models"),
        ("human_models", "human_models"),
        ("human_models/human_model_files", "human_models/human_model_files"),
        ("human_models/human_model_files/smplx", "human_models/human_model_files/smplx")
    ]
    
    for desc, path in required_dirs:
        if not check_file_exists(project_root / path, desc):
            structure_ok = False
    
    print()
    
    # Check required files
    print("2. Checking required files...")
    files_ok = True
    
    required_files = [
        ("SMPLest-X config", "external/SMPLest-X/configs/config_smplest_x_h.py"),
        ("SMPLest-X checkpoint", "pretrained_models/smplest_x/smplest_x_h.pth.tar"),
        ("SMPL-X model", "human_models/human_model_files/smplx/SMPLX_NEUTRAL.npz"),
        ("Test image", "data/full_images/test8.jpg")
    ]
    
    for desc, path in required_files:
        if not check_file_exists(project_root / path, desc):
            files_ok = False
    
    # Check optional YOLO model
    yolo_paths = [
        "external/SMPLest-X/pretrained_models/yolov8x.pt",
        "pretrained_models/yolov8x.pt"
    ]
    
    yolo_found = False
    for yolo_path in yolo_paths:
        if check_file_exists(project_root / yolo_path, "YOLO model (optional)"):
            yolo_found = True
            break
    
    if not yolo_found:
        print("  Note: YOLO model will be auto-downloaded on first use")
    
    print()
    
    # Check Python dependencies
    print("3. Checking Python dependencies...")
    deps_ok = True
    
    required_packages = [
        "torch",
        "torchvision", 
        "cv2",
        "numpy",
        "ultralytics"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (install with: pip install {package})")
            deps_ok = False
    
    print()
    
    # Check SMPLest-X imports
    print("4. Checking SMPLest-X module imports...")
    smplestx_path = project_root / "external" / "SMPLest-X"
    smplestx_ok = True
    
    if smplestx_path.exists():
        # Add SMPLest-X to path temporarily
        sys.path.insert(0, str(smplestx_path))
        
        smplestx_modules = [
            "human_models.human_models",
            "main.base",
            "main.config", 
            "utils.data_utils",
            "utils.visualization_utils"
        ]
        
        for module in smplestx_modules:
            try:
                __import__(module)
                print(f"✓ {module}")
            except ImportError as e:
                print(f"✗ {module}: {e}")
                smplestx_ok = False
        
        # Remove from path
        sys.path.remove(str(smplestx_path))
    else:
        print("✗ SMPLest-X directory not found")
        smplestx_ok = False
    
    print()
    
    # Check CUDA availability
    print("5. Checking CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("! CUDA not available - will use CPU (slower)")
    except:
        print("? Could not check CUDA status")
    
    print()
    
    # Summary
    print("Setup Verification Summary:")
    print("=" * 30)
    
    all_ok = structure_ok and files_ok and deps_ok and smplestx_ok
    
    if all_ok:
        print("🎉 All checks passed! Your environment is ready.")
        print()
        print("To run the SMPLest-X adapter:")
        print("  python adapters/smplestx_adapter.py")
    else:
        print("❌ Some issues found. Please resolve the above errors.")
        if not structure_ok:
            print("  - Fix project structure")
        if not files_ok:
            print("  - Ensure all required files are present")
        if not deps_ok:
            print("  - Install missing Python packages")
        if not smplestx_ok:
            print("  - Check SMPLest-X installation")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)