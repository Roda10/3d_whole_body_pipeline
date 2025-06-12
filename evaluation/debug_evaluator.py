#!/usr/bin/env python3
"""
Debug EHF Evaluator
Helps identify why main.py is failing
"""

import os
import sys
import json
import numpy as np
import subprocess
from pathlib import Path
import argparse
import datetime

class DebugEHFEvaluator:
    """Debug version to troubleshoot main.py issues"""
    
    def __init__(self, ehf_path: str = "data/EHF"):
        self.ehf_path = Path(ehf_path)
        
        # Setup output directory for debugging
        self.debug_dir = Path("debug_results") / f"debug_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Load frame list
        self.frames = self._load_frame_list()
        
        print(f"🔍 Debug EHF Evaluator initialized")
        print(f"   Dataset: {len(self.frames)} frames")
        print(f"   Debug output: {self.debug_dir}")

    def _load_frame_list(self) -> list:
        """Load list of EHF frame IDs"""
        frames = []
        for img_file in self.ehf_path.glob("*_img.jpg"):
            frame_id = img_file.stem.replace("_img", "")
            frames.append(frame_id)
        frames.sort()
        return frames

    def test_single_frame(self, frame_id: str):
        """Test main.py on a single frame with full debugging"""
        print(f"\n🔍 DEBUGGING FRAME: {frame_id}")
        
        # Check if input image exists
        img_path = self.ehf_path / f"{frame_id}_img.jpg"
        if not img_path.exists():
            print(f"❌ Input image not found: {img_path}")
            return
        
        print(f"✅ Input image found: {img_path}")
        print(f"   Size: {img_path.stat().st_size / 1024:.1f} KB")
        
        # Create debug temp directory
        temp_dir = self.debug_dir / f"temp_{frame_id}"
        temp_dir.mkdir(exist_ok=True)
        
        # Test main.py with full output capture
        print(f"\n📋 Running main.py...")
        main_cmd = [
            sys.executable, "main.py",
            "--input_image", str(img_path),
            "--output_dir", str(temp_dir)
        ]
        
        print(f"Command: {' '.join(main_cmd)}")
        print(f"Working directory: {os.getcwd()}")
        
        try:
            result = subprocess.run(
                main_cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=300
            )
            
            print(f"\n📊 SUBPROCESS RESULTS:")
            print(f"   Return code: {result.returncode}")
            print(f"   Runtime: Quick execution (likely failed)")
            
            # Save full output for analysis
            with open(self.debug_dir / f"main_stdout_{frame_id}.txt", 'w') as f:
                f.write(result.stdout)
            with open(self.debug_dir / f"main_stderr_{frame_id}.txt", 'w') as f:
                f.write(result.stderr)
            
            # Print stdout (truncated)
            if result.stdout:
                print(f"\n📤 STDOUT (first 500 chars):")
                print(result.stdout[:500])
                if len(result.stdout) > 500:
                    print("... (truncated, see main_stdout.txt)")
            else:
                print(f"\n📤 STDOUT: (empty)")
            
            # Print stderr (truncated)
            if result.stderr:
                print(f"\n📥 STDERR (first 500 chars):")
                print(result.stderr[:500])
                if len(result.stderr) > 500:
                    print("... (truncated, see main_stderr.txt)")
            else:
                print(f"\n📥 STDERR: (empty)")
            
            # Check what was actually created
            print(f"\n📁 OUTPUT DIRECTORY CONTENTS:")
            if temp_dir.exists():
                self._print_directory_tree(temp_dir, prefix="   ")
            else:
                print(f"   ❌ Output directory not created: {temp_dir}")
            
            # Look for run directories
            run_dirs = list(temp_dir.glob("run_*"))
            if run_dirs:
                print(f"\n📂 RUN DIRECTORIES FOUND: {len(run_dirs)}")
                for run_dir in run_dirs:
                    print(f"   📁 {run_dir.name}:")
                    self._print_directory_tree(run_dir, prefix="      ")
            else:
                print(f"\n📂 NO RUN DIRECTORIES FOUND")
            
            # Check for expected result files
            self._check_expected_outputs(temp_dir)
            
        except subprocess.TimeoutExpired:
            print(f"⏰ Process timed out after 5 minutes")
        except Exception as e:
            print(f"❌ Error running main.py: {e}")

    def _print_directory_tree(self, directory: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
        """Print directory tree structure"""
        if current_depth >= max_depth:
            return
        
        try:
            items = sorted(directory.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                
                if item.is_file():
                    size = item.stat().st_size
                    if size > 1024*1024:
                        size_str = f" ({size / (1024*1024):.1f} MB)"
                    elif size > 1024:
                        size_str = f" ({size / 1024:.1f} KB)"
                    else:
                        size_str = f" ({size} B)"
                    print(f"{prefix}{current_prefix}{item.name}{size_str}")
                else:
                    print(f"{prefix}{current_prefix}{item.name}/")
                    if current_depth < max_depth - 1:
                        next_prefix = prefix + ("    " if is_last else "│   ")
                        self._print_directory_tree(item, next_prefix, max_depth, current_depth + 1)
        except PermissionError:
            print(f"{prefix}(permission denied)")

    def _check_expected_outputs(self, temp_dir: Path):
        """Check for expected output files"""
        print(f"\n🔍 CHECKING FOR EXPECTED OUTPUTS:")
        
        expected_patterns = [
            "run_*/smplestx_results/*/person_*/smplx_params_*.json",
            "run_*/wilor_results/*_parameters.json",
            "run_*/emoca_results/*/codes.json"
        ]
        
        for pattern in expected_patterns:
            files = list(temp_dir.glob(pattern))
            if files:
                print(f"   ✅ Found {pattern}: {len(files)} files")
                for file in files[:3]:  # Show first 3
                    print(f"      📄 {file.relative_to(temp_dir)}")
                if len(files) > 3:
                    print(f"      ... and {len(files)-3} more")
            else:
                print(f"   ❌ Missing {pattern}")

    def test_main_py_directly(self):
        """Test if main.py can be imported and run"""
        print(f"\n🧪 TESTING MAIN.PY IMPORT:")
        
        try:
            # Test if we can import main.py
            sys.path.insert(0, os.getcwd())
            
            # Try to import
            print("   Attempting to import main.py...")
            import main
            print("   ✅ main.py imported successfully")
            
            # Check if UnifiedPipeline class exists
            if hasattr(main, 'UnifiedPipeline'):
                print("   ✅ UnifiedPipeline class found")
            else:
                print("   ❌ UnifiedPipeline class not found")
                print(f"   Available attributes: {[attr for attr in dir(main) if not attr.startswith('_')]}")
                
        except ImportError as e:
            print(f"   ❌ Import error: {e}")
        except Exception as e:
            print(f"   ❌ Other error: {e}")

    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print(f"\n🔧 CHECKING DEPENDENCIES:")
        
        # Check external directories
        external_dirs = ['external/SMPLest-X', 'external/WiLoR', 'external/EMOCA']
        for ext_dir in external_dirs:
            path = Path(ext_dir)
            if path.exists():
                print(f"   ✅ {ext_dir} exists")
            else:
                print(f"   ❌ {ext_dir} missing")
        
        # Check adapter files
        adapter_files = ['adapters/smplestx_adapter.py', 'adapters/wilor_adapter.py', 'adapters/emoca_adapter.py']
        for adapter in adapter_files:
            path = Path(adapter)
            if path.exists():
                print(f"   ✅ {adapter} exists")
            else:
                print(f"   ❌ {adapter} missing")
        
        # Check key imports
        key_imports = [
            ('torch', 'PyTorch'),
            ('cv2', 'OpenCV'),
            ('numpy', 'NumPy')
        ]
        
        for module, name in key_imports:
            try:
                __import__(module)
                print(f"   ✅ {name} available")
            except ImportError:
                print(f"   ❌ {name} missing")

    def run_debug_analysis(self, max_frames: int = 1):
        """Run complete debug analysis"""
        print("="*60)
        print("🔍 DEBUG ANALYSIS STARTING")
        print("="*60)
        
        # Check dependencies first
        self.check_dependencies()
        
        # Test main.py import
        self.test_main_py_directly()
        
        # Test on actual frames
        frames_to_test = self.frames[:max_frames]
        for frame_id in frames_to_test:
            self.test_single_frame(frame_id)
        
        print("\n" + "="*60)
        print("🔍 DEBUG ANALYSIS COMPLETE")
        print("="*60)
        print(f"Debug files saved to: {self.debug_dir}")

def main():
    parser = argparse.ArgumentParser(description='Debug EHF Evaluator')
    parser.add_argument('--ehf_path', type=str, default='data/EHF', help='EHF dataset path')
    parser.add_argument('--max_frames', type=int, default=1, help='Max frames to test')
    
    args = parser.parse_args()
    
    evaluator = DebugEHFEvaluator(args.ehf_path)
    evaluator.run_debug_analysis(args.max_frames)

if __name__ == '__main__':
    main()