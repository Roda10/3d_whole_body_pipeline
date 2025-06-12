#!/usr/bin/env python3
"""
STANDALONE FUSION TEST SCRIPT
Mimics exactly how the evaluator calls fusion to isolate issues
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FusionTester:
    """Test fusion exactly as the evaluator would call it"""
    
    def __init__(self, test_results_dir: str):
        self.test_results_dir = Path(test_results_dir)
        self.fusion_script_path = Path(__file__).parent / "debug_fusion_enhanced.py"
        
    def run_filesystem_diagnostics(self):
        """Check file system state before fusion"""
        logger.info("🔍 RUNNING FILESYSTEM DIAGNOSTICS")
        logger.info("=" * 50)
        
        # Check if results directory exists
        logger.info(f"📁 Test results directory: {self.test_results_dir}")
        logger.info(f"   Exists: {self.test_results_dir.exists()}")
        logger.info(f"   Absolute path: {self.test_results_dir.absolute()}")
        
        if self.test_results_dir.exists():
            logger.info(f"📂 Contents of {self.test_results_dir}:")
            for item in self.test_results_dir.iterdir():
                logger.info(f"   {'📁' if item.is_dir() else '📄'} {item.name}")
                
            # Check for expected subdirectories
            expected_dirs = ['smplestx_results', 'wilor_results', 'emoca_results']
            for dir_name in expected_dirs:
                dir_path = self.test_results_dir / dir_name
                logger.info(f"📁 {dir_name}: {'✅ EXISTS' if dir_path.exists() else '❌ MISSING'}")
                if dir_path.exists():
                    files = list(dir_path.rglob('*'))
                    logger.info(f"   Files found: {len(files)}")
                    for file in files[:5]:  # Show first 5 files
                        logger.info(f"     📄 {file.relative_to(dir_path)}")
                    if len(files) > 5:
                        logger.info(f"     ... and {len(files) - 5} more files")
        
        # Check coordinate analysis
        coord_file = self.test_results_dir / 'coordinate_analysis_summary.json'
        logger.info(f"📄 Coordinate analysis: {'✅ EXISTS' if coord_file.exists() else '❌ MISSING'}")
        
        # Check permissions
        logger.info(f"✅ Directory permissions:")
        logger.info(f"   Readable: {os.access(self.test_results_dir, os.R_OK)}")
        logger.info(f"   Writable: {os.access(self.test_results_dir, os.W_OK)}")
        logger.info(f"   Executable: {os.access(self.test_results_dir, os.X_OK)}")
        
        logger.info("=" * 50)
    
    def check_fusion_script(self):
        """Verify fusion script exists and is executable"""
        logger.info("🔍 CHECKING FUSION SCRIPT")
        logger.info("=" * 50)
        
        logger.info(f"📄 Fusion script: {self.fusion_script_path}")
        logger.info(f"   Exists: {self.fusion_script_path.exists()}")
        logger.info(f"   Readable: {os.access(self.fusion_script_path, os.R_OK)}")
        logger.info(f"   Executable: {os.access(self.fusion_script_path, os.X_OK)}")
        
        if not self.fusion_script_path.exists():
            logger.error("❌ FUSION SCRIPT NOT FOUND!")
            return False
            
        logger.info("=" * 50)
        return True
    
    def run_fusion_test(self):
        """Run fusion exactly as evaluator would"""
        logger.info("🚀 RUNNING FUSION TEST")
        logger.info("=" * 50)
        
        # Command that mimics evaluator call
        cmd = [
            sys.executable,
            str(self.fusion_script_path),
            '--results_dir', str(self.test_results_dir)
        ]
        
        logger.info(f"🔧 Command: {' '.join(cmd)}")
        logger.info(f"📁 Working directory: {os.getcwd()}")
        logger.info(f"🐍 Python executable: {sys.executable}")
        
        # Check environment
        logger.info("🌍 Environment check:")
        logger.info(f"   PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
        logger.info(f"   Current user: {os.environ.get('USER', 'Unknown')}")
        
        start_time = time.time()
        
        try:
            logger.info("▶️  Starting fusion process...")
            
            # Run with full output capture
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=os.getcwd()
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            logger.info(f"⏱️  Execution time: {execution_time:.2f} seconds")
            logger.info(f"📊 Return code: {result.returncode}")
            
            # Log stdout
            if result.stdout:
                logger.info("📤 STDOUT:")
                logger.info("-" * 30)
                for line in result.stdout.split('\n'):
                    if line.strip():
                        logger.info(f"   {line}")
                logger.info("-" * 30)
            
            # Log stderr
            if result.stderr:
                logger.error("📥 STDERR:")
                logger.error("-" * 30)
                for line in result.stderr.split('\n'):
                    if line.strip():
                        logger.error(f"   {line}")
                logger.error("-" * 30)
            
            # Check if fusion succeeded
            if result.returncode == 0:
                logger.info("✅ FUSION PROCESS COMPLETED SUCCESSFULLY")
            else:
                logger.error(f"❌ FUSION PROCESS FAILED (exit code: {result.returncode})")
            
            return result
            
        except subprocess.TimeoutExpired:
            logger.error("❌ FUSION PROCESS TIMED OUT (> 5 minutes)")
            return None
        except Exception as e:
            logger.error(f"❌ FAILED TO RUN FUSION PROCESS: {e}")
            return None
    
    def check_fusion_outputs(self):
        """Check if fusion created expected output files"""
        logger.info("🔍 CHECKING FUSION OUTPUTS")
        logger.info("=" * 50)
        
        fusion_dir = self.test_results_dir / 'fusion_results'
        logger.info(f"📁 Fusion directory: {fusion_dir}")
        logger.info(f"   Exists: {fusion_dir.exists()}")
        
        if not fusion_dir.exists():
            logger.error("❌ Fusion directory does not exist!")
            return False
        
        # Check expected files
        expected_files = [
            'enhanced_mesh.npy',        # CRITICAL FILE
            'fused_parameters.json',
            'original_final.obj',
            'fused_final.obj'
        ]
        
        all_files_exist = True
        for filename in expected_files:
            file_path = fusion_dir / filename
            exists = file_path.exists()
            logger.info(f"📄 {filename}: {'✅ EXISTS' if exists else '❌ MISSING'}")
            
            if exists and filename == 'enhanced_mesh.npy':
                # Extra validation for critical file
                try:
                    import numpy as np
                    mesh_data = np.load(file_path)
                    logger.info(f"   ✅ enhanced_mesh.npy shape: {mesh_data.shape}")
                    logger.info(f"   ✅ enhanced_mesh.npy size: {file_path.stat().st_size} bytes")
                except Exception as e:
                    logger.error(f"   ❌ enhanced_mesh.npy corrupt: {e}")
                    all_files_exist = False
            
            if not exists:
                all_files_exist = False
        
        # List all files in fusion directory
        logger.info(f"📂 All files in fusion directory:")
        for file in fusion_dir.rglob('*'):
            if file.is_file():
                logger.info(f"   📄 {file.relative_to(fusion_dir)} ({file.stat().st_size} bytes)")
        
        logger.info("=" * 50)
        return all_files_exist
    
    def run_complete_test(self):
        """Run complete fusion test sequence"""
        logger.info("🧪 STARTING COMPLETE FUSION TEST")
        logger.info("=" * 60)
        
        # Step 1: Filesystem diagnostics
        self.run_filesystem_diagnostics()
        
        # Step 2: Check fusion script
        if not self.check_fusion_script():
            logger.error("❌ ABORTING: Fusion script not found")
            return False
        
        # Step 3: Run fusion
        result = self.run_fusion_test()
        if result is None:
            logger.error("❌ ABORTING: Fusion process failed to run")
            return False
        
        # Step 4: Check outputs
        outputs_ok = self.check_fusion_outputs()
        
        # Final summary
        logger.info("=" * 60)
        logger.info("📊 FUSION TEST SUMMARY")
        logger.info("=" * 60)
        
        success = result.returncode == 0 and outputs_ok
        
        logger.info(f"🔧 Process execution: {'✅ SUCCESS' if result.returncode == 0 else '❌ FAILED'}")
        logger.info(f"📄 Output files: {'✅ SUCCESS' if outputs_ok else '❌ FAILED'}")
        logger.info(f"🎯 Overall result: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if not success:
            logger.error("❌ FUSION TEST FAILED - Check logs above for details")
        else:
            logger.info("✅ FUSION TEST PASSED - All components working correctly")
        
        logger.info("=" * 60)
        return success

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone Fusion Test Script')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Directory containing pipeline results to test fusion on')
    args = parser.parse_args()
    
    tester = FusionTester(args.results_dir)
    success = tester.run_complete_test()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()