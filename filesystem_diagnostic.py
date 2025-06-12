#!/usr/bin/env python3
"""
FILE SYSTEM DIAGNOSTIC TOOL
Comprehensive analysis of file system state for fusion debugging
"""

import os
import sys
import json
import stat
import pwd
import grp
import time
from pathlib import Path
import psutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FileSystemDiagnostic:
    """Comprehensive file system diagnostics for fusion debugging"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.diagnostic_results = {}
        
    def check_basic_info(self):
        """Check basic system and environment info"""
        logger.info("🔍 BASIC SYSTEM INFORMATION")
        logger.info("=" * 50)
        
        info = {
            'current_user': os.environ.get('USER', 'Unknown'),
            'current_uid': os.getuid(),
            'current_gid': os.getgid(),
            'working_directory': os.getcwd(),
            'python_executable': sys.executable,
            'python_version': sys.version,
            'platform': sys.platform,
        }
        
        for key, value in info.items():
            logger.info(f"   {key}: {value}")
        
        self.diagnostic_results['basic_info'] = info
        logger.info("=" * 50)
    
    def check_disk_space(self):
        """Check available disk space"""
        logger.info("💾 DISK SPACE ANALYSIS")
        logger.info("=" * 50)
        
        try:
            disk_usage = psutil.disk_usage('/')
            
            space_info = {
                'total_gb': disk_usage.total / (1024**3),
                'used_gb': disk_usage.used / (1024**3),
                'free_gb': disk_usage.free / (1024**3),
                'percent_used': (disk_usage.used / disk_usage.total) * 100
            }
            
            logger.info(f"   Total space: {space_info['total_gb']:.2f} GB")
            logger.info(f"   Used space: {space_info['used_gb']:.2f} GB")
            logger.info(f"   Free space: {space_info['free_gb']:.2f} GB")
            logger.info(f"   Percent used: {space_info['percent_used']:.1f}%")
            
            if space_info['free_gb'] < 1.0:
                logger.warning("⚠️  WARNING: Less than 1GB free space!")
            
            self.diagnostic_results['disk_space'] = space_info
            
        except Exception as e:
            logger.error(f"❌ Failed to check disk space: {e}")
            self.diagnostic_results['disk_space'] = {'error': str(e)}
        
        logger.info("=" * 50)
    
    def check_directory_permissions(self, path: Path):
        """Check detailed permissions for a directory"""
        try:
            stat_info = path.stat()
            
            # Get permission bits
            permissions = stat.filemode(stat_info.st_mode)
            
            # Get owner/group info
            try:
                owner_name = pwd.getpwuid(stat_info.st_uid).pw_name
            except KeyError:
                owner_name = f"UID:{stat_info.st_uid}"
            
            try:
                group_name = grp.getgrgid(stat_info.st_gid).gr_name
            except KeyError:
                group_name = f"GID:{stat_info.st_gid}"
            
            perm_info = {
                'permissions': permissions,
                'owner': owner_name,
                'group': group_name,
                'size_bytes': stat_info.st_size,
                'modified_time': time.ctime(stat_info.st_mtime),
                'can_read': os.access(path, os.R_OK),
                'can_write': os.access(path, os.W_OK),
                'can_execute': os.access(path, os.X_OK)
            }
            
            return perm_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_results_directory(self):
        """Comprehensive analysis of results directory structure"""
        logger.info("📁 RESULTS DIRECTORY ANALYSIS")
        logger.info("=" * 50)
        
        logger.info(f"📂 Target directory: {self.results_dir}")
        logger.info(f"   Absolute path: {self.results_dir.absolute()}")
        logger.info(f"   Exists: {self.results_dir.exists()}")
        
        if not self.results_dir.exists():
            logger.error("❌ CRITICAL: Results directory does not exist!")
            self.diagnostic_results['results_directory'] = {'exists': False}
            return
        
        # Check directory permissions
        dir_perms = self.check_directory_permissions(self.results_dir)
        logger.info(f"📋 Directory permissions:")
        for key, value in dir_perms.items():
            logger.info(f"   {key}: {value}")
        
        # Analyze subdirectories
        expected_subdirs = ['smplestx_results', 'wilor_results', 'emoca_results', 'fusion_results']
        subdir_analysis = {}
        
        for subdir_name in expected_subdirs:
            subdir_path = self.results_dir / subdir_name
            logger.info(f"📁 {subdir_name}:")
            
            if subdir_path.exists():
                perms = self.check_directory_permissions(subdir_path)
                file_count = len(list(subdir_path.rglob('*')))
                
                subdir_info = {
                    'exists': True,
                    'permissions': perms,
                    'file_count': file_count
                }
                
                logger.info(f"   ✅ EXISTS - {file_count} files")
                logger.info(f"   Permissions: {perms.get('permissions', 'unknown')}")
                logger.info(f"   Can write: {perms.get('can_write', False)}")
                
            else:
                subdir_info = {'exists': False}
                logger.info(f"   ❌ MISSING")
            
            subdir_analysis[subdir_name] = subdir_info
        
        self.diagnostic_results['results_directory'] = {
            'exists': True,
            'permissions': dir_perms,
            'subdirectories': subdir_analysis
        }
        
        logger.info("=" * 50)
    
    def check_required_files(self):
        """Check for required input files"""
        logger.info("📄 REQUIRED FILES CHECK")
        logger.info("=" * 50)
        
        # Define file patterns to search for
        file_patterns = {
            'smplestx_params': [
                'smplestx_results/inference_output_*/person_*/smplx_params_*.json',
                'smplestx_results/*/person_*/smplx_params_*.json'
            ],
            'smplestx_camera': [
                'smplestx_results/inference_output_*/person_*/camera_metadata.json',
                'smplestx_results/*/person_*/camera_metadata.json'
            ],
            'wilor_params': [
                'wilor_results/*_parameters.json',
                'wilor_results/parameters.json'
            ],
            'emoca_params': [
                'emoca_results/test*/codes.json',
                'emoca_results/*/codes.json',
                'emoca_results/EMOCA*/test*/codes.json'
            ],
            'coordinate_analysis': [
                'coordinate_analysis_summary.json'
            ]
        }
        
        files_analysis = {}
        
        for file_type, patterns in file_patterns.items():
            logger.info(f"🔍 Searching for {file_type}:")
            found_files = []
            
            for pattern in patterns:
                matches = list(self.results_dir.glob(pattern))
                found_files.extend(matches)
                logger.info(f"   Pattern '{pattern}': {len(matches)} matches")
                for match in matches:
                    logger.info(f"     📄 {match.relative_to(self.results_dir)}")
            
            if found_files:
                # Check first found file in detail
                first_file = found_files[0]
                file_perms = self.check_directory_permissions(first_file)
                
                files_analysis[file_type] = {
                    'found': True,
                    'count': len(found_files),
                    'first_file': str(first_file.relative_to(self.results_dir)),
                    'permissions': file_perms
                }
                logger.info(f"   ✅ FOUND {len(found_files)} files")
            else:
                files_analysis[file_type] = {'found': False}
                logger.info(f"   ❌ NOT FOUND")
        
        self.diagnostic_results['required_files'] = files_analysis
        logger.info("=" * 50)
    
    def test_file_operations(self):
        """Test actual file creation/deletion operations"""
        logger.info("🧪 FILE OPERATIONS TEST")
        logger.info("=" * 50)
        
        test_results = {}
        
        # Test 1: Create directory
        test_dir = self.results_dir / 'fusion_test'
        logger.info("📁 Testing directory creation...")
        try:
            test_dir.mkdir(exist_ok=True)
            logger.info("   ✅ Directory creation: SUCCESS")
            test_results['directory_creation'] = True
        except Exception as e:
            logger.error(f"   ❌ Directory creation: FAILED - {e}")
            test_results['directory_creation'] = False
        
        # Test 2: Create file
        test_file = test_dir / 'test_file.txt'
        logger.info("📄 Testing file creation...")
        try:
            with open(test_file, 'w') as f:
                f.write("test content")
            logger.info("   ✅ File creation: SUCCESS")
            test_results['file_creation'] = True
        except Exception as e:
            logger.error(f"   ❌ File creation: FAILED - {e}")
            test_results['file_creation'] = False
        
        # Test 3: Create numpy array (fusion-specific test)
        logger.info("🔢 Testing numpy array creation...")
        try:
            import numpy as np
            test_array = np.random.rand(10, 3)
            array_file = test_dir / 'test_array.npy'
            np.save(array_file, test_array)
            
            # Verify it can be loaded
            loaded_array = np.load(array_file)
            if np.array_equal(test_array, loaded_array):
                logger.info("   ✅ Numpy array save/load: SUCCESS")
                test_results['numpy_operations'] = True
            else:
                logger.error("   ❌ Numpy array save/load: DATA MISMATCH")
                test_results['numpy_operations'] = False
        except Exception as e:
            logger.error(f"   ❌ Numpy array operations: FAILED - {e}")
            test_results['numpy_operations'] = False
        
        # Cleanup
        logger.info("🧹 Cleaning up test files...")
        try:
            if test_dir.exists():
                import shutil
                shutil.rmtree(test_dir)
            logger.info("   ✅ Cleanup: SUCCESS")
            test_results['cleanup'] = True
        except Exception as e:
            logger.error(f"   ❌ Cleanup: FAILED - {e}")
            test_results['cleanup'] = False
        
        self.diagnostic_results['file_operations'] = test_results
        logger.info("=" * 50)
    
    def check_python_environment(self):
        """Check Python environment and imports"""
        logger.info("🐍 PYTHON ENVIRONMENT CHECK")
        logger.info("=" * 50)
        
        # Test critical imports
        imports_to_test = [
            'numpy', 'torch', 'json', 'pathlib', 'cv2', 
            'scipy', 'trimesh', 'pickle'
        ]
        
        import_results = {}
        
        for module_name in imports_to_test:
            logger.info(f"📦 Testing import: {module_name}")
            try:
                __import__(module_name)
                logger.info(f"   ✅ SUCCESS")
                import_results[module_name] = True
            except ImportError as e:
                logger.error(f"   ❌ FAILED: {e}")
                import_results[module_name] = False
        
        # Check PYTHONPATH
        pythonpath = os.environ.get('PYTHONPATH', '')
        logger.info(f"🛤️  PYTHONPATH: {pythonpath if pythonpath else 'Not set'}")
        
        # Check sys.path
        logger.info("🛤️  sys.path (first 5 entries):")
        for i, path in enumerate(sys.path[:5]):
            logger.info(f"   {i+1}. {path}")
        
        self.diagnostic_results['python_environment'] = {
            'imports': import_results,
            'pythonpath': pythonpath,
            'sys_path_length': len(sys.path)
        }
        
        logger.info("=" * 50)
    
    def save_diagnostic_report(self):
        """Save comprehensive diagnostic report"""
        report_file = self.results_dir / 'filesystem_diagnostic_report.json'
        
        try:
            with open(report_file, 'w') as f:
                json.dump(self.diagnostic_results, f, indent=2, default=str)
            logger.info(f"💾 Diagnostic report saved: {report_file}")
        except Exception as e:
            logger.error(f"❌ Failed to save diagnostic report: {e}")
    
    def run_complete_diagnostic(self):
        """Run all diagnostic checks"""
        logger.info("🧪 STARTING COMPLETE FILE SYSTEM DIAGNOSTIC")
        logger.info("=" * 70)
        
        self.check_basic_info()
        self.check_disk_space()
        self.analyze_results_directory()
        self.check_required_files()
        self.test_file_operations()
        self.check_python_environment()
        
        # Summary
        logger.info("📊 DIAGNOSTIC SUMMARY")
        logger.info("=" * 50)
        
        critical_checks = [
            ('Results directory exists', self.diagnostic_results.get('results_directory', {}).get('exists', False)),
            ('Can write to results dir', self.diagnostic_results.get('results_directory', {}).get('permissions', {}).get('can_write', False)),
            ('SMPLest-X files found', self.diagnostic_results.get('required_files', {}).get('smplestx_params', {}).get('found', False)),
            ('WiLoR files found', self.diagnostic_results.get('required_files', {}).get('wilor_params', {}).get('found', False)),
            ('Coordinate analysis found', self.diagnostic_results.get('required_files', {}).get('coordinate_analysis', {}).get('found', False)),
            ('File operations work', self.diagnostic_results.get('file_operations', {}).get('file_creation', False)),
            ('Numpy operations work', self.diagnostic_results.get('file_operations', {}).get('numpy_operations', False))
        ]
        
        all_critical_passed = True
        for check_name, result in critical_checks:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"   {check_name}: {status}")
            if not result:
                all_critical_passed = False
        
        logger.info("=" * 50)
        if all_critical_passed:
            logger.info("✅ ALL CRITICAL CHECKS PASSED - Fusion should work")
        else:
            logger.error("❌ CRITICAL CHECKS FAILED - Fusion likely to fail")
        
        self.save_diagnostic_report()
        logger.info("=" * 70)
        
        return all_critical_passed

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='File System Diagnostic Tool for Fusion Debugging')
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Directory containing pipeline results to diagnose')
    args = parser.parse_args()
    
    diagnostic = FileSystemDiagnostic(args.results_dir)
    success = diagnostic.run_complete_diagnostic()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()