#!/usr/bin/env python3
"""
Unified 3D Whole Body Pipeline
Orchestrates SMPLest-X, WiLoR, and EMOCA adapters concurrently
"""

import os
import sys
import asyncio
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import argparse
import shutil

class UnifiedPipeline:
    """Orchestrates multiple 3D human analysis adapters"""
    
    def __init__(self, input_path: str, output_dir: str = "unified_results", 
                 continue_on_error: bool = True, max_workers: int = 3):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.continue_on_error = continue_on_error
        self.max_workers = max_workers
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_output_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Define adapter configurations
        self.adapters = {
            'smplestx': {
                'script': 'adapters/smplestx_adapter.py',
                'output_subdir': 'smplestx_results',
                'args': self._get_smplestx_args
            },
            'wilor': {
                'script': 'adapters/wilor_adapter.py', 
                'output_subdir': 'wilor_results',
                'args': self._get_wilor_args
            },
            'emoca': {
                'script': 'adapters/emoca_adapter.py',
                'output_subdir': 'emoca_results', 
                'args': self._get_emoca_args
            }
        }
        
    def setup_logging(self):
        """Setup logging for the pipeline"""
        log_file = self.run_output_dir / "pipeline.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('UnifiedPipeline')
        
    def _get_smplestx_args(self, adapter_output_dir: Path) -> List[str]:
        """Generate arguments for SMPLest-X adapter"""
        config_path = os.path.relpath("pretrained_models/smplest_x/config_base.py", os.getcwd())
        input_path = os.path.relpath(self.input_path, os.getcwd())
        
        return [
            '--cfg_path', config_path,
            '--input_image', input_path,
            '--output_dir', str(adapter_output_dir)
            #'--multi_person'  # Enable multi-person detection
        ]
    
    def _get_wilor_args(self, adapter_output_dir: Path) -> List[str]:
        """Generate arguments for WiLoR adapter"""
        # WiLoR expects input folder, so create temp folder with single image
        temp_input_dir = adapter_output_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        
        # Copy input image to temp folder
        temp_image = temp_input_dir / self.input_path.name
        shutil.copy2(self.input_path, temp_image)
        
        # Convert paths to be relative from adapters directory
        temp_input_rel = os.path.relpath(temp_input_dir, Path(os.getcwd()) / 'adapters')
        output_rel = os.path.relpath(adapter_output_dir, Path(os.getcwd()) / 'adapters')
        
        return [
            '--img_folder', temp_input_rel,
            '--out_folder', output_rel,
            #'--save_mesh',
            '--rescale_factor', '2.0'
        ]
    
    def _get_emoca_args(self, adapter_output_dir: Path) -> List[str]:
        """Generate arguments for EMOCA adapter"""
        # EMOCA expects input folder, so create temp folder with single image  
        temp_input_dir = adapter_output_dir / "temp_input"
        temp_input_dir.mkdir(exist_ok=True)
        
        # Copy input image to temp folder
        temp_image = temp_input_dir / self.input_path.name
        shutil.copy2(self.input_path, temp_image)
        
        return [
            '--input_folder', str(temp_input_dir),
            '--output_folder', str(adapter_output_dir),
            '--save_images', 'True',
            '--save_codes', 'True', 
            #'--save_mesh', 'True',
            '--mode', 'detail'
        ]
    
    def run_adapter(self, adapter_name: str, adapter_config: Dict) -> Tuple[str, bool, str]:
        """Run a single adapter subprocess"""
        adapter_output_dir = self.run_output_dir / adapter_config['output_subdir']
        adapter_output_dir.mkdir(exist_ok=True)
        
        script_path = adapter_config['script']
        args = adapter_config['args'](adapter_output_dir)
        
        # Set working directory based on adapter requirements
        if adapter_name == 'wilor':
            # WiLoR needs to be run from adapters directory due to relative imports
            cwd = Path(os.getcwd()) / 'adapters'
            script_path = 'wilor_adapter.py'  # Use relative path from adapters dir
        else:
            cwd = os.getcwd()
        
        cmd = [sys.executable, script_path] + args
        
        self.logger.info(f"Starting {adapter_name} with command: {' '.join(cmd)}")
        self.logger.info(f"Working directory: {cwd}")
        
        try:
            # Run the adapter
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per adapter
                cwd=str(cwd)
            )
            
            # Log outputs
            log_file = adapter_output_dir / f"{adapter_name}.log"
            with open(log_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"STDOUT:\n{result.stdout}\n")
                f.write(f"STDERR:\n{result.stderr}\n")
            
            if result.returncode == 0:
                self.logger.info(f"✓ {adapter_name} completed successfully")
                return adapter_name, True, f"Success: outputs in {adapter_output_dir}"
            else:
                error_msg = f"Failed with return code {result.returncode}. Check {log_file}"
                self.logger.error(f"✗ {adapter_name} failed: {error_msg}")
                return adapter_name, False, error_msg
                
        except subprocess.TimeoutExpired:
            error_msg = "Timeout after 10 minutes"
            self.logger.error(f"✗ {adapter_name} timed out")
            return adapter_name, False, error_msg
            
        except Exception as e:
            error_msg = f"Exception: {str(e)}"
            self.logger.error(f"✗ {adapter_name} failed with exception: {error_msg}")
            return adapter_name, False, error_msg
    
    def run_pipeline(self) -> Dict[str, Tuple[bool, str]]:
        """Run all adapters concurrently"""
        self.logger.info(f"Starting unified pipeline for image: {self.input_path}")
        self.logger.info(f"Output directory: {self.run_output_dir}")
        
        results = {}
        
        # Run adapters concurrently using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all adapter tasks
            future_to_adapter = {
                executor.submit(self.run_adapter, name, config): name 
                for name, config in self.adapters.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_adapter):
                adapter_name = future_to_adapter[future]
                try:
                    name, success, message = future.result()
                    results[name] = (success, message)
                except Exception as e:
                    error_msg = f"Executor exception: {str(e)}"
                    results[adapter_name] = (False, error_msg)
                    self.logger.error(f"Adapter {adapter_name} failed in executor: {error_msg}")
        
        # Generate summary
        self.generate_summary(results)
        return results
    
    def generate_summary(self, results: Dict[str, Tuple[bool, str]]):
        """Generate a summary of the pipeline run"""
        summary = {
            'timestamp': self.timestamp,
            'input_image': str(self.input_path),
            'output_directory': str(self.run_output_dir),
            'results': {}
        }
        
        successful_adapters = []
        failed_adapters = []
        
        for adapter_name, (success, message) in results.items():
            summary['results'][adapter_name] = {
                'success': success,
                'message': message,
                'output_dir': str(self.run_output_dir / self.adapters[adapter_name]['output_subdir'])
            }
            
            if success:
                successful_adapters.append(adapter_name)
            else:
                failed_adapters.append(adapter_name)
        
        summary['summary'] = {
            'total_adapters': len(self.adapters),
            'successful': len(successful_adapters),
            'failed': len(failed_adapters),
            'success_rate': len(successful_adapters) / len(self.adapters) * 100
        }
        
        # Save summary
        summary_file = self.run_output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log summary
        self.logger.info("\n" + "="*60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Input: {self.input_path}")
        self.logger.info(f"Output: {self.run_output_dir}")
        self.logger.info(f"Success rate: {summary['summary']['success_rate']:.1f}%")
        
        if successful_adapters:
            self.logger.info(f"✓ Successful: {', '.join(successful_adapters)}")
        
        if failed_adapters:
            self.logger.info(f"✗ Failed: {', '.join(failed_adapters)}")
            
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='Unified 3D Whole Body Pipeline')
    parser.add_argument('--input_image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='unified_results',
                       help='Output directory for all results')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                       help='Continue pipeline even if some adapters fail')
    parser.add_argument('--max_workers', type=int, default=3,
                       help='Maximum number of concurrent adapters')
    parser.add_argument('--sequential', action='store_true', 
                       help='Run adapters sequentially instead of concurrently')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input
    if not Path(args.input_image).exists():
        print(f"Error: Input image not found: {args.input_image}")
        sys.exit(1)
    
    # Set max_workers to 1 for sequential execution
    max_workers = 1 if args.sequential else args.max_workers
    
    # Create and run pipeline
    pipeline = UnifiedPipeline(
        input_path=args.input_image,
        output_dir=args.output_dir,
        continue_on_error=args.continue_on_error,
        max_workers=max_workers
    )
    
    results = pipeline.run_pipeline()
    
    # Exit with appropriate code
    failed_count = sum(1 for success, _ in results.values() if not success)
    if failed_count > 0 and not args.continue_on_error:
        sys.exit(1)

if __name__ == '__main__':
    main()