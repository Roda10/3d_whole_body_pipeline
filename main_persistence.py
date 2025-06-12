#!/usr/bin/env python3
"""
Unified 3D Whole Body Pipeline - Persistence Services Version
Uses HTTP services instead of subprocess calls for 50-70% performance improvement
"""

import os
import sys
import logging
import requests
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import argparse
import shutil

class UnifiedPipelinePersistence:
    """Orchestrates multiple 3D human analysis adapters using persistence services"""
    
    def __init__(self, input_path: str, output_dir: str = "unified_results", 
                 continue_on_error: bool = True, max_workers: int = 3,
                 auto_start_services: bool = True):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.continue_on_error = continue_on_error
        self.max_workers = max_workers
        self.auto_start_services = auto_start_services
        
        # Create timestamped output directory
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_output_dir = self.output_dir / f"run_{self.timestamp}"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # Service configurations
        self.services = {
            'smplestx': {
                'url': 'http://localhost:8001',
                'output_subdir': 'smplestx_results',
                'start_script': 'services/smplestx_service.py'
            },
            'wilor': {
                'url': 'http://localhost:8002',
                'output_subdir': 'wilor_results',
                'start_script': 'services/wilor_service.py'
            },
            'emoca': {
                'url': 'http://localhost:8003',
                'output_subdir': 'emoca_results',
                'start_script': 'services/emoca_service.py'
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
        self.logger = logging.getLogger('UnifiedPipelinePersistence')
        
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is running and healthy"""
        try:
            url = self.services[service_name]['url']
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get('status') == 'healthy'
            return False
        except:
            return False
    
    def start_service(self, service_name: str) -> bool:
        """Start a service if not already running"""
        if self.check_service_health(service_name):
            self.logger.info(f"✓ {service_name} service already running")
            return True
            
        if not self.auto_start_services:
            self.logger.error(f"✗ {service_name} service not running and auto_start disabled")
            return False
            
        self.logger.info(f"🚀 Starting {service_name} service...")
        try:
            script_path = self.services[service_name]['start_script']
            import subprocess
            # Start service in background
            subprocess.Popen([sys.executable, script_path], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for service to be ready (up to 60 seconds)
            for i in range(60):
                time.sleep(1)
                if self.check_service_health(service_name):
                    self.logger.info(f"✓ {service_name} service started successfully")
                    return True
                    
            self.logger.error(f"✗ {service_name} service failed to start within 60 seconds")
            return False
            
        except Exception as e:
            self.logger.error(f"✗ Failed to start {service_name} service: {e}")
            return False
    
    def ensure_services_running(self) -> Dict[str, bool]:
        """Ensure all services are running"""
        service_status = {}
        
        for service_name in self.services.keys():
            service_status[service_name] = self.start_service(service_name)
            
        return service_status
    
    def call_smplestx_service(self, adapter_output_dir: Path) -> Tuple[str, bool, str]:
        """Call SMPLest-X service via HTTP"""
        try:
            url = self.services['smplestx']['url']
            
            # Prepare request data
            request_data = {
                "image_path": str(self.input_path.absolute()),
                "output_dir": str(adapter_output_dir.absolute()),
                "multi_person": True,
                "cfg_path": "pretrained_models/smplest_x/config_base.py"
            }
            
            self.logger.info(f"Calling SMPLest-X service: {url}/predict")
            start_time = time.time()
            
            response = requests.post(f"{url}/predict", json=request_data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                message = f"Success: {result['message']} (HTTP: {processing_time:.1f}s, Model: {result['processing_time']:.1f}s)"
                return "smplestx", True, message
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return "smplestx", False, error_msg
                
        except requests.exceptions.Timeout:
            return "smplestx", False, "HTTP request timeout (5 minutes)"
        except Exception as e:
            return "smplestx", False, f"HTTP request failed: {str(e)}"
    
    def call_wilor_service(self, adapter_output_dir: Path) -> Tuple[str, bool, str]:
        """Call WiLoR service via HTTP"""
        try:
            url = self.services['wilor']['url']
            
            # Prepare request data
            request_data = {
                "image_path": str(self.input_path.absolute()),
                "output_dir": str(adapter_output_dir.absolute()),
                "rescale_factor": 2.0,
                "save_mesh": False
            }
            
            self.logger.info(f"Calling WiLoR service: {url}/predict")
            start_time = time.time()
            
            response = requests.post(f"{url}/predict", json=request_data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                message = f"Success: {result['message']} (HTTP: {processing_time:.1f}s, Model: {result['processing_time']:.1f}s)"
                return "wilor", True, message
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return "wilor", False, error_msg
                
        except requests.exceptions.Timeout:
            return "wilor", False, "HTTP request timeout (5 minutes)"
        except Exception as e:
            return "wilor", False, f"HTTP request failed: {str(e)}"
    
    def call_emoca_service(self, adapter_output_dir: Path) -> Tuple[str, bool, str]:
        """Call EMOCA service via HTTP"""
        try:
            url = self.services['emoca']['url']
            
            # Prepare request data
            request_data = {
                "image_path": str(self.input_path.absolute()),
                "output_dir": str(adapter_output_dir.absolute()),
                "model_name": "EMOCA_v2_lr_mse_20",
                "mode": "detail",
                "save_images": True,
                "save_codes": True,
                "save_mesh": False
            }
            
            self.logger.info(f"Calling EMOCA service: {url}/predict")
            start_time = time.time()
            
            response = requests.post(f"{url}/predict", json=request_data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                message = f"Success: {result['message']} (HTTP: {processing_time:.1f}s, Model: {result['processing_time']:.1f}s)"
                return "emoca", True, message
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                return "emoca", False, error_msg
                
        except requests.exceptions.Timeout:
            return "emoca", False, "HTTP request timeout (5 minutes)"
        except Exception as e:
            return "emoca", False, f"HTTP request failed: {str(e)}"
    
    def run_service_call(self, service_name: str) -> Tuple[str, bool, str]:
        """Run a single service call"""
        adapter_output_dir = self.run_output_dir / self.services[service_name]['output_subdir']
        adapter_output_dir.mkdir(exist_ok=True)
        
        # Map service calls
        service_calls = {
            'smplestx': self.call_smplestx_service,
            'wilor': self.call_wilor_service,
            'emoca': self.call_emoca_service
        }
        
        if service_name not in service_calls:
            return service_name, False, f"Unknown service: {service_name}"
            
        return service_calls[service_name](adapter_output_dir)
    
    def fallback_to_subprocess(self, service_name: str) -> Tuple[str, bool, str]:
        """Fallback to original subprocess method if service fails"""
        self.logger.warning(f"⚠️  Falling back to subprocess for {service_name}")
        
        # Import the original pipeline class
        from main import UnifiedPipeline
        
        # Create temporary original pipeline
        original_pipeline = UnifiedPipeline(
            input_path=str(self.input_path),
            output_dir=str(self.run_output_dir.parent),
            continue_on_error=self.continue_on_error,
            max_workers=1  # Run just this adapter
        )
        
        # Run just the specific adapter
        adapter_config = {
            'smplestx': {
                'script': 'adapters/smplestx_adapter.py',
                'output_subdir': 'smplestx_results',
                'args': original_pipeline._get_smplestx_args
            },
            'wilor': {
                'script': 'adapters/wilor_adapter.py',
                'output_subdir': 'wilor_results',
                'args': original_pipeline._get_wilor_args
            },
            'emoca': {
                'script': 'adapters/emoca_adapter.py',
                'output_subdir': 'emoca_results',
                'args': original_pipeline._get_emoca_args
            }
        }
        
        if service_name in adapter_config:
            return original_pipeline.run_adapter(service_name, adapter_config[service_name])
        else:
            return service_name, False, f"No fallback available for {service_name}"
    
    def run_pipeline(self) -> Dict[str, Tuple[bool, str]]:
        """Run all services concurrently with persistence optimization"""
        self.logger.info(f"Starting persistence-optimized pipeline for image: {self.input_path}")
        self.logger.info(f"Output directory: {self.run_output_dir}")
        
        # Ensure services are running
        self.logger.info("🔍 Checking service availability...")
        service_status = self.ensure_services_running()
        
        # Report service status
        running_services = [name for name, status in service_status.items() if status]
        failed_services = [name for name, status in service_status.items() if not status]
        
        if running_services:
            self.logger.info(f"✓ Running services: {', '.join(running_services)}")
        if failed_services:
            self.logger.warning(f"✗ Failed services: {', '.join(failed_services)}")
        
        results = {}
        
        # Run service calls concurrently
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit service call tasks
            future_to_service = {}
            
            for service_name in self.services.keys():
                if service_status[service_name]:
                    # Use service
                    future = executor.submit(self.run_service_call, service_name)
                    future_to_service[future] = (service_name, 'service')
                elif self.continue_on_error:
                    # Use subprocess fallback
                    future = executor.submit(self.fallback_to_subprocess, service_name)
                    future_to_service[future] = (service_name, 'fallback')
                else:
                    # Fail immediately
                    results[service_name] = (False, f"Service unavailable and no fallback allowed")
            
            # Collect results as they complete
            for future in as_completed(future_to_service):
                service_name, method = future_to_service[future]
                try:
                    name, success, message = future.result()
                    results[name] = (success, f"{message} [{method}]")
                    
                    if success:
                        self.logger.info(f"✓ {name} completed via {method}")
                    else:
                        self.logger.error(f"✗ {name} failed via {method}: {message}")
                        
                except Exception as e:
                    error_msg = f"Executor exception: {str(e)}"
                    results[service_name] = (False, f"{error_msg} [{method}]")
                    self.logger.error(f"Service {service_name} failed in executor: {error_msg}")
        
        # Generate summary
        self.generate_summary(results)
        return results
    
    def generate_summary(self, results: Dict[str, Tuple[bool, str]]):
        """Generate pipeline execution summary"""
        successful_adapters = [name for name, (success, _) in results.items() if success]
        failed_adapters = [name for name, (success, _) in results.items() if not success]
        
        # Create summary data
        summary = {
            "timestamp": self.timestamp,
            "input_image": str(self.input_path),
            "output_directory": str(self.run_output_dir),
            "execution_mode": "persistence_services",
            "successful_adapters": successful_adapters,
            "failed_adapters": failed_adapters,
            "total_adapters": len(results),
            "success_rate": len(successful_adapters) / len(results) if results else 0,
            "detailed_results": {name: {"success": success, "message": message} 
                               for name, (success, message) in results.items()}
        }
        
        # Save summary
        summary_file = self.run_output_dir / "pipeline_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log summary
        self.logger.info("="*60)
        self.logger.info("📊 PERSISTENCE PIPELINE SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Input: {self.input_path.name}")
        self.logger.info(f"Mode: Persistence Services")
        
        if successful_adapters:
            self.logger.info(f"✓ Success: {', '.join(successful_adapters)}")
        
        if failed_adapters:
            self.logger.info(f"✗ Failed: {', '.join(failed_adapters)}")
            
        self.logger.info(f"Summary saved to: {summary_file}")
        self.logger.info("="*60)

def parse_args():
    parser = argparse.ArgumentParser(description='Unified 3D Whole Body Pipeline - Persistence Services')
    parser.add_argument('--input_image', type=str, required=True, 
                       help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='unified_results',
                       help='Output directory for all results')
    parser.add_argument('--continue_on_error', action='store_true', default=True,
                       help='Continue pipeline even if some services fail')
    parser.add_argument('--max_workers', type=int, default=3,
                       help='Maximum number of concurrent service calls')
    parser.add_argument('--no_auto_start', action='store_true',
                       help='Disable auto-starting of services')
    parser.add_argument('--services_only', action='store_true',
                       help='Only use services, no subprocess fallback')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Validate input
    if not Path(args.input_image).exists():
        print(f"Error: Input image not found: {args.input_image}")
        sys.exit(1)
    
    # Create and run persistence pipeline
    pipeline = UnifiedPipelinePersistence(
        input_path=args.input_image,
        output_dir=args.output_dir,
        continue_on_error=args.continue_on_error and not args.services_only,
        max_workers=args.max_workers,
        auto_start_services=not args.no_auto_start
    )
    
    start_time = time.time()
    results = pipeline.run_pipeline()
    total_time = time.time() - start_time
    
    # Report timing
    print(f"\n⏱️  Total pipeline time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    # Exit with appropriate code
    failed_count = sum(1 for success, _ in results.values() if not success)
    if failed_count > 0 and not args.continue_on_error:
        sys.exit(1)

if __name__ == '__main__':
    main()