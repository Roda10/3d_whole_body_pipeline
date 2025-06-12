#!/usr/bin/env python3
"""
Service Management System
Handles starting, stopping, and monitoring of all persistence services
"""

import os
import sys
import time
import json
import signal
import subprocess
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional
import psutil

class ServiceManager:
    """Manages lifecycle of all 3D pose estimation services"""
    
    def __init__(self):
        self.services = {
            'smplestx': {
                'script': 'services/smplestx_service.py',
                'port': 8001,
                'url': 'http://localhost:8001',
                'name': 'SMPLest-X Service'
            },
            'wilor': {
                'script': 'services/wilor_service.py',
                'port': 8002,
                'url': 'http://localhost:8002',
                'name': 'WiLoR Service'
            },
            'emoca': {
                'script': 'services/emoca_service.py',
                'port': 8003,
                'url': 'http://localhost:8003',
                'name': 'EMOCA Service'
            }
        }
        
        self.pid_file = Path("services/.service_pids.json")
        self.log_dir = Path("services/logs")
        self.log_dir.mkdir(exist_ok=True)
    
    def save_pids(self, pids: Dict[str, int]):
        """Save service PIDs to file"""
        self.pid_file.parent.mkdir(exist_ok=True)
        with open(self.pid_file, 'w') as f:
            json.dump(pids, f, indent=2)
    
    def load_pids(self) -> Dict[str, int]:
        """Load service PIDs from file"""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is in use"""
        for conn in psutil.net_connections():
            if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
                return True
        return False
    
    def check_service_health(self, service_name: str) -> Dict[str, any]:
        """Check service health via HTTP"""
        try:
            url = self.services[service_name]['url']
            response = requests.get(f"{url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                return {
                    'status': 'healthy' if health_data.get('status') == 'healthy' else 'unhealthy',
                    'models_loaded': health_data.get('models_loaded', False),
                    'cuda_available': health_data.get('cuda_available', False),
                    'gpu_memory_gb': health_data.get('gpu_memory_gb', 0.0),
                    'response_time_ms': response.elapsed.total_seconds() * 1000
                }
            else:
                return {'status': 'unhealthy', 'error': f'HTTP {response.status_code}'}
                
        except requests.exceptions.ConnectionError:
            return {'status': 'unreachable', 'error': 'Connection refused'}
        except requests.exceptions.Timeout:
            return {'status': 'timeout', 'error': 'Request timeout'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def start_service(self, service_name: str, wait_for_ready: bool = True) -> bool:
        """Start a single service"""
        service_config = self.services[service_name]
        
        # Check if already running
        if self.is_port_in_use(service_config['port']):
            health = self.check_service_health(service_name)
            if health['status'] == 'healthy':
                print(f"✓ {service_config['name']} already running and healthy")
                return True
            else:
                print(f"⚠️  Port {service_config['port']} in use but service unhealthy")
                return False
        
        print(f"🚀 Starting {service_config['name']}...")
        
        # Setup logging
        log_file = self.log_dir / f"{service_name}.log"
        
        try:
            # Start service process
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    [sys.executable, service_config['script']],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create new process group
                )
            
            # Save PID
            pids = self.load_pids()
            pids[service_name] = process.pid
            self.save_pids(pids)
            
            if not wait_for_ready:
                print(f"✓ {service_config['name']} started (PID: {process.pid})")
                return True
            
            # Wait for service to be ready
            print(f"⏳ Waiting for {service_config['name']} to be ready...")
            
            for attempt in range(240):  # Wait up to 60 seconds
                time.sleep(1)
                
                # Check if process is still running
                if process.poll() is not None:
                    print(f"✗ {service_config['name']} process terminated unexpectedly")
                    print(f"   Check log file: {log_file}")
                    return False
                
                # Check service health
                health = self.check_service_health(service_name)
                if health['status'] == 'healthy':
                    models_status = "✓" if health.get('models_loaded') else "⚠️"
                    cuda_status = "🚀" if health.get('cuda_available') else "💻"
                    response_time = health.get('response_time_ms', 0)
                    
                    print(f"✅ {service_config['name']} ready!")
                    print(f"   {models_status} Models loaded | {cuda_status} CUDA | {response_time:.1f}ms response")
                    return True
                elif health['status'] == 'unreachable':
                    continue  # Still starting up
                else:
                    print(f"   Status: {health['status']} - {health.get('error', 'Unknown error')}")
            
            print(f"⏰ {service_config['name']} startup timeout (60s)")
            print(f"   Check log file: {log_file}")
            return False
            
        except Exception as e:
            print(f"✗ Failed to start {service_config['name']}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a single service"""
        service_config = self.services[service_name]
        pids = self.load_pids()
        
        pid = pids.get(service_name)
        if not pid:
            print(f"⚠️  No PID found for {service_config['name']}")
            return True
        
        try:
            # Check if process exists
            if not psutil.pid_exists(pid):
                print(f"✓ {service_config['name']} already stopped")
                # Clean up PID file
                del pids[service_name]
                self.save_pids(pids)
                return True
            
            print(f"🛑 Stopping {service_config['name']} (PID: {pid})...")
            
            # Send SIGTERM for graceful shutdown
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            
            # Wait for graceful shutdown
            for _ in range(10):  # Wait up to 10 seconds
                if not psutil.pid_exists(pid):
                    print(f"✓ {service_config['name']} stopped gracefully")
                    del pids[service_name]
                    self.save_pids(pids)
                    return True
                time.sleep(1)
            
            # Force kill if still running
            print(f"🔨 Force killing {service_config['name']}...")
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            
            time.sleep(1)
            if not psutil.pid_exists(pid):
                print(f"✓ {service_config['name']} force stopped")
                del pids[service_name]
                self.save_pids(pids)
                return True
            else:
                print(f"✗ Failed to stop {service_config['name']}")
                return False
                
        except ProcessLookupError:
            print(f"✓ {service_config['name']} already stopped")
            del pids[service_name]
            self.save_pids(pids)
            return True
        except Exception as e:
            print(f"✗ Error stopping {service_config['name']}: {e}")
            return False
    
    def start_all_services(self, wait_for_ready: bool = True) -> Dict[str, bool]:
        """Start all services"""
        print("🚀 Starting all persistence services...")
        print("="*50)
        
        results = {}
        for service_name in self.services.keys():
            results[service_name] = self.start_service(service_name, wait_for_ready)
            print()  # Add spacing between services
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print("="*50)
        print(f"📊 Startup Summary: {success_count}/{total_count} services running")
        
        if success_count == total_count:
            print("🎉 All services started successfully!")
        else:
            failed_services = [name for name, success in results.items() if not success]
            print(f"⚠️  Failed services: {', '.join(failed_services)}")
        
        return results
    
    def stop_all_services(self) -> Dict[str, bool]:
        """Stop all services"""
        print("🛑 Stopping all persistence services...")
        print("="*50)
        
        results = {}
        for service_name in self.services.keys():
            results[service_name] = self.stop_service(service_name)
        
        # Clean up PID file if all services stopped
        if all(results.values()):
            if self.pid_file.exists():
                self.pid_file.unlink()
        
        success_count = sum(results.values())
        total_count = len(results)
        
        print("="*50)
        print(f"📊 Shutdown Summary: {success_count}/{total_count} services stopped")
        
        return results
    
    def status_all_services(self) -> Dict[str, Dict]:
        """Get status of all services"""
        print("📊 Service Status Report")
        print("="*70)
        
        statuses = {}
        pids = self.load_pids()
        
        for service_name, service_config in self.services.items():
            print(f"\n🔍 {service_config['name']} (Port {service_config['port']})")
            print("-" * 40)
            
            # Check PID
            pid = pids.get(service_name)
            if pid and psutil.pid_exists(pid):
                process = psutil.Process(pid)
                print(f"   PID: {pid} | Status: {process.status()}")
                print(f"   CPU: {process.cpu_percent():.1f}% | Memory: {process.memory_info().rss / 1024**2:.1f}MB")
            else:
                print(f"   PID: Not found")
            
            # Check port
            port_in_use = self.is_port_in_use(service_config['port'])
            print(f"   Port: {'🟢 In use' if port_in_use else '🔴 Free'}")
            
            # Check service health
            health = self.check_service_health(service_name)
            status_icon = {
                'healthy': '🟢',
                'unhealthy': '🟡', 
                'unreachable': '🔴',
                'timeout': '⏰',
                'error': '❌'
            }.get(health['status'], '❓')
            
            print(f"   Health: {status_icon} {health['status'].title()}")
            
            if health['status'] == 'healthy':
                models_icon = "✅" if health.get('models_loaded') else "⚠️"
                cuda_icon = "🚀" if health.get('cuda_available') else "💻"
                response_time = health.get('response_time_ms', 0)
                gpu_memory = health.get('gpu_memory_gb', 0)
                
                print(f"   {models_icon} Models loaded | {cuda_icon} CUDA ({gpu_memory:.1f}GB)")
                print(f"   ⚡ Response time: {response_time:.1f}ms")
            elif 'error' in health:
                print(f"   Error: {health['error']}")
            
            statuses[service_name] = health
        
        print("\n" + "="*70)
        
        # Summary
        healthy_count = sum(1 for status in statuses.values() if status['status'] == 'healthy')
        total_count = len(statuses)
        
        if healthy_count == total_count:
            print("🎉 All services are healthy and ready!")
        else:
            print(f"⚠️  {healthy_count}/{total_count} services healthy")
        
        return statuses
    
    def restart_service(self, service_name: str) -> bool:
        """Restart a single service"""
        print(f"🔄 Restarting {self.services[service_name]['name']}...")
        
        # Stop first
        stop_success = self.stop_service(service_name)
        if not stop_success:
            print(f"⚠️  Failed to stop {service_name}, continuing anyway...")
        
        time.sleep(2)  # Brief pause
        
        # Start again
        return self.start_service(service_name)
    
    def restart_all_services(self) -> Dict[str, bool]:
        """Restart all services"""
        print("🔄 Restarting all persistence services...")
        
        # Stop all first
        self.stop_all_services()
        time.sleep(3)  # Brief pause
        
        # Start all again
        return self.start_all_services()

def main():
    parser = argparse.ArgumentParser(description='Persistence Services Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'], 
                       help='Action to perform')
    parser.add_argument('--service', type=str, choices=['smplestx', 'wilor', 'emoca'],
                       help='Specific service to target (default: all)')
    parser.add_argument('--no-wait', action='store_true',
                       help='Do not wait for services to be ready when starting')
    
    args = parser.parse_args()
    
    manager = ServiceManager()
    
    if args.action == 'start':
        if args.service:
            success = manager.start_service(args.service, not args.no_wait)
            sys.exit(0 if success else 1)
        else:
            results = manager.start_all_services(not args.no_wait)
            sys.exit(0 if all(results.values()) else 1)
    
    elif args.action == 'stop':
        if args.service:
            success = manager.stop_service(args.service)
            sys.exit(0 if success else 1)
        else:
            results = manager.stop_all_services()
            sys.exit(0 if all(results.values()) else 1)
    
    elif args.action == 'restart':
        if args.service:
            success = manager.restart_service(args.service)
            sys.exit(0 if success else 1)
        else:
            results = manager.restart_all_services()
            sys.exit(0 if all(results.values()) else 1)
    
    elif args.action == 'status':
        statuses = manager.status_all_services()
        healthy_count = sum(1 for status in statuses.values() if status['status'] == 'healthy')
        sys.exit(0 if healthy_count == len(statuses) else 1)

if __name__ == "__main__":
    main()