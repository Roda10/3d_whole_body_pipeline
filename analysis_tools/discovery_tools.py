#!/usr/bin/env python3
"""
Pipeline Output Discovery Tool
Discovers and analyzes what files are actually produced by each adapter
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import pickle

class OutputDiscovery:
    """Discovers and analyzes pipeline outputs"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
    def discover_all_files(self) -> Dict:
        """Discover all files in the results directory"""
        discovery = {
            'total_files': 0,
            'directories': {},
            'file_types': {},
            'detailed_inventory': {}
        }
        
        if not self.results_dir.exists():
            print(f"Results directory does not exist: {self.results_dir}")
            return discovery
            
        # Walk through all directories
        for root, dirs, files in os.walk(self.results_dir):
            root_path = Path(root)
            relative_path = root_path.relative_to(self.results_dir)
            
            if files:  # Only process directories with files
                discovery['directories'][str(relative_path)] = {
                    'file_count': len(files),
                    'files': [],
                    'subdirs': dirs
                }
                
                for file in files:
                    file_path = root_path / file
                    file_info = self._analyze_file(file_path)
                    discovery['directories'][str(relative_path)]['files'].append(file_info)
                    discovery['total_files'] += 1
                    
                    # Track file types
                    ext = file_info['extension']
                    if ext not in discovery['file_types']:
                        discovery['file_types'][ext] = 0
                    discovery['file_types'][ext] += 1
                    
        return discovery
    
    def _analyze_file(self, file_path: Path) -> Dict:
        """Analyze a single file"""
        file_info = {
            'name': file_path.name,
            'extension': file_path.suffix.lower(),
            'size_bytes': file_path.stat().st_size,
            'size_mb': round(file_path.stat().st_size / (1024*1024), 2),
            'is_readable': True,
            'content_preview': None,
            'structure': None
        }
        
        try:
            # Try to analyze content based on file type
            if file_info['extension'] == '.json':
                file_info.update(self._analyze_json(file_path))
            elif file_info['extension'] == '.log':
                file_info.update(self._analyze_log(file_path))
            elif file_info['extension'] in ['.pkl', '.pickle']:
                file_info.update(self._analyze_pickle(file_path))
            elif file_info['extension'] == '.obj':
                file_info.update(self._analyze_obj(file_path))
            elif file_info['extension'] in ['.jpg', '.png', '.jpeg']:
                file_info['content_type'] = 'image'
            else:
                file_info.update(self._analyze_text(file_path))
                
        except Exception as e:
            file_info['is_readable'] = False
            file_info['error'] = str(e)
            
        return file_info
    
    def _analyze_json(self, file_path: Path) -> Dict:
        """Analyze JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            return {
                'content_type': 'json',
                'keys': list(data.keys()) if isinstance(data, dict) else None,
                'structure': self._describe_structure(data),
                'content_preview': str(data)[:200] + '...' if len(str(data)) > 200 else str(data)
            }
        except Exception as e:
            return {'content_type': 'json', 'error': str(e)}
    
    def _analyze_log(self, file_path: Path) -> Dict:
        """Analyze log file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            return {
                'content_type': 'log',
                'line_count': len(lines),
                'first_lines': lines[:3] if lines else [],
                'last_lines': lines[-3:] if len(lines) > 3 else lines
            }
        except Exception as e:
            return {'content_type': 'log', 'error': str(e)}
    
    def _analyze_pickle(self, file_path: Path) -> Dict:
        """Analyze pickle file"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            return {
                'content_type': 'pickle',
                'data_type': type(data).__name__,
                'structure': self._describe_structure(data),
                'content_preview': str(data)[:200] + '...' if len(str(data)) > 200 else str(data)
            }
        except Exception as e:
            return {'content_type': 'pickle', 'error': str(e)}
    
    def _analyze_obj(self, file_path: Path) -> Dict:
        """Analyze OBJ mesh file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            vertex_count = sum(1 for line in lines if line.strip().startswith('v '))
            face_count = sum(1 for line in lines if line.strip().startswith('f '))
            
            return {
                'content_type': 'obj_mesh',
                'vertex_count': vertex_count,
                'face_count': face_count,
                'total_lines': len(lines)
            }
        except Exception as e:
            return {'content_type': 'obj_mesh', 'error': str(e)}
    
    def _analyze_text(self, file_path: Path) -> Dict:
        """Analyze generic text file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            return {
                'content_type': 'text',
                'char_count': len(content),
                'line_count': content.count('\n'),
                'content_preview': content[:200] + '...' if len(content) > 200 else content
            }
        except Exception as e:
            return {'content_type': 'text', 'error': str(e)}
    
    def _describe_structure(self, data) -> str:
        """Describe the structure of data"""
        if isinstance(data, dict):
            return f"dict with {len(data)} keys: {list(data.keys())[:5]}"
        elif isinstance(data, list):
            return f"list with {len(data)} items"
        elif hasattr(data, 'shape'):  # numpy array
            return f"array with shape {data.shape}"
        else:
            return f"{type(data).__name__}"
    
    def print_discovery_report(self, discovery: Dict):
        """Print a comprehensive discovery report"""
        print("="*80)
        print("PIPELINE OUTPUT DISCOVERY REPORT")
        print("="*80)
        print(f"Results directory: {self.results_dir}")
        print(f"Total files found: {discovery['total_files']}")
        print()
        
        # File type summary
        print("FILE TYPES SUMMARY:")
        print("-" * 40)
        for ext, count in sorted(discovery['file_types'].items()):
            print(f"  {ext or '(no extension)'}: {count} files")
        print()
        
        # Directory breakdown
        print("DIRECTORY BREAKDOWN:")
        print("-" * 40)
        for dir_path, dir_info in discovery['directories'].items():
            print(f"\n📁 {dir_path}/")
            print(f"   Files: {dir_info['file_count']}")
            
            # Group files by type
            files_by_type = {}
            for file_info in dir_info['files']:
                ext = file_info['extension']
                if ext not in files_by_type:
                    files_by_type[ext] = []
                files_by_type[ext].append(file_info)
            
            for ext, files in files_by_type.items():
                print(f"   {ext or '(no ext)'}: {len(files)} files")
                for file_info in files[:3]:  # Show first 3 files
                    print(f"     📄 {file_info['name']} ({file_info['size_mb']} MB)")
                    if file_info.get('structure'):
                        print(f"        Structure: {file_info['structure']}")
                    if file_info.get('content_preview'):
                        preview = file_info['content_preview'].replace('\n', ' ')[:100]
                        print(f"        Preview: {preview}")
                if len(files) > 3:
                    print(f"     ... and {len(files) - 3} more")
        
        print("\n" + "="*80)
    
    def identify_parameter_files(self, discovery: Dict) -> Dict:
        """Identify which files likely contain model parameters"""
        parameter_candidates = {
            'smplestx': [],
            'wilor': [],
            'emoca': []
        }
        
        for dir_path, dir_info in discovery['directories'].items():
            # Determine which model this directory belongs to
            model_type = None
            if 'smplestx' in dir_path.lower():
                model_type = 'smplestx'
            elif 'wilor' in dir_path.lower():
                model_type = 'wilor'
            elif 'emoca' in dir_path.lower():
                model_type = 'emoca'
            
            if model_type:
                for file_info in dir_info['files']:
                    # Look for files that likely contain parameters
                    if file_info['extension'] in ['.json', '.pkl', '.pickle']:
                        file_path = self.results_dir / dir_path / file_info['name']
                        parameter_candidates[model_type].append({
                            'file_path': str(file_path),
                            'file_info': file_info
                        })
        
        return parameter_candidates
    
    def suggest_next_steps(self, discovery: Dict, parameter_candidates: Dict):
        """Suggest next steps based on discovery"""
        print("\nNEXT STEPS RECOMMENDATIONS:")
        print("-" * 40)
        
        for model_type, candidates in parameter_candidates.items():
            print(f"\n{model_type.upper()}:")
            if not candidates:
                print("  ❌ No parameter files found")
                print(f"  💡 Check if {model_type} adapter ran successfully")
                continue
            
            print(f"  ✅ Found {len(candidates)} potential parameter files:")
            for candidate in candidates:
                file_info = candidate['file_info']
                print(f"     📄 {file_info['name']}")
                if file_info.get('keys'):
                    print(f"        Keys: {file_info['keys'][:5]}")
                if file_info.get('structure'):
                    print(f"        Structure: {file_info['structure']}")
        
        # Overall recommendations
        print("\nGENERAL RECOMMENDATIONS:")
        if discovery['total_files'] == 0:
            print("  ❌ No files found - check if pipeline ran successfully")
        elif not any(parameter_candidates.values()):
            print("  ⚠️  No parameter files found - may need to modify adapters to save parameters")
        else:
            print("  ✅ Parameter files found - can proceed with coordinate analysis")
            print("  📝 Next: Examine specific parameter file contents")
            print("  🔧 Then: Modify coordinate analyzer to read these specific formats")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Discover pipeline output files')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing pipeline results')
    
    args = parser.parse_args()
    
    discoverer = OutputDiscovery(args.results_dir)
    discovery = discoverer.discover_all_files()
    discoverer.print_discovery_report(discovery)
    
    parameter_candidates = discoverer.identify_parameter_files(discovery)
    discoverer.suggest_next_steps(discovery, parameter_candidates)
    
    # Save discovery report
    output_file = Path(args.results_dir) / 'output_discovery.json'
    with open(output_file, 'w') as f:
        json.dump({
            'discovery': discovery,
            'parameter_candidates': parameter_candidates
        }, f, indent=2)
    
    print(f"\n📁 Full discovery report saved to: {output_file}")

if __name__ == '__main__':
    main()