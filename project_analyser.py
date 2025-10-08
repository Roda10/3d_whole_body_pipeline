#!/usr/bin/env python3
"""
Deep Project Structure Analysis Tool
Analyzes code structure, dependencies, data flows, and architectural patterns
"""

import os
import json
import ast
from pathlib import Path
from collections import defaultdict
import re

class ProjectAnalyzer:
    def __init__(self, project_root):
        self.root = Path(project_root)
        self.structure = {
            'modules': {},
            'data_flows': [],
            'dependencies': defaultdict(set),
            'class_hierarchy': {},
            'function_calls': defaultdict(list),
            'file_purposes': {}
        }
        
    def analyze_python_file(self, filepath):
        """Deep AST analysis of Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
                
            analysis = {
                'path': str(filepath),
                'imports': [],
                'classes': [],
                'functions': [],
                'globals': [],
                'docstring': ast.get_docstring(tree),
                'key_operations': []
            }
            
            for node in ast.walk(tree):
                # Imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        analysis['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        analysis['imports'].append(f"{module}.{alias.name}")
                
                # Classes
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [self._get_name(base) for base in node.bases],
                        'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        'docstring': ast.get_docstring(node)
                    }
                    analysis['classes'].append(class_info)
                
                # Functions
                elif isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'returns': self._get_name(node.returns) if node.returns else None,
                        'docstring': ast.get_docstring(node),
                        'calls': []
                    }
                    
                    # Find function calls within this function
                    for subnode in ast.walk(node):
                        if isinstance(subnode, ast.Call):
                            call_name = self._get_call_name(subnode)
                            if call_name:
                                func_info['calls'].append(call_name)
                    
                    analysis['functions'].append(func_info)
                
                # Key operations (file operations, model loading, etc.)
                elif isinstance(node, ast.Call):
                    call_name = self._get_call_name(node)
                    if call_name in ['open', 'json.load', 'json.dump', 'torch.load', 
                                    'cv2.imread', 'cv2.imwrite', 'np.load', 'np.save']:
                        analysis['key_operations'].append(call_name)
            
            return analysis
            
        except Exception as e:
            return {'path': str(filepath), 'error': str(e)}
    
    def _get_name(self, node):
        """Extract name from AST node"""
        if node is None:
            return None
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(type(node).__name__)
    
    def _get_call_name(self, node):
        """Extract function call name"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return f"{self._get_name(node.func.value)}.{node.func.attr}"
        return None
    
    def scan_directory(self, target_dirs=None):
        """Scan specific directories only"""
        if target_dirs is None:
            target_dirs = ['adapters', 'analysis_tools', 'evaluation', 'fusion']
            
        python_files = []
        json_files = []
        
        for dir_name in target_dirs:
            dir_path = self.root / dir_name
            if dir_path.exists():
                for item in dir_path.rglob('*.py'):
                    python_files.append(item)
                for item in dir_path.rglob('*.json'):
                    json_files.append(item)
        
        return python_files, json_files
    
    def identify_data_flows(self, analyses):
        """Identify data flow patterns"""
        flows = []
        
        # Look for save/load patterns
        for analysis in analyses:
            file_path = analysis['path']
            
            # Check for data saving
            if any('dump' in op or 'save' in op or 'write' in op 
                   for op in analysis.get('key_operations', [])):
                flows.append({
                    'type': 'data_producer',
                    'file': file_path,
                    'operations': analysis['key_operations']
                })
            
            # Check for data loading
            if any('load' in op or 'read' in op 
                   for op in analysis.get('key_operations', [])):
                flows.append({
                    'type': 'data_consumer',
                    'file': file_path,
                    'operations': analysis['key_operations']
                })
        
        return flows
    
    def analyze_io_patterns(self, analysis):
        """Extract input/output patterns from code"""
        io_info = {
            'inputs': [],
            'outputs': [],
            'data_structures': []
        }
        
        # Look for file operations
        for func in analysis.get('functions', []):
            # Input patterns
            if any(keyword in func['name'].lower() for keyword in ['load', 'read', 'parse']):
                io_info['inputs'].append({
                    'function': func['name'],
                    'args': func['args']
                })
            
            # Output patterns
            if any(keyword in func['name'].lower() for keyword in ['save', 'write', 'dump', 'export']):
                io_info['outputs'].append({
                    'function': func['name'],
                    'args': func['args']
                })
            
            # Look for dictionary/data structure definitions
            if 'dict' in str(func['calls']) or 'json' in str(func['calls']):
                io_info['data_structures'].append(func['name'])
        
        return io_info
        """Categorize files by purpose"""
        categories = {
            'adapters': [],
            'models': [],
            'utilities': [],
            'analysis_tools': [],
            'configuration': [],
            'main_pipeline': [],
            'tests': []
        }
        
        for analysis in analyses:
            path = Path(analysis['path'])
            rel_path = path.relative_to(self.root)
            
            # Categorize by directory
            if 'adapter' in str(rel_path).lower():
                categories['adapters'].append(analysis)
            elif 'model' in str(rel_path).lower():
                categories['models'].append(analysis)
            elif 'utils' in str(rel_path).lower() or 'util' in str(rel_path).lower():
                categories['utilities'].append(analysis)
            elif 'analysis' in str(rel_path).lower():
                categories['analysis_tools'].append(analysis)
            elif 'config' in str(rel_path).lower():
                categories['configuration'].append(analysis)
            elif 'test' in str(rel_path).lower():
                categories['tests'].append(analysis)
            elif path.name in ['main.py', 'pipeline.py', 'run.py']:
                categories['main_pipeline'].append(analysis)
        
        return categories
    
    def generate_report(self):
        """Generate focused I/O analysis report"""
        print("\n" + "="*80)
        print("🔬 FOCUSED PROJECT ANALYSIS: adapters, analysis_tools, evaluation, fusion")
        print("="*80)
        
        python_files, json_files = self.scan_directory()
        
        print(f"\n📂 Found: {len(python_files)} Python files, {len(json_files)} JSON files")
        
        # Analyze each file
        print("\n" + "="*80)
        print("📄 FILE-BY-FILE ANALYSIS")
        print("="*80)
        
        analyses_by_dir = defaultdict(list)
        
        for py_file in sorted(python_files):
            rel_path = py_file.relative_to(self.root)
            dir_name = rel_path.parts[0]
            
            analysis = self.analyze_python_file(py_file)
            io_info = self.analyze_io_patterns(analysis)
            
            analyses_by_dir[dir_name].append({
                'file': rel_path,
                'analysis': analysis,
                'io': io_info
            })
        
        # Print organized by directory
        for dir_name in ['adapters', 'analysis_tools', 'evaluation', 'fusion']:
            if dir_name in analyses_by_dir:
                print(f"\n{'='*80}")
                print(f"📁 {dir_name.upper()}/")
                print('='*80)
                
                for item in analyses_by_dir[dir_name]:
                    analysis = item['analysis']
                    io_info = item['io']
                    
                    print(f"\n📄 {item['file'].name}")
                    print(f"   Path: {item['file']}")
                    
                    # Purpose
                    if analysis.get('docstring'):
                        print(f"   Purpose: {analysis['docstring'][:120]}")
                    
                    # Main classes
                    if analysis.get('classes'):
                        for cls in analysis['classes']:
                            print(f"   Class: {cls['name']}")
                            if cls['methods']:
                                print(f"      Methods: {', '.join(cls['methods'][:5])}")
                    
                    # Key functions
                    if analysis.get('functions'):
                        key_funcs = [f for f in analysis['functions'] 
                                    if not f['name'].startswith('_')][:3]
                        if key_funcs:
                            print(f"   Functions:")
                            for func in key_funcs:
                                args_str = ', '.join(func['args'][:3])
                                print(f"      • {func['name']}({args_str})")
                    
                    # I/O patterns
                    if io_info['inputs']:
                        print(f"   📥 Inputs: {', '.join([i['function'] for i in io_info['inputs']])}")
                    if io_info['outputs']:
                        print(f"   📤 Outputs: {', '.join([o['function'] for o in io_info['outputs']])}")
                    
                    # Key operations
                    if analysis.get('key_operations'):
                        ops = set(analysis['key_operations'])
                        print(f"   🔧 Operations: {', '.join(sorted(ops))}")
        
        # Save detailed JSON
        output = {}
        for dir_name, items in analyses_by_dir.items():
            output[dir_name] = {}
            for item in items:
                filename = item['file'].name
                output[dir_name][filename] = {
                    'path': str(item['file']),
                    'purpose': item['analysis'].get('docstring'),
                    'classes': [c['name'] for c in item['analysis'].get('classes', [])],
                    'functions': [f['name'] for f in item['analysis'].get('functions', [])],
                    'inputs': item['io']['inputs'],
                    'outputs': item['io']['outputs'],
                    'imports': item['analysis'].get('imports', [])[:10]
                }
        
        output_file = self.root / 'focused_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n\n💾 Detailed JSON saved: {output_file}")
        
        return output

if __name__ == "__main__":
    import sys
    
    project_root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    analyzer = ProjectAnalyzer(project_root)
    output = analyzer.generate_report()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)