#!/usr/bin/env python3
"""
Parameter Structure Analyzer
Analyzes pipeline outputs to create clear documentation of what each model outputs
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Union

class ParameterStructureAnalyzer:
    """Analyzes and documents the structure of model outputs"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # Parameter definitions based on the code files
        self.parameter_definitions = {
            'smplestx': {
                'joints_3d': 'Body joints in 3D space (x,y,z coordinates)',
                'joints_2d': '2D projections of body joints on image (pixel coordinates)',
                'root_pose': 'Global body orientation (3 axis-angle rotations)',
                'body_pose': 'Body joint rotations (21 joints × 3 axis-angle each = 63 values)',
                'left_hand_pose': 'Left hand joint rotations (15 finger joints × 3 axis-angle each = 45 values)',
                'right_hand_pose': 'Right hand joint rotations (15 finger joints × 3 axis-angle each = 45 values)',
                'jaw_pose': 'Jaw rotation (3 axis-angle rotations)',
                'betas': 'Body shape parameters (10 values controlling height, weight, build)',
                'expression': 'Facial expression parameters (10 values controlling facial expressions)',
                'translation': 'Global position in 3D space (x,y,z offset)',
                'mesh': 'Full body mesh vertices (10475 vertices × 3D coordinates)'
            },
            'wilor': {
                'vertices_3d': 'Hand mesh vertices in 3D space (778 vertices per hand × x,y,z)',
                'keypoints_3d': 'Hand keypoints in 3D space (21 keypoints per hand × x,y,z)',
                'camera_translation': 'Camera position relative to hand (x,y,z)',
                'camera_prediction': 'Raw camera parameters from model',
                'box_center': 'Center of bounding box around hand in image',
                'box_size': 'Size of bounding box around hand',
                'mano_parameters': {
                    'pose_coefficients': 'MANO hand pose parameters (45 values = 15 joints × 3 rotations)',
                    'shape_coefficients': 'MANO hand shape parameters (10 values controlling hand size/shape)'
                }
            },
            'emoca': {
                'shapecode': 'Facial shape parameters (100D latent space controlling face geometry)',
                'expcode': 'Facial expression parameters (50D latent space controlling expressions)',
                'posecode': 'Head pose parameters (6D controlling head rotation and position)',
                'texcode': 'Facial texture parameters (50D latent space controlling skin appearance)',
                'detailcode': 'Facial detail parameters (128D latent space controlling fine details like wrinkles)'
            }
        }
        
    def analyze_file_structure(self, file_path: Path) -> Dict:
        """Analyze a single JSON file structure"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            structure = {}
            self._analyze_recursive(data, structure, "")
            return structure
            
        except Exception as e:
            return {'error': f'Could not analyze {file_path}: {str(e)}'}
    
    def _analyze_recursive(self, data: Any, structure: Dict, path: str):
        """Recursively analyze data structure"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self._analyze_recursive(value, structure, current_path)
        
        elif isinstance(data, list):
            # Analyze list structure
            if len(data) > 0:
                if isinstance(data[0], (int, float)):
                    # Numeric array
                    structure[path] = {
                        'type': 'numeric_array',
                        'shape': [len(data)],
                        'sample_values': data[:3] if len(data) >= 3 else data,
                        'value_range': [min(data), max(data)] if data else [0, 0]
                    }
                elif isinstance(data[0], list):
                    # Multi-dimensional array
                    array = np.array(data)
                    structure[path] = {
                        'type': 'multidimensional_array',
                        'shape': list(array.shape),
                        'sample_values': array.flatten()[:6].tolist(),
                        'value_range': [float(array.min()), float(array.max())]
                    }
                else:
                    # Continue recursion for first element
                    self._analyze_recursive(data[0], structure, f"{path}[0]")
            else:
                structure[path] = {'type': 'empty_list', 'shape': [0]}
        
        else:
            # Simple value
            structure[path] = {
                'type': type(data).__name__,
                'value': data
            }
    
    def analyze_smplestx_outputs(self) -> Dict:
        """Analyze SMPLest-X output structure"""
        analysis = {
            'model_name': 'SMPLest-X',
            'description': 'Full-body 3D human pose and shape estimation',
            'outputs': {},
            'files_found': []
        }
        
        # Find SMPLest-X parameter files
        for param_file in self.results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
            analysis['files_found'].append(str(param_file.relative_to(self.results_dir)))
            
            structure = self.analyze_file_structure(param_file)
            
            # Process each parameter
            for param_path, param_info in structure.items():
                if param_path.startswith('error'):
                    continue
                    
                param_name = param_path.split('.')[-1] if '.' in param_path else param_path
                
                if param_name not in analysis['outputs']:
                    analysis['outputs'][param_name] = {
                        'structure': param_info,
                        'meaning': self.parameter_definitions['smplestx'].get(param_name, 'Unknown parameter'),
                        'examples_found': 1
                    }
                else:
                    analysis['outputs'][param_name]['examples_found'] += 1
            
            # Only analyze first file for structure (assume others are similar)
            break
        
        return analysis
    
    def analyze_wilor_outputs(self) -> Dict:
        """Analyze WiLoR output structure"""
        analysis = {
            'model_name': 'WiLoR',
            'description': 'Hand pose estimation with MANO parameters',
            'outputs': {},
            'files_found': []
        }
        
        # Find WiLoR parameter files
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            analysis['files_found'].append(str(param_file.relative_to(self.results_dir)))
            
            structure = self.analyze_file_structure(param_file)
            
            # Process each parameter
            for param_path, param_info in structure.items():
                if param_path.startswith('error'):
                    continue
                
                # Handle nested paths like hands[0].vertices_3d
                param_name = param_path.split('.')[-1] if '.' in param_path else param_path
                
                if param_name not in analysis['outputs']:
                    # Get definition from nested structure if needed
                    if 'hands' in param_path and param_name in self.parameter_definitions['wilor']:
                        meaning = self.parameter_definitions['wilor'][param_name]
                    elif param_name in self.parameter_definitions['wilor'].get('mano_parameters', {}):
                        meaning = self.parameter_definitions['wilor']['mano_parameters'][param_name]
                    else:
                        meaning = self.parameter_definitions['wilor'].get(param_name, 'Unknown parameter')
                    
                    analysis['outputs'][param_name] = {
                        'structure': param_info,
                        'meaning': meaning,
                        'examples_found': 1
                    }
                else:
                    analysis['outputs'][param_name]['examples_found'] += 1
            
            break  # Analyze first file only
        
        return analysis
    
    def analyze_emoca_outputs(self) -> Dict:
        """Analyze EMOCA output structure"""
        analysis = {
            'model_name': 'EMOCA',
            'description': 'Facial expression and identity modeling',
            'outputs': {},
            'files_found': []
        }
        
        # Look for EMOCA code files
        code_files = list(self.results_dir.glob('emoca_results/*/*/*.json'))
        
        for code_file in code_files:
            if code_file.name in ['shape.json', 'exp.json', 'pose.json', 'tex.json', 'detail.json', 'codes.json']:
                analysis['files_found'].append(str(code_file.relative_to(self.results_dir)))
                
                structure = self.analyze_file_structure(code_file)
                
                # Map file names to parameter names
                file_to_param = {
                    'shape.json': 'shapecode',
                    'exp.json': 'expcode', 
                    'pose.json': 'posecode',
                    'tex.json': 'texcode',
                    'detail.json': 'detailcode'
                }
                
                if code_file.name in file_to_param:
                    param_name = file_to_param[code_file.name]
                    
                    # For single-parameter files, the structure is just the array
                    for key, value in structure.items():
                        analysis['outputs'][param_name] = {
                            'structure': value,
                            'meaning': self.parameter_definitions['emoca'].get(param_name, 'Unknown parameter'),
                            'source_file': code_file.name
                        }
                        break  # Take first (and usually only) key
                
                elif code_file.name == 'codes.json':
                    # Combined codes file
                    for param_path, param_info in structure.items():
                        param_name = param_path.split('.')[-1] if '.' in param_path else param_path
                        
                        if param_name in self.parameter_definitions['emoca']:
                            analysis['outputs'][param_name] = {
                                'structure': param_info,
                                'meaning': self.parameter_definitions['emoca'][param_name],
                                'source_file': 'codes.json'
                            }
        
        return analysis
    
    def create_human_readable_summary(self, all_analyses: Dict) -> Dict:
        """Create a human-readable summary for outsiders"""
        
        summary = {
            'pipeline_overview': {
                'description': '3D Whole Body Human Analysis Pipeline',
                'models_used': 3,
                'what_it_does': 'Takes a single image and extracts detailed 3D human body, hand, and facial parameters'
            },
            'model_outputs': {},
            'fusion_possibilities': {
                'coordinate_systems': 'Different models use different 3D coordinate systems',
                'overlapping_parts': {
                    'hands': 'SMPLest-X provides basic hand pose, WiLoR provides detailed hand mesh',
                    'face': 'SMPLest-X provides basic expression, EMOCA provides detailed facial parameters'
                },
                'fusion_strategy': 'Use SMPLest-X as body foundation, enhance with WiLoR hands and EMOCA face'
            }
        }
        
        for model_name, analysis in all_analyses.items():
            if not analysis.get('outputs'):
                continue
                
            model_summary = {
                'model_name': analysis['model_name'],
                'description': analysis['description'],
                'files_analyzed': len(analysis.get('files_found', [])),
                'parameters': {}
            }
            
            for param_name, param_info in analysis['outputs'].items():
                structure = param_info['structure']
                
                # Create simple explanation
                param_summary = {
                    'what_it_is': param_info['meaning'],
                    'data_structure': self._explain_structure(structure),
                    'technical_details': structure
                }
                
                model_summary['parameters'][param_name] = param_summary
            
            summary['model_outputs'][model_name] = model_summary
        
        return summary
    
    def _explain_structure(self, structure: Dict) -> str:
        """Convert technical structure to human explanation"""
        if structure.get('type') == 'numeric_array':
            shape = structure['shape']
            if len(shape) == 1:
                return f"List of {shape[0]} numbers"
            else:
                dims = ' × '.join(map(str, shape))
                return f"Array of {dims} numbers"
        
        elif structure.get('type') == 'multidimensional_array':
            shape = structure['shape']
            dims = ' × '.join(map(str, shape))
            return f"Multi-dimensional array: {dims}"
        
        elif 'shape' in structure:
            shape = structure['shape']
            if len(shape) == 1:
                return f"List of {shape[0]} values"
            elif len(shape) == 2:
                return f"Table with {shape[0]} rows and {shape[1]} columns"
            else:
                dims = ' × '.join(map(str, shape))
                return f"Multi-dimensional data: {dims}"
        
        else:
            return f"Single {structure.get('type', 'unknown')} value"
    
    def run_analysis(self):
        """Run complete parameter structure analysis"""
        print("="*80)
        print("PARAMETER STRUCTURE ANALYSIS")
        print("="*80)
        print(f"Analyzing pipeline outputs in: {self.results_dir}")
        print()
        
        # Analyze each model
        analyses = {}
        
        print("🔍 Analyzing SMPLest-X outputs...")
        analyses['smplestx'] = self.analyze_smplestx_outputs()
        
        print("🔍 Analyzing WiLoR outputs...")
        analyses['wilor'] = self.analyze_wilor_outputs()
        
        print("🔍 Analyzing EMOCA outputs...")
        analyses['emoca'] = self.analyze_emoca_outputs()
        
        # Create summary
        print("📝 Creating human-readable summary...")
        summary = self.create_human_readable_summary(analyses)
        
        # Save detailed analysis
        detailed_file = self.results_dir / 'parameter_structure_analysis.json'
        with open(detailed_file, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        # Save human-readable summary
        summary_file = self.results_dir / 'pipeline_output_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        self.print_summary(summary)
        
        print(f"\n💾 Analysis saved to:")
        print(f"   📄 Detailed: {detailed_file}")
        print(f"   📄 Summary: {summary_file}")
        print("="*80)
    
    def print_summary(self, summary: Dict):
        """Print a quick summary to console"""
        print("\n📊 SUMMARY:")
        print("-" * 50)
        
        for model_name, model_info in summary['model_outputs'].items():
            print(f"\n{model_info['model_name']}:")
            print(f"  {model_info['description']}")
            print(f"  Parameters found: {len(model_info['parameters'])}")
            
            for param_name, param_info in model_info['parameters'].items():
                structure_desc = param_info['data_structure']
                print(f"    • {param_name}: {structure_desc}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze pipeline parameter structures')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing pipeline results')
    
    args = parser.parse_args()
    
    analyzer = ParameterStructureAnalyzer(args.results_dir)
    analyzer.run_analysis()

if __name__ == '__main__':
    main()