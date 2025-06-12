#!/usr/bin/env python3
"""
Persistence-Compatible Coordinate System Analyzer
Fixed to work with persistence service output structure
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class PersistenceCoordinateAnalyzer:
    """Analyzes coordinate systems from persistence service outputs"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.analysis_results = {}
        print(f"🔍 Analyzing results in: {self.results_dir}")
        
    def load_smplestx_data(self) -> Dict:
        """Load SMPLest-X data with flexible path matching"""
        print("📥 Loading SMPLest-X data...")
        
        smplestx_data = {}
        
        # More flexible pattern matching for persistence services
        patterns = [
            'smplestx_results/*/person_*/smplx_params_person_*.json',
            'smplestx_results/inference_output_*/person_*/smplx_params_person_*.json',
            'smplestx_results/person_*/smplx_params_person_*.json'
        ]
        
        files_found = []
        for pattern in patterns:
            files_found.extend(list(self.results_dir.glob(pattern)))
        
        print(f"   🔍 Found {len(files_found)} SMPLest-X parameter files:")
        
        for param_file in files_found:
            print(f"      📄 {param_file.relative_to(self.results_dir)}")
            
            # Extract person ID from filename
            person_id = param_file.stem.replace('smplx_params_', '')
            
            with open(param_file, 'r') as f:
                params = json.load(f)
            
            smplestx_data[person_id] = {
                'joints_3d': np.array(params['joints_3d']),           # (137, 3)
                'mesh': np.array(params['mesh']),                     # (10475, 3)
                'translation': np.array(params['translation']),       # (3,)
                'body_pose': np.array(params['body_pose']),           # (63,)
                'left_hand_pose': np.array(params['left_hand_pose']), # (45,)
                'right_hand_pose': np.array(params['right_hand_pose']), # (45,)
                'betas': np.array(params['betas']),                   # (10,)
                'expression': np.array(params['expression'])          # (10,)
            }
            
            print(f"      ✅ Loaded {person_id}: {smplestx_data[person_id]['joints_3d'].shape[0]} joints, {smplestx_data[person_id]['mesh'].shape[0]} mesh vertices")
                
        return smplestx_data
    
    def load_wilor_data(self) -> Dict:
        """Load WiLoR data with flexible path matching"""
        print("📥 Loading WiLoR data...")
        
        wilor_data = {}
        
        # More flexible pattern matching
        patterns = [
            'wilor_results/*_parameters.json',
            'wilor_results/*/*_parameters.json'
        ]
        
        files_found = []
        for pattern in patterns:
            files_found.extend(list(self.results_dir.glob(pattern)))
            
        print(f"   🔍 Found {len(files_found)} WiLoR parameter files:")
        
        for param_file in files_found:
            print(f"      📄 {param_file.relative_to(self.results_dir)}")
            
            image_name = param_file.stem.replace('_parameters', '')
            
            with open(param_file, 'r') as f:
                params = json.load(f)
            
            hands_data = params.get('hands', [])
            wilor_data[image_name] = {
                'hands': hands_data,
                'metadata': params.get('metadata', {})
            }
            
            # Extract and organize coordinate data
            all_vertices = []
            all_keypoints = []
            hand_details = []
            
            for hand in hands_data:
                if 'vertices_3d' in hand and isinstance(hand['vertices_3d'], list):
                    vertices = np.array(hand['vertices_3d'])
                    keypoints = np.array(hand['keypoints_3d'])
                    
                    all_vertices.append(vertices)
                    all_keypoints.append(keypoints)
                    
                    hand_details.append({
                        'hand_type': hand['hand_type'],
                        'vertices': vertices,
                        'keypoints': keypoints,
                        'camera_translation': np.array(hand['camera_translation'])
                    })
            
            if all_vertices:
                wilor_data[image_name]['all_vertices'] = np.concatenate(all_vertices, axis=0)
                wilor_data[image_name]['all_keypoints'] = np.concatenate(all_keypoints, axis=0)
                wilor_data[image_name]['hand_details'] = hand_details
                
                print(f"      ✅ Loaded {image_name}: {len(hands_data)} hands, {len(all_vertices)} hand meshes")
                for i, hand in enumerate(hand_details):
                    print(f"         🖐️  {hand['hand_type']} hand: {hand['vertices'].shape[0]} vertices, {hand['keypoints'].shape[0]} keypoints")
            
        return wilor_data
    
    def load_emoca_data(self) -> Dict:
        """Load EMOCA data with flexible path matching"""
        print("📥 Loading EMOCA data...")
        
        emoca_data = {}
        
        # More flexible pattern matching for EMOCA
        patterns = [
            'emoca_results/*/codes.json',
            'emoca_results/*/*/codes.json',
            'emoca_results/EMOCA_*/test*/codes.json',
            'emoca_results/test*/codes.json'  # New pattern for persistence services
        ]
        
        files_found = []
        for pattern in patterns:
            found = list(self.results_dir.glob(pattern))
            files_found.extend(found)
            if found:
                print(f"   🔍 Pattern '{pattern}' found {len(found)} files")
        
        print(f"   🔍 Found {len(files_found)} EMOCA code files:")
        
        for codes_file in files_found:
            print(f"      📄 {codes_file.relative_to(self.results_dir)}")
            
            image_name = codes_file.parent.name
            
            with open(codes_file, 'r') as f:
                codes = json.load(f)
            
            # Handle different EMOCA output formats
            if 'shapecode' in codes:
                # Direct codes format
                emoca_data[image_name] = {
                    'shapecode': np.array(codes['shapecode']),     # (100,)
                    'expcode': np.array(codes['expcode']),         # (50,)
                    'texcode': np.array(codes['texcode']),         # (50,)
                    'posecode': np.array(codes['posecode']),       # (6,)
                    'detailcode': np.array(codes['detailcode'])    # (128,)
                }
            elif 'expression' in codes:
                # Individual parameter format
                exp_data = codes['expression']
                if isinstance(exp_data, list):
                    emoca_data[image_name] = {
                        'expcode': np.array(exp_data)
                    }
            
            print(f"      ✅ Loaded {image_name}: {len(codes)} parameter sets")
            break  # Take first valid file found
                
        # Try individual files if combined not found
        if not emoca_data:
            print("   🔍 Trying individual EMOCA parameter files...")
            for emoca_subdir in self.results_dir.glob('emoca_results/test*/'):
                image_name = emoca_subdir.name
                print(f"      📁 Checking {image_name}/")
                
                code_files = {
                    'shapecode': emoca_subdir / 'shape.json',
                    'expcode': emoca_subdir / 'expression.json',
                    'texcode': emoca_subdir / 'texture.json',
                    'posecode': emoca_subdir / 'pose.json',
                    'detailcode': emoca_subdir / 'detail.json'
                }
                
                codes = {}
                files_found = 0
                
                for code_type, file_path in code_files.items():
                    if file_path.exists():
                        print(f"         📄 Found {file_path.name}")
                        with open(file_path, 'r') as f:
                            file_data = json.load(f)
                            # Handle different JSON structures
                            if isinstance(file_data, dict) and code_type in file_data:
                                codes[code_type] = np.array(file_data[code_type])
                            elif isinstance(file_data, dict) and len(file_data) == 1:
                                # Single key-value pair
                                key, value = list(file_data.items())[0]
                                codes[code_type] = np.array(value)
                            elif isinstance(file_data, list):
                                # Direct array
                                codes[code_type] = np.array(file_data)
                            files_found += 1
                
                if files_found >= 2:  # Need at least expression + another param
                    emoca_data[image_name] = codes
                    print(f"      ✅ Loaded {image_name}: {files_found} code files")
                    break
        
        return emoca_data

    # [Rest of the analysis methods remain the same as the original...]
    # I'll include the key ones that are needed:
    
    def analyze_smplestx_coordinate_system(self, smplestx_data: Dict) -> Dict:
        """Comprehensive analysis of SMPLest-X coordinate system"""
        print("🔍 Analyzing SMPLest-X coordinate system...")
        
        all_joints = []
        all_mesh_vertices = []
        all_translations = []
        
        for person_id, data in smplestx_data.items():
            all_joints.append(data['joints_3d'])
            all_mesh_vertices.append(data['mesh'])
            all_translations.append(data['translation'])
        
        if not all_joints:
            return {'error': 'No SMPLest-X data found'}
        
        all_joints = np.concatenate(all_joints, axis=0)
        all_mesh_vertices = np.concatenate(all_mesh_vertices, axis=0)
        all_translations = np.array(all_translations)
        
        # Detailed coordinate analysis
        analysis = {
            'coordinate_system': 'SMPL-X_Full_Body',
            'total_points_analyzed': {
                'joints': all_joints.shape[0],
                'mesh_vertices': all_mesh_vertices.shape[0]
            },
            'coordinate_statistics': {
                'joints': {
                    'mean': all_joints.mean(axis=0).tolist(),
                    'std': all_joints.std(axis=0).tolist(),
                    'min': all_joints.min(axis=0).tolist(),
                    'max': all_joints.max(axis=0).tolist(),
                    'range': (all_joints.max(axis=0) - all_joints.min(axis=0)).tolist(),
                    'centroid': all_joints.mean(axis=0).tolist()
                }
            },
            'scale_metrics': {
                'overall_scale': float(np.std(all_joints)),
                'body_height_estimate': float(all_joints[:, 1].max() - all_joints[:, 1].min()),
            }
        }
        
        print(f"   📊 Analyzed {analysis['total_points_analyzed']['joints']:,} joints")
        return analysis
    
    def analyze_wilor_coordinate_system(self, wilor_data: Dict) -> Dict:
        """Comprehensive analysis of WiLoR coordinate system"""
        print("🔍 Analyzing WiLoR coordinate system...")
        
        all_vertices = []
        hand_analyses = []
        
        for image_name, data in wilor_data.items():
            if 'hand_details' in data:
                for hand in data['hand_details']:
                    all_vertices.append(hand['vertices'])
        
        if not all_vertices:
            return {'error': 'No WiLoR coordinate data found'}
        
        all_vertices = np.concatenate(all_vertices, axis=0)
        
        analysis = {
            'coordinate_system': 'WiLoR_Hand_Centric',
            'coordinate_statistics': {
                'vertices': {
                    'centroid': all_vertices.mean(axis=0).tolist()
                }
            },
            'scale_metrics': {
                'overall_scale': float(np.std(all_vertices)),
            }
        }
        
        print(f"   📊 Analyzed {all_vertices.shape[0]} hand vertices")
        return analysis
    
    def calculate_transformation_parameters(self, smplestx_analysis: Dict, wilor_analysis: Dict) -> Dict:
        """Calculate transformation parameters"""
        print("🧮 Calculating transformation parameters...")
        
        smplx_centroid = np.array(smplestx_analysis['coordinate_statistics']['joints']['centroid'])
        smplx_scale = smplestx_analysis['scale_metrics']['overall_scale']
        
        wilor_centroid = np.array(wilor_analysis['coordinate_statistics']['vertices']['centroid'])
        wilor_scale = wilor_analysis['scale_metrics']['overall_scale']
        
        translation = smplx_centroid - wilor_centroid
        scale_factor = smplx_scale / wilor_scale if wilor_scale > 0 else 1.0
        
        transformation = {
            'scale_analysis': {
                'scale_ratio': float(scale_factor)
            },
            'translation_vector': {
                'xyz_translation': translation.tolist()
            }
        }
        
        print(f"   📐 Scale factor: {scale_factor:.4f}")
        print(f"   📍 Translation: {translation}")
        
        return transformation
    
    def save_coordinate_analysis_summary(self, transformation: Dict):
        """Save the essential coordinate analysis summary for fusion"""
        summary = {
            'analysis_timestamp': str(np.datetime64('now')),
            'transformation_parameters': {
                'scale_factor': transformation['scale_analysis']['scale_ratio'],
                'translation_vector': transformation['translation_vector']['xyz_translation']
            }
        }
        
        output_file = self.results_dir / 'coordinate_analysis_summary.json'
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   📄 Saved: {output_file}")
    
    def run_analysis(self):
        """Execute the coordinate system analysis"""
        print("🚀 PERSISTENCE COORDINATE SYSTEM ANALYSIS")
        print("="*50)
        
        # Load all data
        smplestx_data = self.load_smplestx_data()
        wilor_data = self.load_wilor_data()
        
        if not smplestx_data:
            print("❌ Error: No SMPLest-X data found")
            return
            
        if not wilor_data:
            print("❌ Error: No WiLoR data found")  
            return
        
        # Perform core analyses
        smplestx_analysis = self.analyze_smplestx_coordinate_system(smplestx_data)
        wilor_analysis = self.analyze_wilor_coordinate_system(wilor_data)
        transformation = self.calculate_transformation_parameters(smplestx_analysis, wilor_analysis)
        
        # Save results
        self.save_coordinate_analysis_summary(transformation)
        
        print("\n✅ ANALYSIS COMPLETE!")
        print(f"📊 Key Results:")
        print(f"   Scale factor: {transformation['scale_analysis']['scale_ratio']:.4f}")
        print(f"   Translation: {transformation['translation_vector']['xyz_translation']}")
        print(f"📄 Results saved to: coordinate_analysis_summary.json")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3 or sys.argv[1] != '--results_dir':
        print("Usage: python persistence_coordinate_analyzer.py --results_dir /path/to/results")
        sys.exit(1)
    
    results_dir = sys.argv[2]
    analyzer = PersistenceCoordinateAnalyzer(results_dir)
    analyzer.run_analysis()
