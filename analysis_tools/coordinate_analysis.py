#!/usr/bin/env python3
"""
Coordinate System Analysis Tool for 3D Model Fusion
Analyzes outputs from WiLoR, EMOCA, and SMPLest-X to understand coordinate systems
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class CoordinateAnalyzer:
    """Analyzes coordinate systems from different 3D human models"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.models = ['smplestx', 'wilor', 'emoca']
        self.coordinate_info = {}
        
    def load_model_outputs(self) -> Dict:
        """Load outputs from all three models"""
        outputs = {}
        
        # Load SMPLest-X outputs
        smplestx_dir = self.results_dir / 'smplestx_results'
        if smplestx_dir.exists():
            outputs['smplestx'] = self._load_smplestx_outputs(smplestx_dir)
            
        # Load WiLoR outputs  
        wilor_dir = self.results_dir / 'wilor_results'
        if wilor_dir.exists():
            outputs['wilor'] = self._load_wilor_outputs(wilor_dir)
            
        # Load EMOCA outputs
        emoca_dir = self.results_dir / 'emoca_results'
        if emoca_dir.exists():
            outputs['emoca'] = self._load_emoca_outputs(emoca_dir)
            
        return outputs
    
    def _load_smplestx_outputs(self, smplestx_dir: Path) -> Dict:
        """Load SMPLest-X parameter files"""
        smplestx_data = {}
        
        # Find parameter JSON files
        for person_dir in smplestx_dir.glob('person_*'):
            person_id = person_dir.name
            param_file = person_dir / f'smplx_params_{person_id}.json'
            
            if param_file.exists():
                with open(param_file, 'r') as f:
                    params = json.load(f)
                    
                smplestx_data[person_id] = {
                    'joints_3d': np.array(params['joints_3d']),
                    'joints_2d': np.array(params['joints_2d']), 
                    'translation': np.array(params['translation']),
                    'body_pose': np.array(params['body_pose']),
                    'left_hand_pose': np.array(params['left_hand_pose']),
                    'right_hand_pose': np.array(params['right_hand_pose']),
                    'betas': np.array(params['betas']),
                    'expression': np.array(params['expression'])
                }
                
        return smplestx_data
    
    def _load_wilor_outputs(self, wilor_dir: Path) -> Dict:
        """Load WiLoR parameter files"""
        wilor_data = {}
        
        # Find parameter JSON files
        for param_file in wilor_dir.glob('*_parameters.json'):
            with open(param_file, 'r') as f:
                params = json.load(f)
                
            image_name = param_file.stem.replace('_parameters', '')
            wilor_data[image_name] = {
                'vertices_3d': np.array(params['vertices_3d']) if 'vertices_3d' in params else None,
                'keypoints_3d': np.array(params['keypoints_3d']) if 'keypoints_3d' in params else None,
                'camera_translation': np.array(params['camera_translation']) if 'camera_translation' in params else None,
                'hand_poses': params.get('hand_poses', []),
                'hand_shapes': params.get('hand_shapes', [])
            }
                
        return wilor_data
    
    def _load_emoca_outputs(self, emoca_dir: Path) -> Dict:
        """Load EMOCA outputs (mesh files and codes)"""
        emoca_data = {}
        
        # Look for saved codes/parameters
        for code_file in emoca_dir.glob('*/codes.pkl'):
            # This would require loading EMOCA's pickle files
            # For now, we'll look for any JSON outputs or mesh files
            pass
            
        # Look for mesh files
        for mesh_file in emoca_dir.glob('*/*.obj'):
            image_name = mesh_file.parent.name
            emoca_data[image_name] = {
                'mesh_file': str(mesh_file),
                'has_geometry': True
            }
            
        return emoca_data
    
    def analyze_coordinate_systems(self, outputs: Dict) -> Dict:
        """Analyze coordinate systems from model outputs"""
        analysis = {}
        
        for model_name, model_data in outputs.items():
            if not model_data:
                continue
                
            analysis[model_name] = self._analyze_model_coordinates(model_name, model_data)
            
        return analysis
    
    def _analyze_model_coordinates(self, model_name: str, model_data: Dict) -> Dict:
        """Analyze coordinates for a specific model"""
        coord_analysis = {
            'model': model_name,
            'coordinate_ranges': {},
            'joint_positions': {},
            'coordinate_center': None,
            'scale_estimate': None
        }
        
        if model_name == 'smplestx':
            # Analyze SMPL-X coordinates
            all_joints = []
            all_translations = []
            
            for person_id, data in model_data.items():
                joints_3d = data['joints_3d']
                translation = data['translation']
                
                all_joints.append(joints_3d)
                all_translations.append(translation)
                
            if all_joints:
                all_joints = np.concatenate(all_joints, axis=0)
                coord_analysis['coordinate_ranges'] = {
                    'x': [float(all_joints[:, 0].min()), float(all_joints[:, 0].max())],
                    'y': [float(all_joints[:, 1].min()), float(all_joints[:, 1].max())],
                    'z': [float(all_joints[:, 2].min()), float(all_joints[:, 2].max())]
                }
                coord_analysis['coordinate_center'] = [
                    float(all_joints[:, 0].mean()),
                    float(all_joints[:, 1].mean()), 
                    float(all_joints[:, 2].mean())
                ]
                coord_analysis['scale_estimate'] = float(np.std(all_joints))
                
        elif model_name == 'wilor':
            # Analyze MANO coordinates
            all_keypoints = []
            all_translations = []
            
            for image_name, data in model_data.items():
                if data['keypoints_3d'] is not None:
                    keypoints = data['keypoints_3d']
                    all_keypoints.append(keypoints)
                    
                if data['camera_translation'] is not None:
                    all_translations.append(data['camera_translation'])
                    
            if all_keypoints:
                all_keypoints = np.concatenate(all_keypoints, axis=0)
                coord_analysis['coordinate_ranges'] = {
                    'x': [float(all_keypoints[:, 0].min()), float(all_keypoints[:, 0].max())],
                    'y': [float(all_keypoints[:, 1].min()), float(all_keypoints[:, 1].max())],
                    'z': [float(all_keypoints[:, 2].min()), float(all_keypoints[:, 2].max())]
                }
                coord_analysis['coordinate_center'] = [
                    float(all_keypoints[:, 0].mean()),
                    float(all_keypoints[:, 1].mean()),
                    float(all_keypoints[:, 2].mean())
                ]
                coord_analysis['scale_estimate'] = float(np.std(all_keypoints))
                
        return coord_analysis
    
    def visualize_coordinate_systems(self, analysis: Dict):
        """Visualize coordinate systems from different models"""
        fig = plt.figure(figsize=(15, 5))
        
        model_count = len([m for m in analysis.keys() if analysis[m]])
        
        for i, (model_name, model_analysis) in enumerate(analysis.items()):
            if not model_analysis or not model_analysis.get('coordinate_ranges'):
                continue
                
            ax = fig.add_subplot(1, model_count, i+1, projection='3d')
            
            # Plot coordinate ranges as a bounding box
            ranges = model_analysis['coordinate_ranges']
            center = model_analysis['coordinate_center']
            
            # Create bounding box
            x_range = ranges['x']
            y_range = ranges['y'] 
            z_range = ranges['z']
            
            # Plot center point
            ax.scatter(center[0], center[1], center[2], 
                      c='red', s=100, label='Center')
            
            # Plot bounding box corners
            corners = np.array([
                [x_range[0], y_range[0], z_range[0]],
                [x_range[1], y_range[0], z_range[0]],
                [x_range[1], y_range[1], z_range[0]],
                [x_range[0], y_range[1], z_range[0]],
                [x_range[0], y_range[0], z_range[1]],
                [x_range[1], y_range[0], z_range[1]],
                [x_range[1], y_range[1], z_range[1]],
                [x_range[0], y_range[1], z_range[1]]
            ])
            
            ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], 
                      alpha=0.6, label='Bounds')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y') 
            ax.set_zlabel('Z')
            ax.set_title(f'{model_name.upper()} Coordinate System')
            ax.legend()
            
        plt.tight_layout()
        plt.savefig(self.results_dir / 'coordinate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_alignment_recommendations(self, analysis: Dict) -> Dict:
        """Generate recommendations for coordinate alignment"""
        recommendations = {
            'primary_reference': 'smplestx',  # Use SMPL-X as reference
            'alignment_steps': [],
            'transformation_matrices': {},
            'scale_factors': {}
        }
        
        if 'smplestx' in analysis and analysis['smplestx']:
            reference_center = analysis['smplestx']['coordinate_center']
            reference_scale = analysis['smplestx']['scale_estimate']
            
            for model_name, model_analysis in analysis.items():
                if model_name == 'smplestx' or not model_analysis:
                    continue
                    
                model_center = model_analysis['coordinate_center']
                model_scale = model_analysis['scale_estimate']
                
                # Calculate translation needed
                translation = np.array(reference_center) - np.array(model_center)
                
                # Calculate scale factor
                scale_factor = reference_scale / model_scale if model_scale > 0 else 1.0
                
                recommendations['transformation_matrices'][model_name] = {
                    'translation': translation.tolist(),
                    'scale': float(scale_factor),
                    'rotation': [0, 0, 0]  # To be determined from keypoint matching
                }
                
                recommendations['alignment_steps'].append({
                    'model': model_name,
                    'step': f'Transform {model_name} to SMPL-X coordinate system',
                    'translation': translation.tolist(),
                    'scale_factor': float(scale_factor)
                })
                
        return recommendations
    
    def save_analysis(self, analysis: Dict, recommendations: Dict):
        """Save analysis results to JSON"""
        output = {
            'coordinate_analysis': analysis,
            'alignment_recommendations': recommendations,
            'analysis_timestamp': str(np.datetime64('now'))
        }
        
        output_file = self.results_dir / 'coordinate_system_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Analysis saved to: {output_file}")
        
    def run_full_analysis(self):
        """Run complete coordinate system analysis"""
        print("Loading model outputs...")
        outputs = self.load_model_outputs()
        
        print("Analyzing coordinate systems...")
        analysis = self.analyze_coordinate_systems(outputs)
        
        print("Generating alignment recommendations...")
        recommendations = self.generate_alignment_recommendations(analysis)
        
        print("Creating visualizations...")
        self.visualize_coordinate_systems(analysis)
        
        print("Saving analysis...")
        self.save_analysis(analysis, recommendations)
        
        # Print summary
        print("\n" + "="*60)
        print("COORDINATE SYSTEM ANALYSIS SUMMARY")
        print("="*60)
        
        for model_name, model_analysis in analysis.items():
            if model_analysis and model_analysis.get('coordinate_ranges'):
                print(f"\n{model_name.upper()}:")
                print(f"  Center: {model_analysis['coordinate_center']}")
                print(f"  Scale: {model_analysis['scale_estimate']:.4f}")
                ranges = model_analysis['coordinate_ranges']
                print(f"  X range: [{ranges['x'][0]:.3f}, {ranges['x'][1]:.3f}]")
                print(f"  Y range: [{ranges['y'][0]:.3f}, {ranges['y'][1]:.3f}]") 
                print(f"  Z range: [{ranges['z'][0]:.3f}, {ranges['z'][1]:.3f}]")
                
        print(f"\nRecommendations saved. Primary reference: {recommendations['primary_reference']}")
        print("="*60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze coordinate systems from 3D model outputs')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing model outputs (e.g., pipeline_results/run_TIMESTAMP)')
    
    args = parser.parse_args()
    
    analyzer = CoordinateAnalyzer(args.results_dir)
    analyzer.run_full_analysis()

if __name__ == '__main__':
    main()