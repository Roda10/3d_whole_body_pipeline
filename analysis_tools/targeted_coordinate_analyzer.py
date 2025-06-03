#!/usr/bin/env python3
"""
Targeted Coordinate Analyzer for Your Specific Pipeline Outputs
Reads the exact file formats you have: SMPL-X params, WiLoR params, EMOCA codes
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PipelineCoordinateAnalyzer:
    """Analyzes coordinates from your specific pipeline outputs"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
    def load_smplestx_parameters(self) -> Dict:
        """Load SMPLest-X parameters from your actual file structure"""
        smplestx_data = {}
        
        # Look for person directories
        for person_dir in self.results_dir.glob('smplestx_results/*/person_*'):
            person_id = person_dir.name
            param_file = person_dir / f'smplx_params_{person_id}.json'
            
            if param_file.exists():
                with open(param_file, 'r') as f:
                    params = json.load(f)
                
                # Extract the coordinate data we need
                smplestx_data[person_id] = {
                    'joints_3d': np.array(params['joints_3d']),
                    'joints_2d': np.array(params['joints_2d']),
                    'translation': np.array(params['translation']),
                    'root_pose': np.array(params['root_pose']),
                    'body_pose': np.array(params['body_pose']),
                    'left_hand_pose': np.array(params['left_hand_pose']),
                    'right_hand_pose': np.array(params['right_hand_pose']),
                    'betas': np.array(params['betas']),
                    'expression': np.array(params['expression'])
                }
                
                print(f"✅ Loaded SMPLest-X data for {person_id}")
                print(f"   Joints 3D shape: {smplestx_data[person_id]['joints_3d'].shape}")
                print(f"   Translation: {smplestx_data[person_id]['translation']}")
                
        return smplestx_data
    
    def load_wilor_parameters(self) -> Dict:
        """Load WiLoR parameters from your actual file structure"""
        wilor_data = {}
        
        # Look for WiLoR parameter files
        for param_file in self.results_dir.glob('wilor_results/*_parameters.json'):
            image_name = param_file.stem.replace('_parameters', '')
            
            with open(param_file, 'r') as f:
                params = json.load(f)
            
            # Extract coordinate data
            hands_data = params.get('hands', [])
            
            wilor_data[image_name] = {
                'metadata': params.get('metadata', {}),
                'hands': hands_data,
                'hand_count': len(hands_data)
            }
            
            # Extract 3D coordinates (updated for new format)
            all_vertices = []
            all_keypoints = []
            all_cam_translations = []
            
            for hand_idx, hand_data in enumerate(hands_data):
                # Check for actual coordinate data (new format)
                if 'vertices_3d' in hand_data and isinstance(hand_data['vertices_3d'], list):
                    vertices = np.array(hand_data['vertices_3d'])
                    all_vertices.append(vertices)
                    
                if 'keypoints_3d' in hand_data and isinstance(hand_data['keypoints_3d'], list):
                    keypoints = np.array(hand_data['keypoints_3d'])
                    all_keypoints.append(keypoints)
                    
                if 'camera_translation' in hand_data and isinstance(hand_data['camera_translation'], list):
                    cam_trans = np.array(hand_data['camera_translation'])
                    all_cam_translations.append(cam_trans)
            
            if all_vertices:
                wilor_data[image_name]['all_vertices'] = np.concatenate(all_vertices, axis=0)
            if all_keypoints:
                wilor_data[image_name]['all_keypoints'] = np.concatenate(all_keypoints, axis=0)
            if all_cam_translations:
                wilor_data[image_name]['all_camera_translations'] = np.array(all_cam_translations)
                
            print(f"✅ Loaded WiLoR data for {image_name}")
            print(f"   Found {len(hands_data)} hands")
            if all_vertices:
                print(f"   Total vertices: {wilor_data[image_name]['all_vertices'].shape}")
            if all_keypoints:
                print(f"   Total keypoints: {wilor_data[image_name]['all_keypoints'].shape}")
            else:
                print(f"   ⚠️  No 3D coordinate data found - may need updated WiLoR extractor")
            
        return wilor_data
    
    def load_emoca_parameters(self) -> Dict:
        """Load EMOCA parameters from your actual file structure"""
        emoca_data = {}
        
        # Look for EMOCA output directories (EMOCA_v2_lr_mse_20/test*/*)
        for emoca_subdir in self.results_dir.glob('emoca_results/EMOCA_*/test*/'):
            image_name = emoca_subdir.name
            
            # Try to load individual code files (your actual format)
            code_files = {
                'shapecode': emoca_subdir / 'shape.json',
                'expcode': emoca_subdir / 'exp.json', 
                'texcode': emoca_subdir / 'tex.json',
                'posecode': emoca_subdir / 'pose.json',
                'detailcode': emoca_subdir / 'detail.json'
            }
            
            codes = {}
            files_found = 0
            
            for code_type, file_path in code_files.items():
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            codes[code_type] = np.array(json.load(f))
                            files_found += 1
                    except Exception as e:
                        print(f"   ⚠️  Error loading {file_path}: {e}")
            
            # Also try the combined codes.json if it exists
            combined_codes_file = emoca_subdir / 'codes.json'
            if combined_codes_file.exists():
                try:
                    with open(combined_codes_file, 'r') as f:
                        combined_codes = json.load(f)
                    
                    codes.update({
                        'shapecode': np.array(combined_codes['shapecode']),
                        'expcode': np.array(combined_codes['expcode']),
                        'texcode': np.array(combined_codes['texcode']),
                        'posecode': np.array(combined_codes['posecode']),
                        'detailcode': np.array(combined_codes['detailcode'])
                    })
                    files_found += 5
                except Exception as e:
                    print(f"   ⚠️  Error loading combined codes: {e}")
            
            if files_found > 0:
                emoca_data[image_name] = codes
                print(f"✅ Loaded EMOCA data for {image_name}")
                print(f"   Found {files_found} code files")
                if 'shapecode' in codes:
                    print(f"   Shape code dims: {codes['shapecode'].shape}")
                if 'expcode' in codes:
                    print(f"   Expression code dims: {codes['expcode'].shape}")
                if 'posecode' in codes:
                    print(f"   Pose code dims: {codes['posecode'].shape}")
            else:
                print(f"❌ No EMOCA code files found in {emoca_subdir}")
                
        return emoca_data
    
    def analyze_coordinate_systems(self) -> Dict:
        """Analyze coordinate systems from your actual data"""
        print("Loading all model outputs...")
        
        smplestx_data = self.load_smplestx_parameters()
        wilor_data = self.load_wilor_parameters()
        emoca_data = self.load_emoca_parameters()
        
        analysis = {
            'smplestx': self._analyze_smplestx_coordinates(smplestx_data),
            'wilor': self._analyze_wilor_coordinates(wilor_data),
            'emoca': self._analyze_emoca_coordinates(emoca_data)
        }
        
        return analysis
    
    def _analyze_smplestx_coordinates(self, data: Dict) -> Dict:
        """Analyze SMPLest-X coordinate system"""
        if not data:
            return {'error': 'No SMPLest-X data found'}
        
        all_joints = []
        all_translations = []
        
        for person_id, person_data in data.items():
            joints_3d = person_data['joints_3d']
            translation = person_data['translation']
            
            all_joints.append(joints_3d)
            all_translations.append(translation)
        
        all_joints = np.concatenate(all_joints, axis=0)
        all_translations = np.array(all_translations)
        
        return {
            'coordinate_ranges': {
                'x': [float(all_joints[:, 0].min()), float(all_joints[:, 0].max())],
                'y': [float(all_joints[:, 1].min()), float(all_joints[:, 1].max())],
                'z': [float(all_joints[:, 2].min()), float(all_joints[:, 2].max())]
            },
            'coordinate_center': [
                float(all_joints[:, 0].mean()),
                float(all_joints[:, 1].mean()),
                float(all_joints[:, 2].mean())
            ],
            'scale_estimate': float(np.std(all_joints)),
            'translation_range': {
                'x': [float(all_translations[:, 0].min()), float(all_translations[:, 0].max())],
                'y': [float(all_translations[:, 1].min()), float(all_translations[:, 1].max())],
                'z': [float(all_translations[:, 2].min()), float(all_translations[:, 2].max())]
            },
            'joint_count': all_joints.shape[0],
            'person_count': len(data)
        }
    
    def _analyze_wilor_coordinates(self, data: Dict) -> Dict:
        """Analyze WiLoR coordinate system"""
        if not data:
            return {'error': 'No WiLoR data found'}
        
        all_coords = []
        
        for image_name, image_data in data.items():
            if 'all_vertices' in image_data:
                all_coords.append(image_data['all_vertices'])
            elif 'all_keypoints' in image_data:
                all_coords.append(image_data['all_keypoints'])
        
        if not all_coords:
            return {'error': 'No 3D coordinate data found in WiLoR outputs'}
        
        all_coords = np.concatenate(all_coords, axis=0)
        
        return {
            'coordinate_ranges': {
                'x': [float(all_coords[:, 0].min()), float(all_coords[:, 0].max())],
                'y': [float(all_coords[:, 1].min()), float(all_coords[:, 1].max())],
                'z': [float(all_coords[:, 2].min()), float(all_coords[:, 2].max())]
            },
            'coordinate_center': [
                float(all_coords[:, 0].mean()),
                float(all_coords[:, 1].mean()),
                float(all_coords[:, 2].mean())
            ],
            'scale_estimate': float(np.std(all_coords)),
            'point_count': all_coords.shape[0],
            'hand_count': sum(d['hand_count'] for d in data.values())
        }
    
    def _analyze_emoca_coordinates(self, data: Dict) -> Dict:
        """Analyze EMOCA parameter space (not 3D coordinates but parameter ranges)"""
        if not data:
            return {'error': 'No EMOCA data found'}
        
        all_shape = []
        all_exp = []
        all_pose = []
        
        for image_name, image_data in data.items():
            shape_code = image_data['shapecode']
            exp_code = image_data['expcode']
            pose_code = image_data['posecode']
            
            # Handle both 1D and 2D arrays
            if len(shape_code.shape) == 1:
                shape_code = shape_code.reshape(1, -1)
            if len(exp_code.shape) == 1:
                exp_code = exp_code.reshape(1, -1)
            if len(pose_code.shape) == 1:
                pose_code = pose_code.reshape(1, -1)
                
            all_shape.append(shape_code)
            all_exp.append(exp_code)
            all_pose.append(pose_code)
        
        all_shape = np.concatenate(all_shape, axis=0)
        all_exp = np.concatenate(all_exp, axis=0)
        all_pose = np.concatenate(all_pose, axis=0)
        
        return {
            'parameter_ranges': {
                'shape': [float(all_shape.min()), float(all_shape.max())],
                'expression': [float(all_exp.min()), float(all_exp.max())],
                'pose': [float(all_pose.min()), float(all_pose.max())]
            },
            'parameter_dims': {
                'shape': int(all_shape.shape[1]),
                'expression': int(all_exp.shape[1]),
                'pose': int(all_pose.shape[1])
            },
            'parameter_stats': {
                'shape_mean': float(all_shape.mean()),
                'expression_mean': float(all_exp.mean()),
                'pose_mean': float(all_pose.mean()),
                'shape_std': float(all_shape.std()),
                'expression_std': float(all_exp.std()),
                'pose_std': float(all_pose.std())
            },
            'sample_count': all_shape.shape[0]
        }
    
    def generate_fusion_strategy(self, analysis: Dict) -> Dict:
        """Generate specific fusion strategy based on your data"""
        strategy = {
            'coordinate_system_choice': 'smplestx',
            'reason': 'SMPL-X provides full-body coordinate system with proper scale',
            'transformations_needed': {},
            'parameter_mappings': {},
            'implementation_steps': []
        }
        
        # Analyze coordinate differences
        if 'smplestx' in analysis and 'wilor' in analysis:
            smplx_center = analysis['smplestx'].get('coordinate_center', [0, 0, 0])
            smplx_scale = analysis['smplestx'].get('scale_estimate', 1.0)
            
            if 'coordinate_center' in analysis['wilor']:
                wilor_center = analysis['wilor']['coordinate_center']
                wilor_scale = analysis['wilor']['scale_estimate']
                
                # Calculate transformation from WiLoR to SMPL-X space
                translation = np.array(smplx_center) - np.array(wilor_center)
                scale_factor = smplx_scale / wilor_scale if wilor_scale > 0 else 1.0
                
                strategy['transformations_needed']['wilor_to_smplx'] = {
                    'translation': translation.tolist(),
                    'scale': float(scale_factor),
                    'notes': 'Transform WiLoR hand coordinates to SMPL-X body coordinate system'
                }
        
        # Parameter mapping strategy
        strategy['parameter_mappings'] = {
            'body_shape': {
                'primary': 'smplestx.betas',
                'source': 'Use SMPL-X body shape as foundation'
            },
            'hand_pose': {
                'primary': 'wilor.hand_poses',
                'fallback': 'smplestx.left_hand_pose, smplestx.right_hand_pose',
                'source': 'WiLoR provides more detailed hand articulation'
            },
            'facial_expression': {
                'primary': 'emoca.expcode',
                'target': 'smplestx.expression',
                'source': 'Map EMOCA rich expression space to SMPL-X'
            },
            'head_pose': {
                'primary': 'smplestx.root_pose',
                'refinement': 'emoca.posecode',
                'source': 'SMPL-X for global, EMOCA for facial pose details'
            }
        }
        
        # Implementation steps
        strategy['implementation_steps'] = [
            '1. Load SMPL-X parameters as foundation',
            '2. Transform WiLoR hand coordinates to SMPL-X space',
            '3. Map EMOCA expression codes to SMPL-X expression parameters',
            '4. Optimize joint positions for consistency',
            '5. Validate with anatomical constraints'
        ]
        
        return strategy
    
    def visualize_coordinate_comparison(self, analysis: Dict):
        """Visualize coordinate systems side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Coordinate ranges comparison
        models = []
        x_ranges = []
        y_ranges = []
        z_ranges = []
        
        for model_name, model_analysis in analysis.items():
            if 'coordinate_ranges' in model_analysis:
                models.append(model_name)
                ranges = model_analysis['coordinate_ranges']
                x_ranges.append(ranges['x'][1] - ranges['x'][0])
                y_ranges.append(ranges['y'][1] - ranges['y'][0])
                z_ranges.append(ranges['z'][1] - ranges['z'][0])
        
        if models:
            x_pos = np.arange(len(models))
            width = 0.25
            
            axes[0].bar(x_pos - width, x_ranges, width, label='X Range', alpha=0.8)
            axes[0].bar(x_pos, y_ranges, width, label='Y Range', alpha=0.8)
            axes[0].bar(x_pos + width, z_ranges, width, label='Z Range', alpha=0.8)
            
            axes[0].set_xlabel('Models')
            axes[0].set_ylabel('Coordinate Range')
            axes[0].set_title('Coordinate System Ranges Comparison')
            axes[0].set_xticks(x_pos)
            axes[0].set_xticklabels(models)
            axes[0].legend()
            axes[0].set_yscale('log')  # Log scale to handle different magnitudes
        
        # Plot 2: Scale comparison
        scales = []
        scale_models = []
        
        for model_name, model_analysis in analysis.items():
            if 'scale_estimate' in model_analysis:
                scale_models.append(model_name)
                scales.append(model_analysis['scale_estimate'])
        
        if scales:
            axes[1].bar(scale_models, scales, alpha=0.8, color=['red', 'green', 'blue'][:len(scales)])
            axes[1].set_xlabel('Models')
            axes[1].set_ylabel('Scale Estimate (std dev)')
            axes[1].set_title('Coordinate System Scale Comparison')
            axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'coordinate_fusion_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_fusion_analysis(self):
        """Run complete fusion-oriented analysis"""
        print("="*80)
        print("FUSION-ORIENTED COORDINATE ANALYSIS")
        print("="*80)
        
        analysis = self.analyze_coordinate_systems()
        
        print("\nCOORDINATE SYSTEM ANALYSIS:")
        print("-" * 50)
        for model_name, model_analysis in analysis.items():
            print(f"\n{model_name.upper()}:")
            if 'error' in model_analysis:
                print(f"  ❌ {model_analysis['error']}")
                continue
                
            if 'coordinate_center' in model_analysis:
                center = model_analysis['coordinate_center']
                scale = model_analysis['scale_estimate']
                print(f"  📍 Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
                print(f"  📏 Scale: {scale:.4f}")
                
                ranges = model_analysis['coordinate_ranges']
                print(f"  📊 X: [{ranges['x'][0]:.3f}, {ranges['x'][1]:.3f}]")
                print(f"     Y: [{ranges['y'][0]:.3f}, {ranges['y'][1]:.3f}]")
                print(f"     Z: [{ranges['z'][0]:.3f}, {ranges['z'][1]:.3f}]")
            
            # Handle EMOCA parameter space (different from 3D coordinates)
            elif 'parameter_dims' in model_analysis:
                dims = model_analysis['parameter_dims']
                ranges = model_analysis['parameter_ranges']
                stats = model_analysis['parameter_stats']
                
                print(f"  📊 Parameter Space Analysis:")
                print(f"     Shape: {dims['shape']}D, range[{ranges['shape'][0]:.3f}, {ranges['shape'][1]:.3f}], mean={stats['shape_mean']:.3f}")
                print(f"     Expression: {dims['expression']}D, range[{ranges['expression'][0]:.3f}, {ranges['expression'][1]:.3f}], mean={stats['expression_mean']:.3f}")
                print(f"     Pose: {dims['pose']}D, range[{ranges['pose'][0]:.3f}, {ranges['pose'][1]:.3f}], mean={stats['pose_mean']:.3f}")
                print(f"  🎯 Fusion Target: Map to SMPL-X 10D expression space")
        
        strategy = self.generate_fusion_strategy(analysis)
        
        print(f"\nFUSION STRATEGY:")
        print("-" * 50)
        print(f"Primary coordinate system: {strategy['coordinate_system_choice']}")
        print(f"Reason: {strategy['reason']}")
        
        print(f"\nRequired transformations:")
        for transform_name, transform_data in strategy['transformations_needed'].items():
            print(f"  {transform_name}:")
            print(f"    Translation: {transform_data['translation']}")
            print(f"    Scale: {transform_data['scale']:.4f}")
        
        print(f"\nImplementation steps:")
        for step in strategy['implementation_steps']:
            print(f"  {step}")
        
        self.visualize_coordinate_comparison(analysis)
        
        # Save results
        results = {
            'analysis': analysis,
            'fusion_strategy': strategy
        }
        
        output_file = self.results_dir / 'fusion_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"\n💾 Fusion analysis saved to: {output_file}")
        print("="*80)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze coordinates for fusion')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing pipeline results')
    
    args = parser.parse_args()
    
    analyzer = PipelineCoordinateAnalyzer(args.results_dir)
    analyzer.run_fusion_analysis()

if __name__ == '__main__':
    main()