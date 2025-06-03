# #!/usr/bin/env python3
# """
# Proper Mesh Rendering Visualizer
# Uses SMPLest-X visualization_utils to render the fused mesh properly
# """

# import numpy as np
# import json
# import cv2
# import os
# import sys
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# import matplotlib.pyplot as plt

# # Add path for visualization utils
# sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'SMPLest-X', 'utils'))

# # Import the render_mesh function you provided
# try:
#     from visualization_utils import render_mesh, save_obj
# except ImportError:
#     print("❌ Could not import visualization_utils. Make sure the path is correct.")
#     print("   Trying to use local visualization functions...")
    
#     # Define local versions if import fails
#     def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
#         """Local fallback render function"""
#         print("⚠️  Using fallback rendering (scatter plot)")
#         # Simple 2D projection fallback
#         if len(vertices) > 0:
#             # Project 3D to 2D
#             focal = cam_param.get('focal', [500, 500])
#             princpt = cam_param.get('princpt', [img.shape[1]/2, img.shape[0]/2])
            
#             vertices_2d = vertices.copy()
#             if vertices_2d.shape[1] >= 3 and np.any(vertices_2d[:, 2] != 0):
#                 vertices_2d[:, 0] = vertices_2d[:, 0] * focal[0] / vertices_2d[:, 2] + princpt[0]
#                 vertices_2d[:, 1] = vertices_2d[:, 1] * focal[1] / vertices_2d[:, 2] + princpt[1]
            
#             # Draw points
#             for vertex in vertices_2d:
#                 if 0 <= vertex[0] < img.shape[1] and 0 <= vertex[1] < img.shape[0]:
#                     cv2.circle(img, (int(vertex[0]), int(vertex[1])), 2, (0, 255, 0), -1)
        
#         return img
    
#     def save_obj(vertices, faces, filename):
#         """Local save_obj function"""
#         with open(filename, 'w') as f:
#             for v in vertices:
#                 f.write(f"v {v[0]} {v[1]} {v[2]}\n")
#             for face in faces:
#                 f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

# class MeshRenderingVisualizer:
#     """Proper mesh rendering using SMPLest-X visualization system"""
    
#     def __init__(self, results_dir: str):
#         self.results_dir = Path(results_dir)
        
#     def load_fused_mesh_data(self) -> Dict:
#         """Load the mesh blending results"""
        
#         # Load mesh blending output
#         mesh_file = self.results_dir / 'mesh_blending_output.json'
#         if not mesh_file.exists():
#             raise FileNotFoundError(f"Run mesh blending first. Missing: {mesh_file}")
            
#         with open(mesh_file, 'r') as f:
#             mesh_data = json.load(f)
            
#         # Convert back to numpy arrays
#         mesh_data['vertices'] = np.array(mesh_data['vertices'])
#         if 'faces' in mesh_data and len(mesh_data['faces']) > 0:
#             mesh_data['faces'] = np.array(mesh_data['faces'])
#         else:
#             mesh_data['faces'] = None
            
#         return mesh_data
    
#     def load_original_image(self) -> np.ndarray:
#         """Load the original input image for rendering overlay"""
        
#         # Find the original image from pipeline results
#         image_candidates = [
#             self.results_dir / 'smplestx_results' / 'inference_output_*' / 'rendered_*.jpg',
#             self.results_dir / 'wilor_results' / '*.jpg',
#             self.results_dir / 'emoca_results' / 'temp_input' / '*.jpg'
#         ]
        
#         for pattern in image_candidates:
#             for img_file in self.results_dir.rglob(pattern.name):
#                 if 'temp_input' in str(img_file) or 'rendered' not in str(img_file):
#                     try:
#                         img = cv2.imread(str(img_file))
#                         if img is not None:
#                             return img
#                     except:
#                         continue
        
#         # Create default image if not found
#         print("⚠️  Original image not found, creating default background")
#         return np.ones((512, 512, 3), dtype=np.uint8) * 128  # Gray background
    
#     def get_camera_parameters(self, mesh_data: Dict, img_shape: Tuple[int, int]) -> Dict:
#         """Generate appropriate camera parameters for rendering"""
        
#         vertices = mesh_data['vertices']
        
#         # Calculate mesh bounds
#         min_coords = np.min(vertices, axis=0)
#         max_coords = np.max(vertices, axis=0)
#         mesh_center = (min_coords + max_coords) / 2
#         mesh_size = np.max(max_coords - min_coords)
        
#         # Set camera parameters based on mesh size and image dimensions
#         img_height, img_width = img_shape[:2]
        
#         # Focal length should be proportional to image size and mesh size
#         focal_length = max(img_width, img_height) * 0.8  # Adjust for good view
        
#         cam_param = {
#             'focal': [focal_length, focal_length],
#             'princpt': [img_width / 2.0, img_height / 2.0]
#         }
        
#         return cam_param
    
#     def create_mesh_faces(self, vertices: np.ndarray, labels: List[str]) -> np.ndarray:
#         """Create basic triangular faces for point cloud if faces don't exist"""
        
#         if len(vertices) < 3:
#             return np.array([])
            
#         faces = []
        
#         # Group vertices by component type
#         components = {}
#         for i, label in enumerate(labels):
#             component_type = label.split('_')[0]  # Get base type (smplx, left, right)
#             if component_type not in components:
#                 components[component_type] = []
#             components[component_type].append(i)
        
#         # Create faces within each component using simple triangulation
#         for component_type, vertex_indices in components.items():
#             if len(vertex_indices) >= 3:
#                 # Simple fan triangulation from first vertex
#                 for i in range(1, len(vertex_indices) - 1):
#                     face = [vertex_indices[0], vertex_indices[i], vertex_indices[i + 1]]
#                     faces.append(face)
        
#         return np.array(faces) if faces else np.array([])
    
#     def render_fused_mesh(self, mesh_data: Dict, img: np.ndarray) -> np.ndarray:
#         """Render the fused mesh using proper mesh rendering"""
        
#         vertices = mesh_data['vertices']
#         faces = mesh_data.get('faces', None)
#         labels = mesh_data.get('labels', [])
        
#         # Get camera parameters
#         cam_param = self.get_camera_parameters(mesh_data, img.shape)
        
#         print(f"🎨 Rendering mesh with {len(vertices)} vertices")
#         print(f"   Camera focal: {cam_param['focal']}")
#         print(f"   Camera center: {cam_param['princpt']}")
        
#         # Create faces if they don't exist
#         if faces is None or len(faces) == 0:
#             print("   🔺 Creating triangular faces for point cloud...")
#             faces = self.create_mesh_faces(vertices, labels)
            
#         if len(faces) == 0:
#             print("   ⚠️  No faces available, using point rendering...")
#             rendered_img = render_mesh(img, vertices, [], cam_param, mesh_as_vertices=True)
#         else:
#             print(f"   🔺 Rendering with {len(faces)} faces...")
#             rendered_img = render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False)
        
#         return rendered_img
    
#     def create_component_visualizations(self, mesh_data: Dict, img: np.ndarray) -> Dict[str, np.ndarray]:
#         """Create separate visualizations for each component"""
        
#         vertices = mesh_data['vertices']
#         labels = mesh_data.get('labels', [])
        
#         # Group vertices by component
#         components = {}
#         for i, label in enumerate(labels):
#             if 'smplx' in label:
#                 comp_type = 'body'
#             elif 'left' in label:
#                 comp_type = 'left_hand'
#             elif 'right' in label:
#                 comp_type = 'right_hand'
#             elif 'transition' in label:
#                 comp_type = 'transitions'
#             else:
#                 comp_type = 'other'
                
#             if comp_type not in components:
#                 components[comp_type] = []
#             components[comp_type].append(i)
        
#         renderings = {}
#         cam_param = self.get_camera_parameters(mesh_data, img.shape)
        
#         for comp_type, vertex_indices in components.items():
#             if len(vertex_indices) > 0:
#                 comp_vertices = vertices[vertex_indices]
#                 comp_img = img.copy()
                
#                 # Render component
#                 rendered = render_mesh(comp_img, comp_vertices, [], cam_param, mesh_as_vertices=True)
#                 renderings[comp_type] = rendered
                
#                 print(f"   🎨 Rendered {comp_type}: {len(comp_vertices)} vertices")
        
#         return renderings
    
#     def create_comparison_visualization(self, mesh_data: Dict, original_img: np.ndarray) -> np.ndarray:
#         """Create a comparison showing original image vs fused mesh rendering"""
        
#         # Resize images to same height for comparison
#         target_height = 512
#         aspect_ratio = original_img.shape[1] / original_img.shape[0]
#         target_width = int(target_height * aspect_ratio)
        
#         # Resize original image
#         img_resized = cv2.resize(original_img, (target_width, target_height))
        
#         # Render fused mesh
#         rendered_img = self.render_fused_mesh(mesh_data, img_resized.copy())
        
#         # Create side-by-side comparison
#         comparison = np.hstack([img_resized, rendered_img])
        
#         # Add labels
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(comparison, 'Original Image', (10, 30), font, 1, (255, 255, 255), 2)
#         cv2.putText(comparison, 'Fused 3D Mesh', (target_width + 10, 30), font, 1, (255, 255, 255), 2)
        
#         return comparison
    
#     def save_rendered_outputs(self, mesh_data: Dict, original_img: np.ndarray):
#         """Save all rendered outputs"""
        
#         output_dir = self.results_dir / 'mesh_renderings'
#         output_dir.mkdir(exist_ok=True)
        
#         # 1. Full fused mesh rendering
#         print("🎨 Rendering complete fused mesh...")
#         fused_rendering = self.render_fused_mesh(mesh_data, original_img.copy())
#         cv2.imwrite(str(output_dir / 'fused_mesh_rendering.jpg'), fused_rendering)
        
#         # 2. Component-wise renderings
#         print("🎨 Rendering individual components...")
#         component_renderings = self.create_component_visualizations(mesh_data, original_img.copy())
        
#         for comp_type, rendering in component_renderings.items():
#             cv2.imwrite(str(output_dir / f'{comp_type}_rendering.jpg'), rendering)
        
#         # 3. Side-by-side comparison
#         print("🎨 Creating comparison visualization...")
#         comparison = self.create_comparison_visualization(mesh_data, original_img)
#         cv2.imwrite(str(output_dir / 'original_vs_fused_comparison.jpg'), comparison)
        
#         # 4. Save mesh as OBJ with proper faces
#         print("💾 Saving mesh as OBJ...")
#         vertices = mesh_data['vertices']
#         faces = mesh_data.get('faces', None)
#         labels = mesh_data.get('labels', [])
        
#         if faces is None or len(faces) == 0:
#             faces = self.create_mesh_faces(vertices, labels)
        
#         if len(faces) > 0:
#             save_obj(vertices, faces, str(output_dir / 'fused_mesh_with_faces.obj'))
#             print(f"   📄 Saved OBJ with {len(faces)} faces")
#         else:
#             # Save as point cloud OBJ
#             with open(output_dir / 'fused_mesh_points.obj', 'w') as f:
#                 for vertex in vertices:
#                     f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
#             print(f"   📄 Saved point cloud OBJ with {len(vertices)} vertices")
        
#         print(f"✅ All renderings saved to: {output_dir}")
        
#         return {
#             'fused_rendering': fused_rendering,
#             'component_renderings': component_renderings,
#             'comparison': comparison,
#             'output_directory': output_dir
#         }
    
#     def run_mesh_rendering_visualization(self):
#         """Run complete mesh rendering visualization"""
        
#         print("="*80)
#         print("PROPER MESH RENDERING VISUALIZATION")
#         print("="*80)
#         print("Using SMPLest-X visualization system for proper 3D mesh rendering")
#         print()
        
#         try:
#             # Load mesh data
#             print("🔄 Loading fused mesh data...")
#             mesh_data = self.load_fused_mesh_data()
#             print(f"   ✅ Loaded mesh with {len(mesh_data['vertices'])} vertices")
            
#             # Load original image
#             print("🔄 Loading original image...")
#             original_img = self.load_original_image()
#             print(f"   ✅ Loaded image: {original_img.shape}")
            
#             # Create and save all renderings
#             print("\n🎨 Creating mesh renderings...")
#             results = self.save_rendered_outputs(mesh_data, original_img)
            
#             print(f"\n🎯 MESH RENDERING COMPLETE!")
#             print("-" * 50)
#             print("Generated visualizations:")
#             print("✓ Complete fused mesh rendering")
#             print("✓ Individual component renderings")
#             print("✓ Original vs fused comparison")
#             print("✓ OBJ mesh files with proper faces")
            
#             print(f"\n📁 All outputs saved to: {results['output_directory']}")
#             print("="*80)
#             print("You can now see the proper 3D mesh rendering!")
#             print("Check the 'original_vs_fused_comparison.jpg' for best overview.")
            
#         except Exception as e:
#             print(f"❌ Mesh rendering failed: {str(e)}")
#             import traceback
#             traceback.print_exc()

# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Proper Mesh Rendering Visualization')
#     parser.add_argument('--results_dir', type=str, required=True,
#                        help='Directory containing mesh blending results')
    
#     args = parser.parse_args()
    
#     visualizer = MeshRenderingVisualizer(args.results_dir)
#     visualizer.run_mesh_rendering_visualization()

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
Fixed Mesh Visualizer Using Real SMPL-X Mesh
Now uses actual SMPL-X mesh vertices instead of just joints
"""

import numpy as np
import json
import cv2
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Add path for visualization utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'external', 'SMPLest-X', 'utils'))

# Import the render_mesh function
try:
    from visualization_utils import render_mesh, save_obj
except ImportError:
    print("❌ Could not import visualization_utils. Using fallback...")
    
    def render_mesh(img, vertices, faces, cam_param, mesh_as_vertices=False):
        """Fallback render function"""
        if len(vertices) > 0:
            focal = cam_param.get('focal', [500, 500])
            princpt = cam_param.get('princpt', [img.shape[1]/2, img.shape[0]/2])
            
            vertices_2d = vertices.copy()
            if vertices_2d.shape[1] >= 3 and np.any(vertices_2d[:, 2] != 0):
                vertices_2d[:, 0] = vertices_2d[:, 0] * focal[0] / vertices_2d[:, 2] + princpt[0]
                vertices_2d[:, 1] = vertices_2d[:, 1] * focal[1] / vertices_2d[:, 2] + princpt[1]
            
            for vertex in vertices_2d:
                if 0 <= vertex[0] < img.shape[1] and 0 <= vertex[1] < img.shape[0]:
                    cv2.circle(img, (int(vertex[0]), int(vertex[1])), 2, (0, 255, 0), -1)
        return img
    
    def save_obj(vertices, faces, filename):
        with open(filename, 'w') as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

class FixedMeshVisualizer:
    """Fixed mesh visualizer using real SMPL-X mesh vertices"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
    def load_smplx_mesh_data(self) -> Dict:
        """Load actual SMPL-X mesh vertices from parameters"""
        
        # Find SMPL-X parameter file
        param_file = None
        for person_dir in self.results_dir.glob('smplestx_results/*/person_*'):
            person_id = person_dir.name
            candidate = person_dir / f'smplx_params_{person_id}.json'
            if candidate.exists():
                param_file = candidate
                break
        
        if not param_file:
            raise FileNotFoundError("No SMPL-X parameter file found")
            
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        # Check if mesh vertices are available
        if 'mesh' not in params:
            raise ValueError("SMPL-X mesh vertices not found! Please update smplestx_adapter.py to save 'mesh' parameter.")
        
        return {
            'mesh_vertices': np.array(params['mesh']),  # Full body mesh vertices
            'joints_3d': np.array(params['joints_3d']),
            'translation': np.array(params['translation']),
            'body_pose': np.array(params['body_pose']),
            'betas': np.array(params['betas']),
            'expression': np.array(params['expression'])
        }
    
    def load_wilor_data(self) -> Dict:
        """Load WiLoR hand data"""
        param_file = None
        for candidate in self.results_dir.glob('wilor_results/*_parameters.json'):
            param_file = candidate
            break
            
        if not param_file:
            raise FileNotFoundError("No WiLoR parameter file found")
            
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        hands = []
        for hand_data in params['hands']:
            hands.append({
                'hand_type': hand_data['hand_type'],
                'vertices_3d': np.array(hand_data['vertices_3d']),
                'keypoints_3d': np.array(hand_data['keypoints_3d'])
            })
            
        return {'hands': hands}
    
    def load_fusion_analysis(self) -> Dict:
        """Load fusion transformation parameters"""
        analysis_file = self.results_dir / 'fusion_analysis.json'
        if not analysis_file.exists():
            raise FileNotFoundError("Run coordinate analysis first")
            
        with open(analysis_file, 'r') as f:
            return json.load(f)
    
    def transform_wilor_to_smplx(self, wilor_coords: np.ndarray, transform_params: Dict) -> np.ndarray:
        """Transform WiLoR coordinates to SMPL-X space"""
        translation = np.array(transform_params['translation'])
        scale = transform_params['scale']
        
        return wilor_coords * scale + translation
    
    def load_original_image(self) -> np.ndarray:
        """Load original input image"""
        # Try multiple locations
        for pattern in ['**/rendered_*.jpg', '**/test*.jpg', '**/*.jpg']:
            for img_file in self.results_dir.rglob(pattern):
                if 'temp_input' in str(img_file) or any(term in img_file.name.lower() for term in ['test', 'input']):
                    try:
                        img = cv2.imread(str(img_file))
                        if img is not None and img.size > 0:
                            return img
                    except:
                        continue
        
        # Default image
        return np.ones((512, 512, 3), dtype=np.uint8) * 128
    
    def get_smplx_faces(self) -> np.ndarray:
        """Get SMPL-X face topology (you might need to load this from SMPL-X model files)"""
        # This is a simplified face generation - in practice you'd load actual SMPL-X faces
        # For now, we'll skip faces and use point cloud rendering
        return np.array([])
    
    def create_unified_mesh_visualization(self) -> Dict:
        """Create unified mesh with actual SMPL-X body + transformed WiLoR hands"""
        
        print("🔄 Loading all model data...")
        
        # Load data
        smplx_data = self.load_smplx_mesh_data()
        wilor_data = self.load_wilor_data()
        fusion_analysis = self.load_fusion_analysis()
        
        print(f"✅ SMPL-X: {len(smplx_data['mesh_vertices'])} mesh vertices")
        print(f"✅ WiLoR: {len(wilor_data['hands'])} hands")
        
        # Get transformation parameters
        transform_params = fusion_analysis['fusion_strategy']['transformations_needed']['wilor_to_smplx']
        
        # Transform WiLoR hands to SMPL-X space
        transformed_hands = []
        all_hand_vertices = []
        
        for hand in wilor_data['hands']:
            transformed_vertices = self.transform_wilor_to_smplx(hand['vertices_3d'], transform_params)
            transformed_hands.append({
                'hand_type': hand['hand_type'],
                'vertices': transformed_vertices
            })
            all_hand_vertices.extend(transformed_vertices)
        
        all_hand_vertices = np.array(all_hand_vertices) if all_hand_vertices else np.array([]).reshape(0, 3)
        
        print(f"🔄 Transformed {len(all_hand_vertices)} hand vertices to SMPL-X space")
        
        # Create unified representation
        unified_data = {
            'smplx_body_mesh': smplx_data['mesh_vertices'],
            'transformed_hands': transformed_hands,
            'all_hand_vertices': all_hand_vertices,
            'smplx_joints': smplx_data['joints_3d'],
            'transformation_applied': transform_params
        }
        
        return unified_data
    
    def render_individual_components(self, unified_data: Dict, original_img: np.ndarray) -> Dict:
        """Render each component separately"""
        
        renderings = {}
        img_height, img_width = original_img.shape[:2]
        
        # Camera parameters
        focal_length = max(img_width, img_height) * 0.8
        cam_param = {
            'focal': [focal_length, focal_length],
            'princpt': [img_width / 2.0, img_height / 2.0]
        }
        
        print(f"🎨 Using camera: focal={focal_length:.1f}, center=({img_width/2:.1f}, {img_height/2:.1f})")
        
        # 1. SMPL-X body only
        print("🎨 Rendering SMPL-X body mesh...")
        body_img = original_img.copy()
        smplx_faces = self.get_smplx_faces()
        body_rendered = render_mesh(body_img, unified_data['smplx_body_mesh'], smplx_faces, cam_param, mesh_as_vertices=True)
        renderings['smplx_body'] = body_rendered
        
        # 2. WiLoR hands only (original coordinates)
        print("🎨 Rendering WiLoR hands (original coordinates)...")
        hands_orig_img = original_img.copy()
        if len(unified_data['all_hand_vertices']) > 0:
            # Use original WiLoR coordinates (not transformed)
            original_hand_vertices = []
            for hand in unified_data['transformed_hands']:
                # Reverse transform to get original coordinates
                transformed = hand['vertices']
                scale = unified_data['transformation_applied']['scale']
                translation = np.array(unified_data['transformation_applied']['translation'])
                original = (transformed - translation) / scale
                original_hand_vertices.extend(original)
            
            original_hand_vertices = np.array(original_hand_vertices)
            hands_orig_rendered = render_mesh(hands_orig_img, original_hand_vertices, [], cam_param, mesh_as_vertices=True)
            renderings['wilor_hands_original'] = hands_orig_rendered
        
        # 3. WiLoR hands transformed to SMPL-X space
        print("🎨 Rendering transformed WiLoR hands...")
        hands_transformed_img = original_img.copy()
        if len(unified_data['all_hand_vertices']) > 0:
            hands_transformed_rendered = render_mesh(hands_transformed_img, unified_data['all_hand_vertices'], [], cam_param, mesh_as_vertices=True)
            renderings['wilor_hands_transformed'] = hands_transformed_rendered
        
        # 4. Combined: SMPL-X body + transformed hands
        print("🎨 Rendering unified mesh (body + transformed hands)...")
        unified_img = original_img.copy()
        
        # Render body first
        unified_rendered = render_mesh(unified_img, unified_data['smplx_body_mesh'], smplx_faces, cam_param, mesh_as_vertices=True)
        
        # Then render hands on top
        if len(unified_data['all_hand_vertices']) > 0:
            unified_rendered = render_mesh(unified_rendered, unified_data['all_hand_vertices'], [], cam_param, mesh_as_vertices=True)
        
        renderings['unified_mesh'] = unified_rendered
        
        # 5. Individual hands
        for hand in unified_data['transformed_hands']:
            hand_img = original_img.copy()
            hand_rendered = render_mesh(hand_img, hand['vertices'], [], cam_param, mesh_as_vertices=True)
            renderings[f"{hand['hand_type']}_hand"] = hand_rendered
        
        return renderings
    
    def create_comparison_grid(self, renderings: Dict, original_img: np.ndarray) -> np.ndarray:
        """Create a grid comparison of all renderings"""
        
        # Resize all images to same size
        target_size = (400, 400)
        resized_renderings = {}
        
        for name, img in renderings.items():
            resized_renderings[name] = cv2.resize(img, target_size)
        
        original_resized = cv2.resize(original_img, target_size)
        
        # Create 3x3 grid
        grid_images = [
            [original_resized, resized_renderings.get('smplx_body', original_resized), resized_renderings.get('wilor_hands_original', original_resized)],
            [resized_renderings.get('wilor_hands_transformed', original_resized), resized_renderings.get('unified_mesh', original_resized), resized_renderings.get('left_hand', original_resized)],
            [resized_renderings.get('right_hand', original_resized), np.ones(target_size + (3,), dtype=np.uint8) * 255, np.ones(target_size + (3,), dtype=np.uint8) * 255]
        ]
        
        # Add labels
        labels = [
            ['Original Image', 'SMPL-X Body Mesh', 'WiLoR Hands (Original)'],
            ['WiLoR Hands (Transformed)', 'Unified Mesh', 'Left Hand Detail'],
            ['Right Hand Detail', '', '']
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (0, 0, 0)
        thickness = 2
        
        for i, row in enumerate(grid_images):
            for j, img in enumerate(row):
                if labels[i][j]:
                    cv2.putText(img, labels[i][j], (10, 30), font, font_scale, color, thickness)
        
        # Combine into grid
        row1 = np.hstack(grid_images[0])
        row2 = np.hstack(grid_images[1]) 
        row3 = np.hstack(grid_images[2])
        
        grid = np.vstack([row1, row2, row3])
        return grid
    
    def save_all_visualizations(self, unified_data: Dict, renderings: Dict, grid: np.ndarray):
        """Save all visualization outputs"""
        
        output_dir = self.results_dir / 'fixed_mesh_renderings'
        output_dir.mkdir(exist_ok=True)
        
        # Save individual renderings
        for name, img in renderings.items():
            cv2.imwrite(str(output_dir / f'{name}_rendering.jpg'), img)
        
        # Save comparison grid
        cv2.imwrite(str(output_dir / 'complete_comparison_grid.jpg'), grid)
        
        # Save unified mesh as OBJ
        combined_vertices = []
        vertex_labels = []
        
        # Add SMPL-X body vertices
        combined_vertices.extend(unified_data['smplx_body_mesh'])
        vertex_labels.extend(['smplx_body'] * len(unified_data['smplx_body_mesh']))
        
        # Add transformed hand vertices
        if len(unified_data['all_hand_vertices']) > 0:
            combined_vertices.extend(unified_data['all_hand_vertices'])
            vertex_labels.extend(['hands'] * len(unified_data['all_hand_vertices']))
        
        combined_vertices = np.array(combined_vertices)
        
        # Save as point cloud OBJ
        with open(output_dir / 'unified_mesh_fixed.obj', 'w') as f:
            for vertex in combined_vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Save statistics
        stats = {
            'smplx_body_vertices': len(unified_data['smplx_body_mesh']),
            'hand_vertices': len(unified_data['all_hand_vertices']),
            'total_vertices': len(combined_vertices),
            'transformation_applied': unified_data['transformation_applied'],
            'has_actual_smplx_mesh': True
        }
        
        with open(output_dir / 'fixed_mesh_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ All fixed visualizations saved to: {output_dir}")
        return output_dir
    
    def run_fixed_mesh_visualization(self):
        """Run complete fixed mesh visualization"""
        
        print("="*80)
        print("FIXED MESH VISUALIZATION - Using Real SMPL-X Mesh")
        print("="*80)
        
        try:
            # Load original image
            print("🔄 Loading original image...")
            original_img = self.load_original_image()
            print(f"   ✅ Image shape: {original_img.shape}")
            
            # Create unified mesh data
            print("🔄 Creating unified mesh with real SMPL-X vertices...")
            unified_data = self.create_unified_mesh_visualization()
            
            # Render all components
            print("🎨 Rendering all components...")
            renderings = self.render_individual_components(unified_data, original_img)
            
            # Create comparison grid
            print("🎨 Creating comparison grid...")
            grid = self.create_comparison_grid(renderings, original_img)
            
            # Save everything
            print("💾 Saving all visualizations...")
            output_dir = self.save_all_visualizations(unified_data, renderings, grid)
            
            print(f"\n🎯 FIXED MESH VISUALIZATION COMPLETE!")
            print("-" * 50)
            print(f"SMPL-X body vertices: {len(unified_data['smplx_body_mesh']):,}")
            print(f"Hand vertices: {len(unified_data['all_hand_vertices']):,}")
            print(f"Total unified vertices: {len(unified_data['smplx_body_mesh']) + len(unified_data['all_hand_vertices']):,}")
            
            print(f"\n📁 Check these key files:")
            print(f"   • complete_comparison_grid.jpg - Full comparison")
            print(f"   • unified_mesh_rendering.jpg - Final result")
            print(f"   • smplx_body_rendering.jpg - Body mesh only")
            print("="*80)
            
        except Exception as e:
            print(f"❌ Fixed mesh visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Fixed Mesh Visualization')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing results')
    
    args = parser.parse_args()
    
    visualizer = FixedMeshVisualizer(args.results_dir)
    visualizer.run_fixed_mesh_visualization()

if __name__ == '__main__':
    main()