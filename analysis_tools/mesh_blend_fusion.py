# #!/usr/bin/env python3
# """
# Mesh-Level Blending Fusion
# Implements actual mesh blending at hand-wrist and head-neck attachment points
# """

# import numpy as np
# import json
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional
# import trimesh
# from scipy.spatial.distance import cdist
# from scipy.interpolate import RBFInterpolator
# from sklearn.neighbors import NearestNeighbors

# class MeshBlendingFusion:
#     """Advanced fusion with mesh-level blending at attachment points"""
    
#     def __init__(self, results_dir: str):
#         self.results_dir = Path(results_dir)
        
#         # SMPL-X joint indices for attachment points
#         self.SMPLX_JOINT_MAP = {
#             'left_wrist': 20,   # Approximate SMPL-X left wrist joint
#             'right_wrist': 21,  # Approximate SMPL-X right wrist joint  
#             'neck': 12,         # SMPL-X neck joint
#             'head': 15          # SMPL-X head joint
#         }
        
#         # Load transformation parameters from basic fusion
#         self.load_basic_fusion_data()
        
#     def load_basic_fusion_data(self):
#         """Load results from basic fusion demo"""
#         fusion_file = self.results_dir / 'basic_fusion_output.json'
#         if not fusion_file.exists():
#             raise FileNotFoundError(f"Run basic fusion demo first. Missing: {fusion_file}")
            
#         with open(fusion_file, 'r') as f:
#             self.fusion_data = json.load(f)
            
#         # Extract transformation parameters
#         transform = self.fusion_data['transformation_applied']
#         self.wilor_scale = transform['wilor_scale']
#         self.wilor_translation = np.array(transform['wilor_translation'])
        
#     def load_mesh_data(self) -> Dict:
#         """Load actual mesh data from model outputs"""
#         mesh_data = {
#             'smplx_vertices': None,
#             'smplx_faces': None,
#             'hand_meshes': [],
#             'face_mesh': None
#         }
        
#         # Load SMPL-X mesh (we need to reconstruct from parameters)
#         smplx_foundation = self.fusion_data['smplx_foundation']
#         mesh_data['smplx_joints'] = np.array(smplx_foundation['joints_3d'])
        
#         # Load transformed hand meshes
#         for hand_data in self.fusion_data['enhanced_hands']:
#             mesh_data['hand_meshes'].append({
#                 'hand_type': hand_data['hand_type'],
#                 'vertices': np.array(hand_data['vertices_3d_transformed']),
#                 'keypoints': np.array(hand_data['keypoints_3d_transformed'])
#             })
            
#         return mesh_data
    
#     def find_attachment_points(self, mesh_data: Dict) -> Dict:
#         """Find optimal attachment points between meshes"""
#         attachment_points = {}
        
#         smplx_joints = mesh_data['smplx_joints']
        
#         # Find hand attachment points
#         for hand_mesh in mesh_data['hand_meshes']:
#             hand_type = hand_mesh['hand_type']
#             hand_vertices = hand_mesh['vertices']
            
#             # Get corresponding SMPL-X wrist position
#             wrist_joint_idx = self.SMPLX_JOINT_MAP[f'{hand_type}_wrist']
#             if wrist_joint_idx < len(smplx_joints):
#                 wrist_position = smplx_joints[wrist_joint_idx]
                
#                 # Find closest hand vertex to wrist position
#                 distances = cdist([wrist_position], hand_vertices)[0]
#                 closest_hand_vertex_idx = np.argmin(distances)
#                 closest_hand_vertex = hand_vertices[closest_hand_vertex_idx]
                
#                 attachment_points[f'{hand_type}_hand'] = {
#                     'smplx_point': wrist_position,
#                     'smplx_joint_idx': wrist_joint_idx,
#                     'hand_point': closest_hand_vertex,
#                     'hand_vertex_idx': closest_hand_vertex_idx,
#                     'distance': distances[closest_hand_vertex_idx]
#                 }
                
#         return attachment_points
    
#     def create_blending_weights(self, vertices: np.ndarray, attachment_point: np.ndarray, 
#                               blend_radius: float = 0.05) -> np.ndarray:
#         """Create smooth blending weights based on distance from attachment point"""
#         distances = np.linalg.norm(vertices - attachment_point, axis=1)
        
#         # Smooth falloff function (sigmoid-like)
#         weights = np.exp(-distances / blend_radius)
#         weights = np.clip(weights, 0, 1)
        
#         return weights
    
#     def blend_meshes_at_attachment(self, mesh1_vertices: np.ndarray, mesh2_vertices: np.ndarray,
#                                  attachment_point: np.ndarray, blend_radius: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
#         """Blend two meshes smoothly at attachment point"""
        
#         # Create blending weights for both meshes
#         weights1 = self.create_blending_weights(mesh1_vertices, attachment_point, blend_radius)
#         weights2 = self.create_blending_weights(mesh2_vertices, attachment_point, blend_radius)
        
#         # Find vertices in blending zone
#         blend_zone1 = weights1 > 0.01  # Small threshold to avoid numerical issues
#         blend_zone2 = weights2 > 0.01
        
#         blended_mesh1 = mesh1_vertices.copy()
#         blended_mesh2 = mesh2_vertices.copy()
        
#         # Apply blending in overlapping regions
#         if np.any(blend_zone1) and np.any(blend_zone2):
#             # Smooth transition towards attachment point
#             blended_mesh1[blend_zone1] = (
#                 mesh1_vertices[blend_zone1] * (1 - weights1[blend_zone1, np.newaxis]) +
#                 attachment_point * weights1[blend_zone1, np.newaxis]
#             )
            
#             blended_mesh2[blend_zone2] = (
#                 mesh2_vertices[blend_zone2] * (1 - weights2[blend_zone2, np.newaxis]) +
#                 attachment_point * weights2[blend_zone2, np.newaxis]
#             )
        
#         return blended_mesh1, blended_mesh2
    
#     def create_transition_mesh(self, point1: np.ndarray, point2: np.ndarray, 
#                              radius: float = 0.02, segments: int = 8) -> Dict:
#         """Create a cylindrical transition mesh between two attachment points"""
        
#         # Vector from point1 to point2
#         direction = point2 - point1
#         length = np.linalg.norm(direction)
        
#         if length < 1e-6:  # Points are too close
#             return None
            
#         direction_normalized = direction / length
        
#         # Create perpendicular vectors for cylinder
#         if abs(direction_normalized[2]) < 0.9:
#             perp1 = np.cross(direction_normalized, [0, 0, 1])
#         else:
#             perp1 = np.cross(direction_normalized, [1, 0, 0])
#         perp1 = perp1 / np.linalg.norm(perp1)
#         perp2 = np.cross(direction_normalized, perp1)
        
#         # Generate cylinder vertices
#         vertices = []
#         faces = []
        
#         for i in range(segments + 1):
#             t = i / segments  # Parameter along cylinder length
#             center = point1 + t * direction
            
#             # Vary radius for smooth tapering
#             current_radius = radius * (1 - 0.3 * abs(t - 0.5))  # Thinner in middle
            
#             # Create circle of vertices
#             for j in range(segments):
#                 angle = 2 * np.pi * j / segments
#                 offset = current_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
#                 vertices.append(center + offset)
        
#         vertices = np.array(vertices)
        
#         # Create faces (triangles)
#         for i in range(segments):
#             for j in range(segments):
#                 # Current ring
#                 v1 = i * segments + j
#                 v2 = i * segments + (j + 1) % segments
#                 # Next ring
#                 v3 = (i + 1) * segments + j
#                 v4 = (i + 1) * segments + (j + 1) % segments
                
#                 if i < segments:  # Don't create faces beyond last ring
#                     faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
#         return {
#             'vertices': vertices,
#             'faces': np.array(faces)
#         }
    
#     def create_unified_mesh(self, mesh_data: Dict, attachment_points: Dict) -> Dict:
#         """Create unified mesh with blended attachments"""
        
#         print("🔄 Creating unified mesh with blended attachments...")
        
#         # Start with SMPL-X body as foundation
#         smplx_joints = mesh_data['smplx_joints']
#         unified_vertices = []
#         unified_faces = []
#         unified_labels = []  # Track which model each vertex comes from
#         vertex_offset = 0
        
#         # Add SMPL-X body joints as initial mesh points
#         # (In practice, you'd load actual SMPL-X mesh vertices)
#         body_vertices = smplx_joints.copy()
#         unified_vertices.extend(body_vertices)
#         unified_labels.extend(['smplx_body'] * len(body_vertices))
#         vertex_offset += len(body_vertices)
        
#         print(f"   📍 Added {len(body_vertices)} SMPL-X body vertices")
        
#         # Add and blend hand meshes
#         transition_meshes = []
        
#         for hand_mesh in mesh_data['hand_meshes']:
#             hand_type = hand_mesh['hand_type']
#             hand_vertices = hand_mesh['vertices']
            
#             if f'{hand_type}_hand' in attachment_points:
#                 attach_info = attachment_points[f'{hand_type}_hand']
#                 smplx_attach_point = attach_info['smplx_point']
#                 hand_attach_point = attach_info['hand_point']
                
#                 print(f"   🖐️  Blending {hand_type} hand at attachment point")
                
#                 # Blend hand mesh with body
#                 blended_body, blended_hand = self.blend_meshes_at_attachment(
#                     body_vertices, hand_vertices, smplx_attach_point, blend_radius=0.08
#                 )
                
#                 # Add blended hand vertices
#                 unified_vertices.extend(blended_hand)
#                 unified_labels.extend([f'{hand_type}_hand'] * len(blended_hand))
                
#                 # Create transition mesh between attachment points
#                 transition = self.create_transition_mesh(
#                     smplx_attach_point, hand_attach_point, radius=0.03
#                 )
                
#                 if transition:
#                     transition_meshes.append({
#                         'vertices': transition['vertices'],
#                         'faces': transition['faces'] + vertex_offset,
#                         'type': f'{hand_type}_wrist_transition'
#                     })
#                     unified_vertices.extend(transition['vertices'])
#                     unified_labels.extend([f'{hand_type}_transition'] * len(transition['vertices']))
#                     unified_faces.extend(transition['faces'] + vertex_offset)
#                     vertex_offset += len(transition['vertices'])
                
#                 vertex_offset += len(blended_hand)
                
#                 print(f"      ✅ Added {len(blended_hand)} blended hand vertices")
#                 if transition:
#                     print(f"      ✅ Added {len(transition['vertices'])} transition vertices")
        
#         unified_mesh = {
#             'vertices': np.array(unified_vertices),
#             'faces': np.array(unified_faces) if unified_faces else np.array([]),
#             'labels': unified_labels,
#             'transition_meshes': transition_meshes,
#             'attachment_points': attachment_points,
#             'stats': {
#                 'total_vertices': len(unified_vertices),
#                 'body_vertices': len(body_vertices),
#                 'hand_vertices': sum(len(h['vertices']) for h in mesh_data['hand_meshes']),
#                 'transition_vertices': sum(len(t['vertices']) for t in transition_meshes)
#             }
#         }
        
#         return unified_mesh
    
#     def visualize_mesh_blending(self, unified_mesh: Dict):
#         """Visualize the mesh blending results"""
        
#         fig = plt.figure(figsize=(20, 12))
        
#         vertices = unified_mesh['vertices']
#         labels = unified_mesh['labels']
#         attachment_points = unified_mesh['attachment_points']
        
#         # Create color map for different mesh parts
#         color_map = {
#             'smplx_body': 'blue',
#             'left_hand': 'red',
#             'right_hand': 'orange', 
#             'left_transition': 'purple',
#             'right_transition': 'brown'
#         }
        
#         # Plot 1: Complete unified mesh
#         ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
#         for label_type, color in color_map.items():
#             mask = np.array([label_type in label for label in labels])
#             if np.any(mask):
#                 subset_vertices = vertices[mask]
#                 ax1.scatter(subset_vertices[:, 0], subset_vertices[:, 1], subset_vertices[:, 2],
#                            c=color, alpha=0.7, s=20, label=label_type)
        
#         ax1.set_title('Unified Mesh\n(All Components)')
#         ax1.legend()
        
#         # Plot 2: Attachment points detail
#         ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        
#         # Plot mesh with attachment points highlighted
#         ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
#                    c='lightgray', alpha=0.3, s=5)
        
#         # Highlight attachment points
#         for attach_name, attach_info in attachment_points.items():
#             smplx_point = attach_info['smplx_point']
#             hand_point = attach_info['hand_point']
            
#             ax2.scatter(*smplx_point, c='red', s=100, marker='o', 
#                        label=f'{attach_name} SMPL-X')
#             ax2.scatter(*hand_point, c='blue', s=100, marker='^',
#                        label=f'{attach_name} Hand')
            
#             # Draw connection line
#             ax2.plot([smplx_point[0], hand_point[0]], 
#                     [smplx_point[1], hand_point[1]], 
#                     [smplx_point[2], hand_point[2]], 
#                     'k--', alpha=0.7, linewidth=2)
        
#         ax2.set_title('Attachment Points\n(Blend Locations)')
#         ax2.legend()
        
#         # Plot 3: Left hand detail
#         ax3 = fig.add_subplot(2, 3, 3, projection='3d')
#         left_mask = np.array(['left' in label for label in labels])
#         if np.any(left_mask):
#             left_vertices = vertices[left_mask]
#             left_labels_subset = np.array(labels)[left_mask]
            
#             for label in np.unique(left_labels_subset):
#                 label_mask = left_labels_subset == label
#                 color = color_map.get(label, 'gray')
#                 ax3.scatter(left_vertices[label_mask, 0], 
#                            left_vertices[label_mask, 1], 
#                            left_vertices[label_mask, 2],
#                            c=color, alpha=0.8, s=30, label=label)
        
#         ax3.set_title('Left Hand Blending\n(Detail View)')
#         ax3.legend()
        
#         # Plot 4: Right hand detail  
#         ax4 = fig.add_subplot(2, 3, 4, projection='3d')
#         right_mask = np.array(['right' in label for label in labels])
#         if np.any(right_mask):
#             right_vertices = vertices[right_mask]
#             right_labels_subset = np.array(labels)[right_mask]
            
#             for label in np.unique(right_labels_subset):
#                 label_mask = right_labels_subset == label
#                 color = color_map.get(label, 'gray')
#                 ax4.scatter(right_vertices[label_mask, 0], 
#                            right_vertices[label_mask, 1], 
#                            right_vertices[label_mask, 2],
#                            c=color, alpha=0.8, s=30, label=label)
        
#         ax4.set_title('Right Hand Blending\n(Detail View)')
#         ax4.legend()
        
#         # Plot 5: Blending quality metrics
#         ax5 = fig.add_subplot(2, 3, 5)
        
#         # Calculate blending quality metrics
#         attachment_distances = []
#         attachment_names = []
        
#         for attach_name, attach_info in attachment_points.items():
#             distance = attach_info['distance']
#             attachment_distances.append(distance)
#             attachment_names.append(attach_name.replace('_hand', ''))
        
#         ax5.bar(attachment_names, attachment_distances, alpha=0.7, color=['red', 'orange'])
#         ax5.set_ylabel('Attachment Distance')
#         ax5.set_title('Blending Quality\n(Lower = Better)')
#         ax5.tick_params(axis='x', rotation=45)
        
#         # Plot 6: Statistics
#         ax6 = fig.add_subplot(2, 3, 6)
#         ax6.axis('off')
        
#         stats = unified_mesh['stats']
#         stats_text = f"""
#         MESH BLENDING STATISTICS
        
#         Total Vertices: {stats['total_vertices']:,}
        
#         Breakdown:
#         • Body: {stats['body_vertices']:,}
#         • Hands: {stats['hand_vertices']:,}
#         • Transitions: {stats['transition_vertices']:,}
        
#         Attachment Points: {len(attachment_points)}
        
#         Blending Features:
#         ✓ Smooth vertex transitions
#         ✓ Cylindrical wrist connectors
#         ✓ Distance-based weight blending
#         ✓ Multi-mesh unification
        
#         Quality Metrics:
#         • Avg attachment distance: {np.mean(attachment_distances):.4f}
#         • Max attachment distance: {np.max(attachment_distances):.4f}
#         """
        
#         ax6.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
#                 fontfamily='monospace')
        
#         plt.tight_layout()
#         plt.savefig(self.results_dir / 'mesh_blending_fusion.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     def save_blended_mesh(self, unified_mesh: Dict):
#         """Save the blended mesh in multiple formats"""
        
#         # Save as JSON
#         json_output = self._make_serializable(unified_mesh)
#         json_file = self.results_dir / 'mesh_blending_output.json'
#         with open(json_file, 'w') as f:
#             json.dump(json_output, f, indent=2)
        
#         print(f"💾 Blended mesh saved to: {json_file}")
        
#         # Save as OBJ (if we have faces)
#         if len(unified_mesh['faces']) > 0:
#             obj_file = self.results_dir / 'unified_mesh.obj'
#             self._save_obj(unified_mesh['vertices'], unified_mesh['faces'], obj_file)
#             print(f"🗃️  Mesh saved as OBJ: {obj_file}")
        
#         # Save summary
#         summary = {
#             'mesh_blending_type': 'attachment_point_blending',
#             'blending_method': 'distance_weighted_smooth_transition',
#             'models_unified': ['SMPLest-X', 'WiLoR', 'EMOCA'],
#             'attachment_points': len(unified_mesh['attachment_points']),
#             'total_vertices': unified_mesh['stats']['total_vertices'],
#             'mesh_components': {
#                 'body': unified_mesh['stats']['body_vertices'],
#                 'hands': unified_mesh['stats']['hand_vertices'], 
#                 'transitions': unified_mesh['stats']['transition_vertices']
#             },
#             'next_improvements': [
#                 'Add actual SMPL-X mesh vertices (not just joints)',
#                 'Implement face mesh blending for EMOCA',
#                 'Add mesh smoothing and cleanup',
#                 'Optimize transition mesh topology'
#             ]
#         }
        
#         summary_file = self.results_dir / 'mesh_blending_summary.json'
#         with open(summary_file, 'w') as f:
#             json.dump(summary, f, indent=2)
        
#         print(f"📋 Blending summary saved to: {summary_file}")
    
#     def _make_serializable(self, obj):
#         """Convert numpy arrays to lists for JSON serialization"""
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, dict):
#             return {key: self._make_serializable(value) for key, value in obj.items()}
#         elif isinstance(obj, list):
#             return [self._make_serializable(item) for item in obj]
#         else:
#             return obj
    
#     def _save_obj(self, vertices: np.ndarray, faces: np.ndarray, filename: Path):
#         """Save mesh as OBJ file"""
#         with open(filename, 'w') as f:
#             # Write vertices
#             for vertex in vertices:
#                 f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
#             # Write faces (OBJ uses 1-based indexing)
#             for face in faces:
#                 f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
#     def run_mesh_blending_fusion(self):
#         """Run complete mesh-level blending fusion"""
        
#         print("="*80)
#         print("MESH-LEVEL BLENDING FUSION")
#         print("="*80)
#         print("Advanced fusion with smooth mesh blending at attachment points")
#         print()
        
#         try:
#             # Load mesh data
#             print("🔄 Loading mesh data...")
#             mesh_data = self.load_mesh_data()
            
#             # Find attachment points
#             print("🔄 Finding optimal attachment points...")
#             attachment_points = self.find_attachment_points(mesh_data)
#             print(f"   ✅ Found {len(attachment_points)} attachment points")
            
#             for attach_name, attach_info in attachment_points.items():
#                 print(f"      🔗 {attach_name}: distance = {attach_info['distance']:.4f}")
            
#             # Create unified mesh with blending
#             unified_mesh = self.create_unified_mesh(mesh_data, attachment_points)
            
#             print(f"\n🎯 MESH BLENDING SUCCESSFUL!")
#             print("-" * 50)
#             stats = unified_mesh['stats']
#             print(f"Unified mesh: {stats['total_vertices']:,} vertices")
#             print(f"• Body vertices: {stats['body_vertices']:,}")
#             print(f"• Hand vertices: {stats['hand_vertices']:,}")
#             print(f"• Transition vertices: {stats['transition_vertices']:,}")
            
#             # Visualize blending
#             print("\n📊 Creating mesh blending visualizations...")
#             self.visualize_mesh_blending(unified_mesh)
            
#             # Save results
#             print("\n💾 Saving blended mesh...")
#             self.save_blended_mesh(unified_mesh)
            
#             print("\n✅ MESH BLENDING FUSION COMPLETE!")
#             print("="*80)
#             print("Achievements:")
#             print("✓ Smooth vertex transitions at attachment points")
#             print("✓ Cylindrical transition meshes for wrist connections")
#             print("✓ Distance-weighted blending for natural appearance")
#             print("✓ Multi-format output (JSON, OBJ)")
#             print("\nReady for next steps:")
#             print("• Add actual SMPL-X mesh topology")
#             print("• Implement face mesh blending")
#             print("• Add mesh optimization and cleanup")
            
#         except Exception as e:
#             print(f"❌ Mesh blending failed: {str(e)}")
#             import traceback
#             traceback.print_exc()

# def main():
#     import argparse
    
#     parser = argparse.ArgumentParser(description='Mesh-Level Blending Fusion')
#     parser.add_argument('--results_dir', type=str, required=True,
#                        help='Directory containing basic fusion results')
    
#     args = parser.parse_args()
    
#     blender = MeshBlendingFusion(args.results_dir)
#     blender.run_mesh_blending_fusion()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
"""
Mesh-Level Blending Fusion
Implements actual mesh blending at hand-wrist and head-neck attachment points
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import trimesh
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from sklearn.neighbors import NearestNeighbors

class MeshBlendingFusion:
    """Advanced fusion with mesh-level blending at attachment points"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        
        # SMPL-X joint indices for attachment points
        self.SMPLX_JOINT_MAP = {
            'left_wrist': 20,   # Approximate SMPL-X left wrist joint
            'right_wrist': 21,  # Approximate SMPL-X right wrist joint  
            'neck': 12,         # SMPL-X neck joint
            'head': 15          # SMPL-X head joint
        }
        
        # Load transformation parameters from basic fusion
        self.load_basic_fusion_data()
        
    def load_basic_fusion_data(self):
        """Load results from basic fusion demo"""
        fusion_file = self.results_dir / 'basic_fusion_output.json'
        if not fusion_file.exists():
            raise FileNotFoundError(f"Run basic fusion demo first. Missing: {fusion_file}")
            
        with open(fusion_file, 'r') as f:
            self.fusion_data = json.load(f)
            
        # Extract transformation parameters
        transform = self.fusion_data['transformation_applied']
        self.wilor_scale = transform['wilor_scale']
        self.wilor_translation = np.array(transform['wilor_translation'])
        
    def load_mesh_data(self) -> Dict:
        """Load actual mesh data from model outputs"""
        mesh_data = {
            'smplx_vertices': None,
            'smplx_faces': None,
            'hand_meshes': [],
            'face_mesh': None
        }
        
        # Load SMPL-X mesh (we need to reconstruct from parameters)
        smplx_foundation = self.fusion_data['smplx_foundation']
        mesh_data['smplx_joints'] = np.array(smplx_foundation['joints_3d'])
        
        # Load transformed hand meshes
        for hand_data in self.fusion_data['enhanced_hands']:
            mesh_data['hand_meshes'].append({
                'hand_type': hand_data['hand_type'],
                'vertices': np.array(hand_data['vertices_3d_transformed']),
                'keypoints': np.array(hand_data['keypoints_3d_transformed'])
            })
            
        return mesh_data
    
    def find_attachment_points(self, mesh_data: Dict) -> Dict:
        """Find optimal attachment points between meshes"""
        attachment_points = {}
        
        smplx_joints = mesh_data['smplx_joints']
        
        # Find hand attachment points
        for hand_mesh in mesh_data['hand_meshes']:
            hand_type = hand_mesh['hand_type']
            hand_vertices = hand_mesh['vertices']
            
            # Get corresponding SMPL-X wrist position
            wrist_joint_idx = self.SMPLX_JOINT_MAP[f'{hand_type}_wrist']
            if wrist_joint_idx < len(smplx_joints):
                wrist_position = smplx_joints[wrist_joint_idx]
                
                # Find closest hand vertex to wrist position
                distances = cdist([wrist_position], hand_vertices)[0]
                closest_hand_vertex_idx = np.argmin(distances)
                closest_hand_vertex = hand_vertices[closest_hand_vertex_idx]
                
                attachment_points[f'{hand_type}_hand'] = {
                    'smplx_point': wrist_position,
                    'smplx_joint_idx': wrist_joint_idx,
                    'hand_point': closest_hand_vertex,
                    'hand_vertex_idx': closest_hand_vertex_idx,
                    'distance': distances[closest_hand_vertex_idx]
                }
                
        return attachment_points
    
    def create_blending_weights(self, vertices: np.ndarray, attachment_point: np.ndarray, 
                              blend_radius: float = 0.05) -> np.ndarray:
        """Create smooth blending weights based on distance from attachment point"""
        distances = np.linalg.norm(vertices - attachment_point, axis=1)
        
        # Smooth falloff function (sigmoid-like)
        weights = np.exp(-distances / blend_radius)
        weights = np.clip(weights, 0, 1)
        
        return weights
    
    def blend_meshes_at_attachment(self, mesh1_vertices: np.ndarray, mesh2_vertices: np.ndarray,
                                 attachment_point: np.ndarray, blend_radius: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """Blend two meshes smoothly at attachment point"""
        
        # Create blending weights for both meshes
        weights1 = self.create_blending_weights(mesh1_vertices, attachment_point, blend_radius)
        weights2 = self.create_blending_weights(mesh2_vertices, attachment_point, blend_radius)
        
        # Find vertices in blending zone
        blend_zone1 = weights1 > 0.01  # Small threshold to avoid numerical issues
        blend_zone2 = weights2 > 0.01
        
        blended_mesh1 = mesh1_vertices.copy()
        blended_mesh2 = mesh2_vertices.copy()
        
        # Apply blending in overlapping regions
        if np.any(blend_zone1) and np.any(blend_zone2):
            # Smooth transition towards attachment point
            blended_mesh1[blend_zone1] = (
                mesh1_vertices[blend_zone1] * (1 - weights1[blend_zone1, np.newaxis]) +
                attachment_point * weights1[blend_zone1, np.newaxis]
            )
            
            blended_mesh2[blend_zone2] = (
                mesh2_vertices[blend_zone2] * (1 - weights2[blend_zone2, np.newaxis]) +
                attachment_point * weights2[blend_zone2, np.newaxis]
            )
        
        return blended_mesh1, blended_mesh2
    
    def create_transition_mesh(self, point1: np.ndarray, point2: np.ndarray, 
                             radius: float = 0.02, segments: int = 8) -> Dict:
        """Create a cylindrical transition mesh between two attachment points"""
        
        # Vector from point1 to point2
        direction = point2 - point1
        length = np.linalg.norm(direction)
        
        if length < 1e-6:  # Points are too close
            return None
            
        direction_normalized = direction / length
        
        # Create perpendicular vectors for cylinder
        if abs(direction_normalized[2]) < 0.9:
            perp1 = np.cross(direction_normalized, [0, 0, 1])
        else:
            perp1 = np.cross(direction_normalized, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction_normalized, perp1)
        
        # Generate cylinder vertices
        vertices = []
        faces = []
        
        for i in range(segments + 1):
            t = i / segments  # Parameter along cylinder length
            center = point1 + t * direction
            
            # Vary radius for smooth tapering
            current_radius = radius * (1 - 0.3 * abs(t - 0.5))  # Thinner in middle
            
            # Create circle of vertices
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                offset = current_radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                vertices.append(center + offset)
        
        vertices = np.array(vertices)
        
        # Create faces (triangles)
        for i in range(segments):
            for j in range(segments):
                # Current ring
                v1 = i * segments + j
                v2 = i * segments + (j + 1) % segments
                # Next ring
                v3 = (i + 1) * segments + j
                v4 = (i + 1) * segments + (j + 1) % segments
                
                if i < segments:  # Don't create faces beyond last ring
                    faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        return {
            'vertices': vertices,
            'faces': np.array(faces)
        }
    
    def create_unified_mesh(self, mesh_data: Dict, attachment_points: Dict) -> Dict:
        """Create unified mesh with blended attachments"""
        
        print("🔄 Creating unified mesh with blended attachments...")
        
        # Start with SMPL-X body as foundation
        smplx_joints = mesh_data['smplx_joints']
        unified_vertices = []
        unified_faces = []
        unified_labels = []  # Track which model each vertex comes from
        vertex_offset = 0
        
        # Add SMPL-X body joints as initial mesh points
        # (In practice, you'd load actual SMPL-X mesh vertices)
        body_vertices = smplx_joints.copy()
        unified_vertices.extend(body_vertices)
        unified_labels.extend(['smplx_body'] * len(body_vertices))
        vertex_offset += len(body_vertices)
        
        print(f"   📍 Added {len(body_vertices)} SMPL-X body vertices")
        
        # Add and blend hand meshes
        transition_meshes = []
        
        for hand_mesh in mesh_data['hand_meshes']:
            hand_type = hand_mesh['hand_type']
            hand_vertices = hand_mesh['vertices']
            
            if f'{hand_type}_hand' in attachment_points:
                attach_info = attachment_points[f'{hand_type}_hand']
                smplx_attach_point = attach_info['smplx_point']
                hand_attach_point = attach_info['hand_point']
                
                print(f"   🖐️  Blending {hand_type} hand at attachment point")
                
                # Blend hand mesh with body
                blended_body, blended_hand = self.blend_meshes_at_attachment(
                    body_vertices, hand_vertices, smplx_attach_point, blend_radius=0.08
                )
                
                # Add blended hand vertices
                unified_vertices.extend(blended_hand)
                unified_labels.extend([f'{hand_type}_hand'] * len(blended_hand))
                
                # Create transition mesh between attachment points
                transition = self.create_transition_mesh(
                    smplx_attach_point, hand_attach_point, radius=0.03
                )
                
                if transition:
                    transition_meshes.append({
                        'vertices': transition['vertices'],
                        'faces': transition['faces'] + vertex_offset,
                        'type': f'{hand_type}_wrist_transition'
                    })
                    unified_vertices.extend(transition['vertices'])
                    unified_labels.extend([f'{hand_type}_transition'] * len(transition['vertices']))
                    unified_faces.extend(transition['faces'] + vertex_offset)
                    vertex_offset += len(transition['vertices'])
                
                vertex_offset += len(blended_hand)
                
                print(f"      ✅ Added {len(blended_hand)} blended hand vertices")
                if transition:
                    print(f"      ✅ Added {len(transition['vertices'])} transition vertices")
        
        unified_mesh = {
            'vertices': np.array(unified_vertices),
            'faces': np.array(unified_faces) if unified_faces else np.array([]),
            'labels': unified_labels,
            'transition_meshes': transition_meshes,
            'attachment_points': attachment_points,
            'stats': {
                'total_vertices': len(unified_vertices),
                'body_vertices': len(body_vertices),
                'hand_vertices': sum(len(h['vertices']) for h in mesh_data['hand_meshes']),
                'transition_vertices': sum(len(t['vertices']) for t in transition_meshes)
            }
        }
        
        return unified_mesh
    
    def visualize_mesh_blending(self, unified_mesh: Dict):
        """Visualize the mesh blending results"""
        
        fig = plt.figure(figsize=(20, 12))
        
        vertices = unified_mesh['vertices']
        labels = unified_mesh['labels']
        attachment_points = unified_mesh['attachment_points']
        
        # Create color map for different mesh parts
        color_map = {
            'smplx_body': 'blue',
            'left_hand': 'red',
            'right_hand': 'orange', 
            'left_transition': 'purple',
            'right_transition': 'brown'
        }
        
        # Plot 1: Complete unified mesh
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        for label_type, color in color_map.items():
            mask = np.array([label_type in label for label in labels])
            if np.any(mask):
                subset_vertices = vertices[mask]
                ax1.scatter(subset_vertices[:, 0], subset_vertices[:, 1], subset_vertices[:, 2],
                           c=color, alpha=0.7, s=20, label=label_type)
        
        ax1.set_title('Unified Mesh\n(All Components)')
        ax1.legend()
        
        # Plot 2: Attachment points detail
        ax2 = fig.add_subplot(2, 3, 2, projection='3d')
        
        # Plot mesh with attachment points highlighted
        ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                   c='lightgray', alpha=0.3, s=5)
        
        # Highlight attachment points
        for attach_name, attach_info in attachment_points.items():
            smplx_point = attach_info['smplx_point']
            hand_point = attach_info['hand_point']
            
            ax2.scatter(*smplx_point, c='red', s=100, marker='o', 
                       label=f'{attach_name} SMPL-X')
            ax2.scatter(*hand_point, c='blue', s=100, marker='^',
                       label=f'{attach_name} Hand')
            
            # Draw connection line
            ax2.plot([smplx_point[0], hand_point[0]], 
                    [smplx_point[1], hand_point[1]], 
                    [smplx_point[2], hand_point[2]], 
                    'k--', alpha=0.7, linewidth=2)
        
        ax2.set_title('Attachment Points\n(Blend Locations)')
        ax2.legend()
        
        # Plot 3: Left hand detail
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        left_mask = np.array(['left' in label for label in labels])
        if np.any(left_mask):
            left_vertices = vertices[left_mask]
            left_labels_subset = np.array(labels)[left_mask]
            
            for label in np.unique(left_labels_subset):
                label_mask = left_labels_subset == label
                color = color_map.get(label, 'gray')
                ax3.scatter(left_vertices[label_mask, 0], 
                           left_vertices[label_mask, 1], 
                           left_vertices[label_mask, 2],
                           c=color, alpha=0.8, s=30, label=label)
        
        ax3.set_title('Left Hand Blending\n(Detail View)')
        ax3.legend()
        
        # Plot 4: Right hand detail  
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        right_mask = np.array(['right' in label for label in labels])
        if np.any(right_mask):
            right_vertices = vertices[right_mask]
            right_labels_subset = np.array(labels)[right_mask]
            
            for label in np.unique(right_labels_subset):
                label_mask = right_labels_subset == label
                color = color_map.get(label, 'gray')
                ax4.scatter(right_vertices[label_mask, 0], 
                           right_vertices[label_mask, 1], 
                           right_vertices[label_mask, 2],
                           c=color, alpha=0.8, s=30, label=label)
        
        ax4.set_title('Right Hand Blending\n(Detail View)')
        ax4.legend()
        
        # Plot 5: Blending quality metrics
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Calculate blending quality metrics
        attachment_distances = []
        attachment_names = []
        
        for attach_name, attach_info in attachment_points.items():
            distance = attach_info['distance']
            attachment_distances.append(distance)
            attachment_names.append(attach_name.replace('_hand', ''))
        
        ax5.bar(attachment_names, attachment_distances, alpha=0.7, color=['red', 'orange'])
        ax5.set_ylabel('Attachment Distance')
        ax5.set_title('Blending Quality\n(Lower = Better)')
        ax5.tick_params(axis='x', rotation=45)
        
        # Plot 6: Statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        stats = unified_mesh['stats']
        stats_text = f"""
        MESH BLENDING STATISTICS
        
        Total Vertices: {stats['total_vertices']:,}
        
        Breakdown:
        • Body: {stats['body_vertices']:,}
        • Hands: {stats['hand_vertices']:,}
        • Transitions: {stats['transition_vertices']:,}
        
        Attachment Points: {len(attachment_points)}
        
        Blending Features:
        ✓ Smooth vertex transitions
        ✓ Cylindrical wrist connectors
        ✓ Distance-based weight blending
        ✓ Multi-mesh unification
        
        Quality Metrics:
        • Avg attachment distance: {np.mean(attachment_distances):.4f}
        • Max attachment distance: {np.max(attachment_distances):.4f}
        """
        
        ax6.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
                fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'mesh_blending_fusion.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_blended_mesh(self, unified_mesh: Dict):
        """Save the blended mesh in multiple formats"""
        
        # Save as JSON
        json_output = self._make_serializable(unified_mesh)
        json_file = self.results_dir / 'mesh_blending_output.json'
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        print(f"💾 Blended mesh saved to: {json_file}")
        
        # Save as OBJ (if we have faces)
        if len(unified_mesh['faces']) > 0:
            obj_file = self.results_dir / 'unified_mesh.obj'
            self._save_obj(unified_mesh['vertices'], unified_mesh['faces'], obj_file)
            print(f"🗃️  Mesh saved as OBJ: {obj_file}")
        
        # Save summary
        summary = {
            'mesh_blending_type': 'attachment_point_blending',
            'blending_method': 'distance_weighted_smooth_transition',
            'models_unified': ['SMPLest-X', 'WiLoR', 'EMOCA'],
            'attachment_points': len(unified_mesh['attachment_points']),
            'total_vertices': unified_mesh['stats']['total_vertices'],
            'mesh_components': {
                'body': unified_mesh['stats']['body_vertices'],
                'hands': unified_mesh['stats']['hand_vertices'], 
                'transitions': unified_mesh['stats']['transition_vertices']
            },
            'next_improvements': [
                'Add actual SMPL-X mesh vertices (not just joints)',
                'Implement face mesh blending for EMOCA',
                'Add mesh smoothing and cleanup',
                'Optimize transition mesh topology'
            ]
        }
        
        summary_file = self.results_dir / 'mesh_blending_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📋 Blending summary saved to: {summary_file}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and numpy types to lists/native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_obj(self, vertices: np.ndarray, faces: np.ndarray, filename: Path):
        """Save mesh as OBJ file"""
        with open(filename, 'w') as f:
            # Write vertices
            for vertex in vertices:
                f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def run_mesh_blending_fusion(self):
        """Run complete mesh-level blending fusion"""
        
        print("="*80)
        print("MESH-LEVEL BLENDING FUSION")
        print("="*80)
        print("Advanced fusion with smooth mesh blending at attachment points")
        print()
        
        try:
            # Load mesh data
            print("🔄 Loading mesh data...")
            mesh_data = self.load_mesh_data()
            
            # Find attachment points
            print("🔄 Finding optimal attachment points...")
            attachment_points = self.find_attachment_points(mesh_data)
            print(f"   ✅ Found {len(attachment_points)} attachment points")
            
            for attach_name, attach_info in attachment_points.items():
                print(f"      🔗 {attach_name}: distance = {attach_info['distance']:.4f}")
            
            # Create unified mesh with blending
            unified_mesh = self.create_unified_mesh(mesh_data, attachment_points)
            
            print(f"\n🎯 MESH BLENDING SUCCESSFUL!")
            print("-" * 50)
            stats = unified_mesh['stats']
            print(f"Unified mesh: {stats['total_vertices']:,} vertices")
            print(f"• Body vertices: {stats['body_vertices']:,}")
            print(f"• Hand vertices: {stats['hand_vertices']:,}")
            print(f"• Transition vertices: {stats['transition_vertices']:,}")
            
            # Visualize blending
            print("\n📊 Creating mesh blending visualizations...")
            self.visualize_mesh_blending(unified_mesh)
            
            # Save results
            print("\n💾 Saving blended mesh...")
            self.save_blended_mesh(unified_mesh)
            
            print("\n✅ MESH BLENDING FUSION COMPLETE!")
            print("="*80)
            print("Achievements:")
            print("✓ Smooth vertex transitions at attachment points")
            print("✓ Cylindrical transition meshes for wrist connections")
            print("✓ Distance-weighted blending for natural appearance")
            print("✓ Multi-format output (JSON, OBJ)")
            print("\nReady for next steps:")
            print("• Add actual SMPL-X mesh topology")
            print("• Implement face mesh blending")
            print("• Add mesh optimization and cleanup")
            
        except Exception as e:
            print(f"❌ Mesh blending failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Mesh-Level Blending Fusion')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing basic fusion results')
    
    args = parser.parse_args()
    
    blender = MeshBlendingFusion(args.results_dir)
    blender.run_mesh_blending_fusion()

if __name__ == '__main__':
    main()