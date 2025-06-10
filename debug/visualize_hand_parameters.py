#!/usr/bin/env python3
"""
Visualize hand parameters to spot issues
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_hand_parameters(results_dir: str):
    """Create detailed visualizations of hand parameters"""
    
    results_dir = Path(results_dir)
    viz_dir = results_dir / 'hand_param_visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    # Load all relevant parameters
    # Original SMPLX
    smplx_params = None
    for param_file in results_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'):
        with open(param_file, 'r') as f:
            smplx_params = json.load(f)
            break
    
    # Fused parameters
    fusion_dir = results_dir / 'fusion_results'
    with open(fusion_dir / 'fused_parameters.json', 'r') as f:
        fused_params = json.load(f)
    
    # WiLoR raw
    wilor_params = None
    for param_file in results_dir.glob('wilor_results/*_parameters.json'):
        with open(param_file, 'r') as f:
            wilor_params = json.load(f)
            break
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Parameter values comparison
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(smplx_params['left_hand_pose'], 'b-', label='SMPLX Original', alpha=0.7)
    ax1.plot(fused_params['left_hand_pose'], 'r-', label='Fused', linewidth=2)
    ax1.set_title('Left Hand Parameters')
    ax1.set_xlabel('Parameter Index')
    ax1.set_ylabel('Value (radians)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(smplx_params['right_hand_pose'], 'b-', label='SMPLX Original', alpha=0.7)
    ax2.plot(fused_params['right_hand_pose'], 'r-', label='Fused', linewidth=2)
    ax2.set_title('Right Hand Parameters')
    ax2.set_xlabel('Parameter Index')
    ax2.set_ylabel('Value (radians)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 2: Per-finger analysis
    ax3 = plt.subplot(3, 2, 3)
    left_fused = np.array(fused_params['left_hand_pose']).reshape(15, 3)
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    finger_norms = []
    for i in range(5):
        joints = left_fused[i*3:(i+1)*3]
        norms = np.linalg.norm(joints, axis=1)
        finger_norms.append(norms)
        ax3.plot(norms, 'o-', label=finger_names[i], markersize=8)
    ax3.set_title('Left Hand - Per Finger Joint Norms')
    ax3.set_xlabel('Joint (Base, Mid, Tip)')
    ax3.set_ylabel('Rotation Magnitude')
    ax3.legend()
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(['Base', 'Mid', 'Tip'])
    ax3.grid(True, alpha=0.3)
    
    # Plot 3: WiLoR raw values
    ax4 = plt.subplot(3, 2, 4)
    for hand in wilor_params.get('hands', []):
        if 'mano_parameters' in hand:
            params = hand['mano_parameters']['parameters']
            if 'hand_pose' in params:
                values = np.array(params['hand_pose']['values']).flatten()
                hand_type = hand.get('hand_type', 'unknown')
                ax4.plot(values[:45], 'o-', label=f'WiLoR {hand_type}', markersize=4)
    ax4.set_title('WiLoR Raw Rotation Matrix Values')
    ax4.set_xlabel('Value Index')
    ax4.set_ylabel('Matrix Element Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 4: Difference heatmap
    ax5 = plt.subplot(3, 2, 5)
    left_diff = np.array(fused_params['left_hand_pose']) - np.array(smplx_params['left_hand_pose'])
    right_diff = np.array(fused_params['right_hand_pose']) - np.array(smplx_params['right_hand_pose'])
    
    diff_matrix = np.vstack([left_diff.reshape(15, 3), right_diff.reshape(15, 3)])
    im = ax5.imshow(diff_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    ax5.set_title('Parameter Differences (Fused - Original)')
    ax5.set_ylabel('Joint Index')
    ax5.set_xlabel('Axis (X, Y, Z)')
    ax5.set_yticks(range(30))
    ax5.set_yticklabels([f'L{i//3}' if i < 15 else f'R{i//3-5}' for i in range(30)])
    plt.colorbar(im, ax=ax5)
    
    # Plot 5: Statistics
    ax6 = plt.subplot(3, 2, 6)
    ax6.axis('off')
    stats_text = f"""HAND PARAMETER STATISTICS
    
Left Hand:
  Original: min={min(smplx_params['left_hand_pose']):.3f}, max={max(smplx_params['left_hand_pose']):.3f}
  Fused:    min={min(fused_params['left_hand_pose']):.3f}, max={max(fused_params['left_hand_pose']):.3f}
  Mean diff: {np.mean(left_diff):.3f}
  Std diff:  {np.std(left_diff):.3f}
  
Right Hand:
  Original: min={min(smplx_params['right_hand_pose']):.3f}, max={max(smplx_params['right_hand_pose']):.3f}
  Fused:    min={min(fused_params['right_hand_pose']):.3f}, max={max(fused_params['right_hand_pose']):.3f}
  Mean diff: {np.mean(right_diff):.3f}
  Std diff:  {np.std(right_diff):.3f}
  
Fusion Success Indicators:
  ✓ Parameters changed: {np.std(left_diff) > 0.01 and np.std(right_diff) > 0.01}
  ✓ Reasonable range: {-4 < min(fused_params['left_hand_pose']) and max(fused_params['left_hand_pose']) < 4}
  ? Natural pose: Check visual output
"""
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'hand_parameter_analysis.png', dpi=150)
    print(f"✅ Saved visualization to {viz_dir / 'hand_parameter_analysis.png'}")
    
    # Save numerical analysis
    analysis = {
        'left_hand': {
            'original_range': [float(min(smplx_params['left_hand_pose'])), 
                              float(max(smplx_params['left_hand_pose']))],
            'fused_range': [float(min(fused_params['left_hand_pose'])), 
                           float(max(fused_params['left_hand_pose']))],
            'mean_change': float(np.mean(left_diff)),
            'std_change': float(np.std(left_diff)),
            'per_finger_norms': [[float(n) for n in norms] for norms in finger_norms]
        },
        'right_hand': {
            'original_range': [float(min(smplx_params['right_hand_pose'])), 
                              float(max(smplx_params['right_hand_pose']))],
            'fused_range': [float(min(fused_params['right_hand_pose'])), 
                           float(max(fused_params['right_hand_pose']))],
            'mean_change': float(np.mean(right_diff)),
            'std_change': float(np.std(right_diff))
        }
    }
    
    with open(viz_dir / 'parameter_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"✅ Saved analysis to {viz_dir / 'parameter_analysis.json'}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    args = parser.parse_args()
    
    visualize_hand_parameters(args.results_dir)

