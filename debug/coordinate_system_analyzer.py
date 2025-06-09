# FILE: fusion/debug_shape_inspector.py
# (Corrected version to fix the file handle error)

import numpy as np
import json
from pathlib import Path
import argparse

class ShapeInspector:
    """
    Inspects and reports the true shape and data type of hand pose parameters
    from SMPLest-X and WiLoR outputs.
    """
    def __init__(self, run_dir: str):
        self.run_dir = Path(run_dir)
        if not self.run_dir.is_dir():
            raise FileNotFoundError(f"The specified run directory does not exist: {self.run_dir}")
        
        self.output_file = self.run_dir / "hand_parameter_shape_report.txt"

    def _inspect_smplestx(self, report_file_handle):
        """Loads and reports on SMPLest-X hand poses."""
        report_file_handle.write("========================================\n")
        report_file_handle.write(" SMPLest-X Hand Pose Inspection\n")
        report_file_handle.write("========================================\n")
        try:
            param_file = next(self.run_dir.glob('smplestx_results/*/person_*/smplx_params_*.json'))
            
            # --- FIX IS HERE: Use a different variable name for the file handle ---
            with open(param_file, 'r') as json_file:
                data = json.load(json_file)
            # --- END OF FIX ---
            
            report_file_handle.write(f"Source file: {param_file.relative_to(self.run_dir.parent)}\n\n")

            left_hand_pose = np.array(data['left_hand_pose'])
            right_hand_pose = np.array(data['right_hand_pose'])

            report_file_handle.write(f"Parameter: 'left_hand_pose'\n")
            report_file_handle.write(f"  - Data Type: {left_hand_pose.dtype}\n")
            report_file_handle.write(f"  - Shape: {left_hand_pose.shape}\n")
            report_file_handle.write(f"  - Total Elements: {left_hand_pose.size}\n\n")
            
            report_file_handle.write(f"Parameter: 'right_hand_pose'\n")
            report_file_handle.write(f"  - Data Type: {right_hand_pose.dtype}\n")
            report_file_handle.write(f"  - Shape: {right_hand_pose.shape}\n")
            report_file_handle.write(f"  - Total Elements: {right_hand_pose.size}\n")

        except StopIteration:
            report_file_handle.write("ERROR: Could not find SMPLest-X parameter file.\n")
        except KeyError as e:
            report_file_handle.write(f"ERROR: Key {e} not found in SMPLest-X parameters.\n")

    def _inspect_wilor(self, report_file_handle):
        """Loads and reports on WiLoR MANO parameters."""
        report_file_handle.write("\n\n========================================\n")
        report_file_handle.write(" WiLoR (MANO) Hand Pose Inspection\n")
        report_file_handle.write("========================================\n")
        try:
            param_file = next(self.run_dir.glob('wilor_results/*_parameters.json'))
            
            # --- FIX IS HERE: Use a different variable name for the file handle ---
            with open(param_file, 'r') as json_file:
                data = json.load(json_file)
            # --- END OF FIX ---

            report_file_handle.write(f"Source file: {param_file.relative_to(self.run_dir.parent)}\n\n")
            
            if not data.get('hands'):
                report_file_handle.write("No 'hands' detected in WiLoR output.\n")
                return

            for i, hand in enumerate(data['hands']):
                hand_type = hand.get('hand_type', f'Unknown_{i}')
                report_file_handle.write(f"--- Hand: {hand_type.upper()} ---\n")
                
                try:
                    mano_params = hand['mano_parameters']['parameters']
                    hand_pose_data = np.array(mano_params['hand_pose']['values'])
                    
                    report_file_handle.write(f"Parameter: mano_parameters['parameters']['hand_pose']['values']\n")
                    report_file_handle.write(f"  - Data Type: {hand_pose_data.dtype}\n")
                    report_file_handle.write(f"  - Shape straight from JSON: {hand_pose_data.shape}\n")
                    report_file_handle.write(f"  - Total Elements: {hand_pose_data.size}\n\n")

                except KeyError as e:
                    report_file_handle.write(f"  ERROR: Key {e} not found in MANO parameters for this hand.\n\n")

        except StopIteration:
            report_file_handle.write("ERROR: Could not find WiLoR parameter file.\n")

    def run_inspection(self):
        """Main execution method."""
        print(f"🔬 Inspecting hand parameter shapes for run: {self.run_dir.name}")
        with open(self.output_file, 'w') as f:
            self._inspect_smplestx(f)
            self._inspect_wilor(f)
        
        print(f"\n✅ Inspection complete. Report saved to:\n   {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Inspect the shape of hand parameters from pipeline outputs.")
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to a single pipeline run directory (e.g., ./pipeline_results/run_...)')
    args = parser.parse_args()
    
    # I have renamed the main class instance to avoid confusion
    inspector = ShapeInspector(args.run_dir)
    inspector.run_inspection()

if __name__ == '__main__':
    main()