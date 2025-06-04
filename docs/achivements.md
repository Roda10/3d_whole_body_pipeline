================================================================================
PARAMETER STRUCTURE ANALYSIS
================================================================================
Analyzing pipeline outputs in: pipeline_results/run_20250604_091347

🔍 Analyzing SMPLest-X outputs...
🔍 Analyzing WiLoR outputs...
🔍 Analyzing EMOCA outputs...
📝 Creating human-readable summary...

📊 SUMMARY:
--------------------------------------------------

SMPLest-X:
  Full-body 3D human pose and shape estimation
  Parameters found: 11
    • joints_3d: Multi-dimensional array: 137 × 3
    • joints_2d: Multi-dimensional array: 137 × 2
    • root_pose: List of 3 numbers
    • body_pose: List of 63 numbers
    • left_hand_pose: List of 45 numbers
    • right_hand_pose: List of 45 numbers
    • jaw_pose: List of 3 numbers
    • betas: List of 10 numbers
    • expression: List of 10 numbers
    • translation: List of 3 numbers
    • mesh: Multi-dimensional array: 10475 × 3

WiLoR:
  Hand pose estimation with MANO parameters
  Parameters found: 27
    • image_path: Single str value
    • batch_size: Single int value
    • detection_count: Single int value
    • scaled_focal_length: Single float value
    • hand_id: Single int value
    • hand_type: Single str value
    • is_right: Single bool value
    • vertices_3d: Multi-dimensional array: 778 × 3
    • keypoints_3d: Multi-dimensional array: 21 × 3
    • camera_prediction: List of 3 numbers
    • camera_translation: List of 3 numbers
    • box_center: List of 2 numbers
    • box_size: Single float value
    • img_size: List of 2 numbers
    • source: Single str value
    • values: Multi-dimensional array: 1 × 3 × 3
    • shape: List of 3 numbers
    • type: Single str value
    • x: List of 2 numbers
    • y: List of 2 numbers
    • z: List of 2 numbers
    • vertices_center: List of 3 numbers
    • extractor_version: Single str value
    • includes_3d_coordinates: Single bool value
    • includes_mano_parameters: Single bool value
    • coordinate_system: Single str value
    • notes: Single str value

EMOCA:
  Facial expression and identity modeling
  Parameters found: 5
    • shapecode: List of 100 numbers
    • expcode: List of 50 numbers
    • texcode: List of 50 numbers
    • posecode: List of 6 numbers
    • detailcode: List of 128 numbers

💾 Analysis saved to:
   📄 Detailed: pipeline_results/run_20250604_091347/parameter_structure_analysis.json
   📄 Summary: pipeline_results/run_20250604_091347/pipeline_output_summary.json
================================================================================
