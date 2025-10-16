# Example Usage of Enhanced Evaluation System

# 1. Initialize camera parameters for your dataset
from evaluation.camera_utils import CameraParameters

# For EHF dataset
camera_params = CameraParameters.from_ehf(Path("data/EHF/EHF_camera.txt"))

# For custom dataset
custom_camera = CameraParameters(
    focal_length=(1000.0, 1000.0),
    principal_point=(512, 512),
    dataset_type='custom'
)

# 2. Initialize metrics calculator
from evaluation.enhanced_metrics import EnhancedMetricsCalculator

# For EHF dataset
metrics_calc = EnhancedMetricsCalculator(
    dataset_type='EHF',
    unit='meters',
    verbose=True
)

# 3. Load and verify meshes
pred_mesh = metrics_calc.load_mesh(
    "path/to/predicted.obj",
    expected_unit='meters'
)
gt_mesh = metrics_calc.load_mesh(
    "path/to/groundtruth.ply",
    expected_unit='meters'
)

# 4. Calculate all metrics
results = metrics_calc.calculate_all_metrics(
    pred_mesh=pred_mesh,
    gt_mesh=gt_mesh,
    pred_joints=pred_joints,  # If available
    gt_joints=gt_joints      # If available
)

print("\nMetrics Results:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# 5. Project points to 2D (if needed)
points_3d = pred_mesh.vertices
points_2d = camera_params.project_points(
    points_3d,
    with_extrinsics=True,
    normalize=False
)