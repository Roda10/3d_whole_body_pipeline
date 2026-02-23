# 3D Whole Body Pipeline Instructions
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

## 1. Running the Complete Pipeline

### Option A: Using the Main Pipeline Script (Recommended)
This runs all three models (SMPLest-X, WiLoR, and EMOCA) in a unified way:
```bash
python main.py --input_image data/full_images/test2.jpg --output_dir pipeline_results
```

### Option B: Using Individual Adapters
If you need to run adapters separately (from the `adapters/` directory):

1. SMPLest-X Adapter:
```bash
python smplestx_adapter.py \
    --cfg_path ../external/body/SMPLest-X/configs/config_smplest_x_h.py \
    --input_image ../data/full_images/test2.jpg \
    --output_dir ./smplestx_results \
    --multi_person
```

2. WiLoR Adapter:
```bash
python wilor_adapter.py \
    --img_folder ../data/full_images/ \
    --out_folder ../data/outputs/wilor/
```

3. EMOCA Adapter:
```bash
python emoca_adapter.py \
    --input_folder ../data/full_images/ \
    --output_folder ../data/outputs/EMOCA_outputs \
    --model_name EMOCA
```

## 2. Running the Evaluation

### Option A: Using the Evaluation Script (with Progress Tracking)
```bash
# Start evaluation
./bash_scripts/run_eval.sh start

# Check status
./bash_scripts/run_eval.sh status

# View logs in real-time
./bash_scripts/run_eval.sh logs

# Stop evaluation if needed
./bash_scripts/run_eval.sh stop
```

### Option B: Direct Evaluation Commands
```bash
# Quick test (10 frames)
python evaluation/ehf_fusion_evaluator.py --verbose_output

# Custom number of frames
python evaluation/ehf_fusion_evaluator.py --max_frames 5 --verbose_output

# Full evaluation (all frames)
python evaluation/ehf_fusion_evaluator.py --max_frames 0 --verbose_output
```

## Output Locations

### Pipeline Results
- Main output: `pipeline_results/run_TIMESTAMP/`
  - SMPLest-X results: `smplestx_results/`
  - WiLoR results: `wilor_results/`
  - EMOCA results: `emoca_results/`
  - Pipeline log: `pipeline.log`
  - Summary: `pipeline_summary.json`

### Evaluation Results
- Main output: `evaluation_results/ehf_compatible_opt_TIMESTAMP/`
  - Comparison results: `evaluation_comparison_results.json`
  - Visual comparisons: `gallery/`
  - Evaluation logs: `logs/`
  - Runtime logs: `final_evaluation.log`

## Required Data Structure
```
data/
├── full_images/          # Your input images
├── EHF/                  # EHF dataset for evaluation
└── outputs/              # Additional output directory

pretrained_models/
└── smplest_x/           # SMPLest-X model files
    └── config_base.py

human_models/
└── human_model_files/   # Required model files
```

## Advanced Usage (Debug & Analysis Tools)

These tools are available for debugging and detailed analysis if needed:

### Parameter Analysis
```bash
python analysis_tools/parameter_analyzer.py --results_dir pipeline_results/run_TIMESTAMP
python analysis_tools/coordinate_analyzer_fixed.py --results_dir pipeline_results/run_20260129_143810
```

### Visualization Tools
```bash
python fusion/fusion_visualizer.py --results_dir pipeline_results/run_TIMESTAMP
```

### Hand-specific Analysis
```bash
python analysis_tools/validate_hand_transformations.py --results_dir pipeline_results/run_TIMESTAMP
python debug/check_joints_ordering.py --results_dir pipeline_results/run_TIMESTAMP
```