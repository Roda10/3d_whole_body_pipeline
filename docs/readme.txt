export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

------------------------
# to run the main
python main.py --input_image data/full_images/test2.jpg --output_dir pipeline_results

>>> Output:

pipeline_results/
└── run_YYYYMMDD_HHMMSS/
    ├── smplestx_results/
    ├── wilor_results/
    ├── emoca_results/
    ├── pipeline.log
    └── pipeline_summary.json

# Basic quick test

python evaluation/ehf_fusion_evaluator.py --max_frames 10 --verbose_output

# Run and evaluate

# Start evaluation
./bash_scripts/run_eval.sh start

# Check status
./bash_scripts/run_eval.sh status

# View logs in real-time
./bash_scripts/run_eval.sh logs

# Stop evaluation if needed
./bash_scripts/run_eval.sh stop