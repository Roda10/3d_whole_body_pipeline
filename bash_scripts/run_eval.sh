#!/bin/bash

# ==============================================================================
#  Robust Evaluation Control Script
#
#  Usage:
#    ./run_eval.sh start    - Starts the evaluation in the background
#    ./run_eval.sh stop     - Stops the running evaluation
#    ./run_eval.sh status   - Checks if the evaluation is running
#    ./run_eval.sh logs     - Shows the live log output
# ==============================================================================

# --- Configuration ---
# The python script to execute
PY_SCRIPT="evaluation/ehf_fusion_evaluator.py"
# The file where we will store the Process ID (PID)
PID_FILE="evaluation.pid"
# The log file where all output will be stored
LOG_FILE="final_evaluation.log"
# Arguments for the python script
SCRIPT_ARGS="--max_frames 1"

# --- Function to start the process ---
start() {
    # 1. The Safety Check: Check if the PID file exists
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        # Check if a process with that PID is actually running
        if ps -p $PID > /dev/null; then
            echo "❌ Error: Evaluation is already running with PID $PID. Please run './run_eval.sh stop' first."
            exit 1
        else
            echo "⚠️  Warning: Found a stale PID file. Cleaning it up."
            rm "$PID_FILE"
        fi
    fi

    # 2. Launch the process
    echo "🚀 Starting evaluation..."
    # Use nohup to run in the background and redirect output
    nohup python -u "$PY_SCRIPT" $SCRIPT_ARGS > "$LOG_FILE" 2>&1 &
    
    # 3. Capture and save the PID of the new process
    # The '$!' variable holds the PID of the last command run in the background
    NEW_PID=$!
    echo $NEW_PID > "$PID_FILE"
    
    echo "✅ Evaluation started successfully!"
    echo "   - Process ID (PID): $NEW_PID"
    echo "   - Log file:       $LOG_FILE"
    echo "   - To view logs, run: ./run_eval.sh logs"
}

# --- Function to stop the process ---
stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "ℹ️  Info: Evaluation is not running (no PID file found)."
        return
    fi
    
    PID=$(cat "$PID_FILE")
    echo "🛑 Stopping evaluation process with PID $PID..."
    # Send the termination signal
    kill $PID
    # Wait a moment to see if it terminated cleanly
    sleep 2
    
    # Check if the process is still alive
    if ps -p $PID > /dev/null; then
        echo "⚠️  Process did not stop cleanly. Forcing shutdown (kill -9)..."
        kill -9 $PID
    fi
    
    # Clean up the PID file
    rm "$PID_FILE"
    echo "✅ Evaluation stopped."
}

# --- Function to check the status ---
status() {
    if [ ! -f "$PID_FILE" ]; then
        echo "ℹ️  Status: Evaluation is STOPPED."
        return
    fi
    
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "✅ Status: Evaluation is RUNNING with PID $PID."
    else
        echo "❌ Status: Evaluation is STOPPED, but a stale PID file was found. Run './run_eval.sh stop' to clean up."
    fi
}

# --- Function to view logs ---
logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo "ℹ️  Info: Log file does not exist yet. Start the process first."
        return
    fi
    echo "📄 Displaying live logs from '$LOG_FILE' (Press Ctrl+C to stop)..."
    tail -f "$LOG_FILE"
}


# --- Main Logic ---
# This part reads the first argument ($1) passed to the script (start, stop, etc.)
# and calls the corresponding function.

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    logs)
        logs
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs}"
        exit 1
esac