#!/bin/bash
# Quick Start Script for Persistence Services
# Usage: ./quick_start.sh [start|stop|restart|status|run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${PURPLE}================================================${NC}"
    echo -e "${PURPLE}🚀 3D Human Pose Fusion - Persistence Services${NC}"
    echo -e "${PURPLE}================================================${NC}"
}

print_usage() {
    echo "Usage: $0 [start|stop|restart|status|run] [options]"
    echo ""
    echo "Commands:"
    echo "  start              Start all persistence services"
    echo "  stop               Stop all persistence services"
    echo "  restart            Restart all persistence services"
    echo "  status             Check service status"
    echo "  run <image_path>   Run pipeline with services"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 run data/full_images/test2.jpg"
    echo "  $0 status"
    echo "  $0 stop"
}

check_requirements() {
    echo -e "${BLUE}🔍 Checking requirements...${NC}"
    
    # Check Python
    if ! command -v python &> /dev/null; then
        echo -e "${RED}❌ Python not found${NC}"
        exit 1
    fi
    
    # Check required directories
    if [ ! -d "services" ]; then
        echo -e "${YELLOW}📁 Creating services directory...${NC}"
        mkdir -p services
    fi
    
    if [ ! -d "external" ]; then
        echo -e "${RED}❌ External models directory not found${NC}"
        echo "Please ensure external/SMPLest-X, external/WiLoR, and external/emoca are available"
        exit 1
    fi
    
    # Check if service files exist
    SERVICES=("smplestx_service.py" "wilor_service.py" "emoca_service.py")
    for service in "${SERVICES[@]}"; do
        if [ ! -f "services/$service" ]; then
            echo -e "${RED}❌ Service file not found: services/$service${NC}"
            echo "Please create the service files first"
            exit 1
        fi
    done
    
    echo -e "${GREEN}✅ Requirements check passed${NC}"
}

install_dependencies() {
    echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
    
    # Check if requirements file exists, if not create basic one
    if [ ! -f "requirements_services.txt" ]; then
        cat > requirements_services.txt << EOF
fastapi
uvicorn
requests
psutil
pydantic
torch
torchvision
ultralytics
opencv-python
numpy
EOF
    fi
    
    pip install -r requirements_services.txt
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

start_services() {
    echo -e "${GREEN}🚀 Starting persistence services...${NC}"
    python service_manager.py start
}

stop_services() {
    echo -e "${YELLOW}🛑 Stopping persistence services...${NC}"
    python service_manager.py stop
}

restart_services() {
    echo -e "${BLUE}🔄 Restarting persistence services...${NC}"
    python service_manager.py restart
}

check_status() {
    echo -e "${BLUE}📊 Checking service status...${NC}"
    python service_manager.py status
}

run_pipeline() {
    local image_path="$1"
    
    if [ -z "$image_path" ]; then
        echo -e "${RED}❌ Image path required for run command${NC}"
        echo "Usage: $0 run <image_path>"
        exit 1
    fi
    
    if [ ! -f "$image_path" ]; then
        echo -e "${RED}❌ Image file not found: $image_path${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}🏃 Running persistence pipeline...${NC}"
    echo -e "${BLUE}Input: $image_path${NC}"
    
    # Start timing
    start_time=$(date +%s)
    
    # Run the persistence pipeline
    python main_persistence.py --input_image "$image_path" --output_dir pipeline_results_persistence
    
    # End timing
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo -e "${GREEN}✅ Pipeline completed in ${minutes}m ${seconds}s${NC}"
    
    # Show results location
    latest_result=$(ls -t pipeline_results_persistence/run_* | head -n1)
    if [ -n "$latest_result" ]; then
        echo -e "${BLUE}📁 Results: $latest_result${NC}"
    fi
}

benchmark_comparison() {
    local image_path="$1"
    
    if [ -z "$image_path" ]; then
        echo -e "${RED}❌ Image path required for benchmark${NC}"
        exit 1
    fi
    
    echo -e "${PURPLE}⚡ Running Performance Benchmark${NC}"
    echo -e "${PURPLE}=================================${NC}"
    
    # Run original pipeline
    echo -e "${BLUE}1️⃣  Testing Original Pipeline (Subprocess)...${NC}"
    start_time=$(date +%s)
    python main.py --input_image "$image_path" --output_dir benchmark_original
    original_time=$(($(date +%s) - start_time))
    
    echo ""
    
    # Run persistence pipeline
    echo -e "${BLUE}2️⃣  Testing Persistence Pipeline (Services)...${NC}"
    start_time=$(date +%s)
    python main_persistence.py --input_image "$image_path" --output_dir benchmark_persistence
    persistence_time=$(($(date +%s) - start_time))
    
    # Calculate improvement
    if [ $original_time -gt 0 ]; then
        improvement=$((100 * (original_time - persistence_time) / original_time))
        speedup_factor=$(echo "scale=2; $original_time / $persistence_time" | bc -l)
    else
        improvement=0
        speedup_factor="N/A"
    fi
    
    echo -e "${PURPLE}📊 Benchmark Results${NC}"
    echo -e "${PURPLE}===================${NC}"
    echo -e "${BLUE}Original Pipeline:    ${original_time}s${NC}"
    echo -e "${GREEN}Persistence Pipeline: ${persistence_time}s${NC}"
    echo -e "${YELLOW}Improvement:          ${improvement}% faster${NC}"
    echo -e "${YELLOW}Speedup Factor:       ${speedup_factor}x${NC}"
}

# Main script logic
case "${1:-}" in
    "start")
        print_header
        check_requirements
        start_services
        ;;
    "stop")
        print_header
        stop_services
        ;;
    "restart")
        print_header
        restart_services
        ;;
    "status")
        print_header
        check_status
        ;;
    "run")
        print_header
        check_requirements
        run_pipeline "$2"
        ;;
    "benchmark")
        print_header
        check_requirements
        benchmark_comparison "$2"
        ;;
    "install")
        print_header
        check_requirements
        install_dependencies
        ;;
    "help" | "--help" | "-h" | "")
        print_header
        print_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        print_usage
        exit 1
        ;;
esac