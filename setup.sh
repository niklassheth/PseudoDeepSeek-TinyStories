#!/bin/bash

# DeepSeek Children's Stories Model Setup Script
# Updated for new TinyStories dataset and modern DataLoader system
# No dataset preprocessing needed - downloads TinyStories automatically

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
VENV_PATH="${VENV_PATH:-${PROJECT_ROOT}/venv}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}"
REQUIRED_SPACE_MB="${REQUIRED_SPACE_MB:-2000}"

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_info() {
    echo -e "${BLUE}[i] $1${NC}"
}

# Function to handle errors
handle_error() {
    print_error "$1"
    exit 1
}

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check disk space
check_disk_space() {
    local available_space_mb=$(df -m . | awk 'NR==2 {print $4}')
    if [ "$available_space_mb" -lt "$REQUIRED_SPACE_MB" ]; then
        print_warning "Low disk space. Only ${available_space_mb}MB available, ${REQUIRED_SPACE_MB}MB required."
        return 1
    fi
    return 0
}

# Function to check GPU memory
check_gpu_memory() {
    if command_exists nvidia-smi; then
        local total_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
        local free_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
        local used_memory=$((total_memory - free_memory))
        print_status "GPU Memory: ${used_memory}MB used, ${free_memory}MB free of ${total_memory}MB total"
        
        # Check if we have enough memory for training
        if [ "$free_memory" -lt 4000 ]; then
            print_warning "Low GPU memory. Consider reducing batch size or model size."
        fi
    else
        print_warning "nvidia-smi not found. GPU training may not be available."
    fi
}

# Function to create project structure
create_project_structure() {
    print_status "Creating project structure..."
    mkdir -p "${PROJECT_ROOT}/src/data" \
            "${PROJECT_ROOT}/src/model" \
            "${PROJECT_ROOT}/src/training" \
            "${PROJECT_ROOT}/src/inference" \
            "${CHECKPOINT_DIR}" \
|| handle_error "Failed to create directories"
}

# Function to setup virtual environment
setup_virtual_env() {
    print_status "Creating virtual environment..."
    python3 -m venv "${VENV_PATH}" || handle_error "Failed to create virtual environment"
    source "${VENV_PATH}/bin/activate" || handle_error "Failed to activate virtual environment"
    
    print_status "Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt || handle_error "Failed to install requirements"
}

# Function to test dataloader
test_dataloader() {
    print_status "Testing TinyStories dataloader..."
    cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
    
    # Test the new dataloader system
    python3 -c "
import sys
sys.path.append('src')
from data.dataloader import create_dataloaders
print('[+] Testing TinyStories dataloader...')
try:
    dataloaders = create_dataloaders(batch_size=4, num_workers=1)
    print('[+] DataLoader test successful!')
    print('[+] TinyStories dataset will be downloaded automatically during training.')
except Exception as e:
    print(f'[-] DataLoader test failed: {e}')
    exit(1)
" || handle_error "DataLoader test failed"
}

# Function to train base model
train_base_model() {
    print_status "Starting DeepSeek base model training..."
    cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
    
    # The new training system uses src/config.py for configuration
    # No command line arguments needed - just run the training script
    python3 src/run_training.py || handle_error "Base model training failed"
}


# Function to test the trained model
test_model() {
    while true; do
        read -p "Do you want to test the trained model? (y/n) " do_test
        case $do_test in
            [Yy]* )
                print_status "Testing the trained model..."
                cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
                
                # Create test prompts
                prompts=(
                    "Once upon a time"
                    "In a magical forest"
                    "The little robot"
                    "The brave knight"
                )
                
                # Test each prompt
                for prompt in "${prompts[@]}"; do
                    print_status "Testing with prompt: '$prompt'"
                    python3 src/generate.py \
                        --model-path "${CHECKPOINT_DIR}/best_model.pt" \
                        --prompt "$prompt" \
                        --max-tokens 100 \
                        --temperature 0.8 \
                        --top-k 40
                    echo
                done
                break
                ;;
            [Nn]* )
                print_status "Skipping model testing..."
                break
                ;;
            * )
                echo "Please answer 'y' or 'n'"
                ;;
        esac
    done
}

# Function to show usage information
show_usage() {
    print_info "DeepSeek Children's Stories Model Setup Complete!"
    print_info ""
    print_info "Next steps:"
    print_info "1. Activate virtual environment: source venv/bin/activate"
    print_info "2. Train the model: python src/run_training.py"
    print_info "3. Test dataloader: python src/data/dataloader.py"
    print_info "4. Generate stories: python src/generate.py --prompt 'your prompt'"
    print_info ""
    print_info "Model files:"
    print_info "- Best model: checkpoints/best_model.pt"
    print_info "- Epoch checkpoints: checkpoints/checkpoint_epoch_*.pt"
    print_info ""
    print_info "Configuration:"
    print_info "- Edit src/config.py to change model/training parameters"
    print_info "- Uses TinyStories dataset (downloads automatically)"
    print_info "- Modern epoch-based training with DataLoader"
    print_info "- Memory-efficient streaming mode"
}

# Main setup function
main() {
    print_info "DeepSeek Children's Stories Model Setup"
    print_info "======================================"
    
    # Check prerequisites
    if ! command_exists python3; then
        handle_error "Python 3 is required but not installed"
    fi
    
    if ! command_exists pip; then
        handle_error "pip is required but not installed"
    fi
    
    # Check disk space
    if ! check_disk_space; then
        print_warning "Continuing with low disk space..."
    fi
    
    # Check GPU
    check_gpu_memory
    
    # Create project structure
    create_project_structure
    
    # Setup virtual environment
    setup_virtual_env
    
    # Skip dataloader test - no dataset preparation needed
    
    # Train base model
    train_base_model
    
    # Optional model testing
    test_model
    
    # Show usage information
    show_usage
    
    print_status "Setup completed successfully!"
}

# Run main function
main "$@" 