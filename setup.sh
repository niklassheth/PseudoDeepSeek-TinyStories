#!/bin/bash

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
LORA_CHECKPOINT_DIR="${LORA_CHECKPOINT_DIR:-${PROJECT_ROOT}/lora_checkpoints}"
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
            "${LORA_CHECKPOINT_DIR}" || handle_error "Failed to create directories"
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

# Function to prepare dataset
prepare_dataset() {
    print_status "Preparing dataset..."
    cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
    
    # Create a Python script to process the data
    cat > process_data.py << 'EOF'
import os
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.data_processor import DeepSeekDataProcessor

def main():
    print("[+] Processing dataset into binary files...")
    processor = DeepSeekDataProcessor()
    processor.prepare_dataset()
    print("[+] Data processing completed successfully!")

if __name__ == "__main__":
    main()
EOF

    # Run the data processing script
    python3 process_data.py || handle_error "Failed to process dataset"
    
    # Verify the files were created
    if [ ! -f "${PROJECT_ROOT}/src/data/train.bin" ] || [ ! -f "${PROJECT_ROOT}/src/data/validation.bin" ]; then
        handle_error "Data processing failed - required files not created"
    fi
}

# Function to train base model
train_base_model() {
    print_status "Starting DeepSeek base model training..."
    cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
    
    python3 src/run_training.py \
        --batch-size "${BATCH_SIZE:-12}" \
        --max-iters "${MAX_ITERS:-20000}" \
        --eval-interval "${EVAL_INTERVAL:-1000}" \
        --eval-iters "${EVAL_ITERS:-200}" \
        --learning-rate "${LEARNING_RATE:-6e-4}" \
        --weight-decay "${WEIGHT_DECAY:-0.1}" \
        --warmup-iters "${WARMUP_ITERS:-2000}" \
        --lr-decay-iters "${LR_DECAY_ITERS:-20000}" \
        --min-lr "${MIN_LR:-6e-5}" \
        --moe-experts "${MOE_EXPERTS:-4}" \
        --multi-token "${MULTI_TOKEN:-2}" || handle_error "Base model training failed"
}

# Function to perform LoRA finetuning
finetune_lora() {
    while true; do
        read -p "Do you want to perform LoRA finetuning? (y/n) " do_finetune
        case $do_finetune in
            [Yy]* )
                print_status "Starting LoRA finetuning..."
                cd "${PROJECT_ROOT}" || handle_error "Failed to change to project directory"
                
                # Create LoRA finetuning script
                cat > finetune_lora.py << 'EOF'
import torch
import os
import sys
sys.path.append('src')

from model.deepseek import DeepSeek, DeepSeekConfig
from peft import get_peft_model, LoraConfig, TaskType

def main():
    print("Loading base model...")
    checkpoint = torch.load('checkpoints/best_model.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepSeek(checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj"]
    )
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print("LoRA finetuning setup complete!")

if __name__ == "__main__":
    main()
EOF
                
                python3 finetune_lora.py || handle_error "LoRA finetuning failed"
                break
                ;;
            [Nn]* )
                print_status "Skipping LoRA finetuning..."
                break
                ;;
            * )
                echo "Please answer 'y' or 'n'"
                ;;
        esac
    done
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
    print_info "3. Generate stories: python src/generate.py --prompt 'your prompt'"
    print_info "4. Interactive mode: python src/generate.py --interactive"
    print_info ""
    print_info "Model files:"
    print_info "- Base model: checkpoints/best_model.pt"
    print_info "- LoRA model: lora_checkpoints/best_lora_model.pt"
    print_info ""
    print_info "Configuration options:"
    print_info "- Adjust model size: --n-layer, --n-head, --n-embd"
    print_info "- Training parameters: --batch-size, --learning-rate, --max-iters"
    print_info "- Advanced features: --moe-experts, --multi-token"
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
    
    # Prepare dataset
    prepare_dataset
    
    # Train base model
    train_base_model
    
    # Optional LoRA finetuning
    finetune_lora
    
    # Optional model testing
    test_model
    
    # Show usage information
    show_usage
    
    print_status "Setup completed successfully!"
}

# Run main function
main "$@" 