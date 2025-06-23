"""
DeepSeek Children's Stories Training Script
Main training script for the DeepSeek model on children's stories
"""

import os
import sys
import argparse
import torch
from dataclasses import dataclass
from typing import Optional

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.deepseek import DeepSeek, DeepSeekConfig
from training.trainer import DeepSeekTrainer, create_deepseek_trainer
from data.data_processor import DeepSeekDataProcessor


@dataclass
class TrainingConfig:
    """Configuration for DeepSeek training"""
    # Model configuration
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    
    # MLA configuration
    use_mla: bool = True
    mla_kv_heads: int = 4
    mla_q_lora_rank: int = 32
    mla_kv_lora_rank: int = 16
    
    # MoE configuration
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_expert_capacity: float = 1.25
    moe_aux_loss_coeff: float = 0.01
    
    # Multi-token prediction
    multi_token_predict: int = 2  # Predict next 2 tokens for efficiency
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 8
    
    # Training configuration
    batch_size: int = 32
    max_iters: int = 20000
    eval_interval: int = 1000
    eval_iters: int = 200
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    warmup_iters: int = 2000
    lr_decay_iters: int = 20000
    min_lr: float = 6e-5
    
    # System configuration
    checkpoint_dir: str = 'checkpoints'
    use_mixed_precision: bool = True
    compile_model: bool = True
    
    # Data configuration
    dataset_name: str = "ajibawa-2023/Children-Stories-Collection"
    data_dir: str = 'src/data'


def setup_environment():
    """Setup the training environment"""
    print("Setting up DeepSeek Children's Stories training environment...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('lora_checkpoints', exist_ok=True)
    os.makedirs('src/data', exist_ok=True)
    
    print("Environment setup complete!")


def prepare_data():
    """Prepare the dataset for training"""
    print("Preparing dataset...")
    
    processor = DeepSeekDataProcessor()
    data_files = processor.prepare_dataset()
    
    print("Dataset preparation complete!")
    return data_files


def create_model(config: TrainingConfig) -> DeepSeek:
    """Create the DeepSeek model"""
    print("Creating DeepSeek model...")
    
    # Create model configuration
    model_config = DeepSeekConfig(
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias,
        use_mla=config.use_mla,
        mla_kv_heads=config.mla_kv_heads,
        mla_q_lora_rank=config.mla_q_lora_rank,
        mla_kv_lora_rank=config.mla_kv_lora_rank,
        moe_num_experts=config.moe_num_experts,
        moe_top_k=config.moe_top_k,
        moe_expert_capacity=config.moe_expert_capacity,
        moe_aux_loss_coeff=config.moe_aux_loss_coeff,
        multi_token_predict=config.multi_token_predict,
        use_quantization=config.use_quantization,
        quantization_bits=config.quantization_bits
    )
    
    # Create model
    model = DeepSeek(model_config)
    
    # Compile model if requested
    if config.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model configuration:")
    print(f"  - Layers: {config.n_layer}")
    print(f"  - Heads: {config.n_head}")
    print(f"  - Embedding dim: {config.n_embd}")
    print(f"  - MLA enabled: {config.use_mla}")
    print(f"  - MLA KV heads: {config.mla_kv_heads}")
    print(f"  - MoE experts: {config.moe_num_experts}")
    print(f"  - Multi-token prediction: {config.multi_token_predict}")
    
    return model


def train_model(model: DeepSeek, config: TrainingConfig):
    """Train the DeepSeek model"""
    print(f"[+] Starting training with config:")
    print(f"    - Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"    - Multi-token prediction: {config.multi_token_predict}")
    print(f"    - MoE experts: {config.moe_num_experts}")
    print(f"    - MLA enabled: {config.use_mla}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Initialize trainer with individual parameters
    trainer = DeepSeekTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        batch_size=config.batch_size,
        max_iters=config.max_iters,
        eval_interval=config.eval_interval,
        eval_iters=config.eval_iters,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_iters=config.warmup_iters,
        lr_decay_iters=config.lr_decay_iters,
        min_lr=config.min_lr,
        checkpoint_dir=config.checkpoint_dir,
        use_mixed_precision=config.use_mixed_precision
    )
    
    try:
        # Start training
        trainer.train()
        print("[+] Training completed successfully!")
        
        # Save final model
        final_model_path = os.path.join(config.checkpoint_dir, "final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'optimizer_state_dict': trainer.optimizer.state_dict(),
        }, final_model_path)
        print(f"[+] Final model saved to {final_model_path}")
        
    except Exception as e:
        print(f"[-] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train DeepSeek model on children\'s stories')
    
    # Model configuration
    parser.add_argument('--n-layer', type=int, default=6, help='Number of layers')
    parser.add_argument('--n-head', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n-embd', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--block-size', type=int, default=1024, help='Context window size')
    
    # Training configuration
    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--max-iters', type=int, default=20000, help='Maximum iterations')
    parser.add_argument('--learning-rate', type=float, default=6e-4, help='Learning rate')
    parser.add_argument('--eval-interval', type=int, default=1000, help='Evaluation interval')
    parser.add_argument('--eval-iters', type=int, default=200, help='Number of evaluation iterations')
    parser.add_argument('--weight-decay', type=float, default=0.1, help='Weight decay')
    parser.add_argument('--warmup-iters', type=int, default=2000, help='Warmup iterations')
    parser.add_argument('--lr-decay-iters', type=int, default=20000, help='Learning rate decay iterations')
    parser.add_argument('--min-lr', type=float, default=6e-5, help='Minimum learning rate')
    
    # Advanced features
    parser.add_argument('--moe-experts', type=int, default=4, help='Number of MoE experts')
    parser.add_argument('--multi-token', type=int, default=2, help='Multi-token prediction')
    parser.add_argument('--no-compile', action='store_true', help='Disable model compilation')
    parser.add_argument('--no-mixed-precision', action='store_true', help='Disable mixed precision')
    
    # Resume training
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        weight_decay=args.weight_decay,
        warmup_iters=args.warmup_iters,
        lr_decay_iters=args.lr_decay_iters,
        min_lr=args.min_lr,
        moe_num_experts=args.moe_experts,
        multi_token_predict=args.multi_token,
        compile_model=not args.no_compile,
        use_mixed_precision=not args.no_mixed_precision
    )
    
    print("DeepSeek Children's Stories Training")
    print("=" * 50)
    print(f"Configuration:")
    print(f"  - Model: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
    print(f"  - MoE: {config.moe_num_experts} experts")
    print(f"  - Multi-token: {config.multi_token_predict}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Max iterations: {config.max_iters}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Weight decay: {config.weight_decay}")
    print(f"  - Warmup iterations: {config.warmup_iters}")
    print(f"  - LR decay iterations: {config.lr_decay_iters}")
    print(f"  - Min learning rate: {config.min_lr}")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Prepare data
    data_files = prepare_data()
    
    # Create model
    model = create_model(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print("Checkpoint loaded successfully!")
    
    # Train model
    train_model(model, config)
    
    print("Training completed successfully!")
    print("Best model saved to: checkpoints/best_model.pt")


if __name__ == "__main__":
    main() 