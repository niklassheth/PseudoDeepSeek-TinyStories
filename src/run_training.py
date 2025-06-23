"""
DeepSeek Children's Stories Training Script
Main training script for the DeepSeek model on children's stories
"""

import os
import sys
import torch

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.deepseek import DeepSeek, DeepSeekConfig
from training.trainer import DeepSeekTrainer, create_deepseek_trainer
from data.data_processor import DeepSeekDataProcessor
from config import get_model_config, get_training_config


def create_model_config(model_config) -> DeepSeekConfig:
    """Create DeepSeekConfig from ModelConfig"""
    return DeepSeekConfig(
        vocab_size=model_config.vocab_size,
        n_layer=model_config.n_layer,
        n_head=model_config.n_head,
        n_embd=model_config.n_embd,
        block_size=model_config.block_size,
        dropout=model_config.dropout,
        bias=model_config.bias,
        use_mla=model_config.use_mla,
        mla_kv_heads=model_config.mla_kv_heads,
        mla_q_lora_rank=model_config.mla_q_lora_rank,
        mla_kv_lora_rank=model_config.mla_kv_lora_rank,
        moe_num_experts=model_config.moe_num_experts,
        moe_top_k=model_config.moe_top_k,
        moe_expert_capacity=model_config.moe_expert_capacity,
        moe_aux_loss_coeff=model_config.moe_aux_loss_coeff,
        multi_token_predict=model_config.multi_token_predict,
        use_quantization=model_config.use_quantization,
        quantization_bits=model_config.quantization_bits
    )


def setup_device_and_precision(training_config):
    """Setup device and mixed precision"""
    # Device configuration
    if training_config.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = training_config.device
    
    # Mixed precision only supported on CUDA
    use_mixed_precision = training_config.use_mixed_precision and device == 'cuda'
    
    print(f"Using device: {device}")
    print(f"Mixed precision: {use_mixed_precision}")
    
    return device, use_mixed_precision


def create_model(model_config, training_config, device):
    """Create and setup the model"""
    config = create_model_config(model_config)
    model = DeepSeek(config)
    model = model.to(device)
    
    # Compile model if requested and supported
    if training_config.compile_model and hasattr(torch, 'compile'):
        print("Compiling model...")
        model = torch.compile(model)
    
    # Calculate model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    return model, config


def create_optimizer(model, training_config):
    """Create optimizer"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )
    return optimizer


def train_model(model, model_config, training_config, device, use_mixed_precision):
    """Train the model"""
    try:
        print(f"Starting training...")
        print(f"Model: {model_config.n_layer} layers, {model_config.n_head} heads, {model_config.n_embd} dims")
        print(f"MLA: {model_config.mla_kv_heads} KV heads, MoE: {model_config.moe_num_experts} experts")
        print(f"Multi-token prediction: {model_config.multi_token_predict} tokens")
        
        # Create optimizer
        optimizer = create_optimizer(model, training_config)
        
        # Initialize trainer
        trainer = DeepSeekTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            batch_size=training_config.batch_size,
            max_iters=training_config.max_iters,
            eval_interval=training_config.eval_interval,
            eval_iters=training_config.eval_iters,
            learning_rate=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
            warmup_iters=training_config.warmup_iters,
            lr_decay_iters=training_config.lr_decay_iters,
            min_lr=training_config.min_lr,
            checkpoint_dir=training_config.checkpoint_dir,
            use_mixed_precision=use_mixed_precision
        )
        
        print("=" * 50)
        print(f"Training Configuration:")
        print(f"  - Batch size: {training_config.batch_size}")
        print(f"  - Max iterations: {training_config.max_iters}")
        print(f"  - Learning rate: {training_config.learning_rate}")
        print(f"  - Weight decay: {training_config.weight_decay}")
        print(f"  - Warmup iterations: {training_config.warmup_iters}")
        print(f"  - LR decay iterations: {training_config.lr_decay_iters}")
        print(f"  - Min learning rate: {training_config.min_lr}")
        print("=" * 50)
        
        # Start training
        trainer.train()
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"[-] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main training function"""
    print("DeepSeek Children's Stories Training")
    print("=" * 50)
    
    # Load configurations
    model_config = get_model_config()
    training_config = get_training_config()
    
    print(f"Model Config: {model_config.n_layer}L/{model_config.n_head}H/{model_config.n_embd}D")
    print(f"Training Config: Batch size {training_config.batch_size}, {training_config.max_iters} iters")
    
    # Set random seed for reproducibility
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.seed)
    
    # Setup device and precision
    device, use_mixed_precision = setup_device_and_precision(training_config)
    
    # Prepare dataset
    print("\nPreparing dataset...")
    processor = DeepSeekDataProcessor()
    processor.prepare_dataset()
    
    # Create model
    print("\nCreating model...")
    model, deepseek_config = create_model(model_config, training_config, device)
    
    # Train model
    print("\nStarting training...")
    train_model(model, model_config, training_config, device, use_mixed_precision)


if __name__ == "__main__":
    main()