"""
Configuration file for DeepSeek Children's Stories Training
All model and training parameters in one place
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Basic model parameters
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    block_size: int = 1024
    dropout: float = 0.1
    bias: bool = True
    
    # MLA (Multihead Latent Attention) config
    use_mla: bool = True
    mla_kv_heads: int = 4
    mla_q_lora_rank: int = 32
    mla_kv_lora_rank: int = 16
    
    # MoE (Mixture of Experts) config
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_expert_capacity: float = 1.25
    moe_aux_loss_coeff: float = 0.01
    
    # Multi-token prediction
    multi_token_predict: int = 2
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 8


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Training parameters
    batch_size: int = 32  # Set to your desired batch size
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
    device: str = 'auto'  # 'auto', 'cuda', or 'cpu'
    use_mixed_precision: bool = True
    compile_model: bool = True
    seed: int = 42


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()


def get_model_config() -> ModelConfig:
    """Get the model configuration"""
    return DEFAULT_MODEL_CONFIG


def get_training_config() -> TrainingConfig:
    """Get the training configuration"""
    return DEFAULT_TRAINING_CONFIG