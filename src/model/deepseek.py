"""
DeepSeek Model Architecture for Children's Stories
Implements advanced features:
- Multihead Latent Attention (MLA)
- Mixture of Experts (MoE)
- Multi-token prediction
- Quantization support
- Rotary Positional Encodings (RoPE)
- Optimized for children's story generation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek model optimized for children's stories"""
    vocab_size: int = 50257  # GPT-2 vocabulary size
    n_layer: int = 6         # Reduced for efficiency
    n_head: int = 8          # Number of attention heads
    n_embd: int = 512        # Embedding dimension
    block_size: int = 1024   # Context window
    dropout: float = 0.1     # Dropout rate
    bias: bool = True        # Use bias in linear layers
    
    # MLA (Multihead Latent Attention) config
    use_mla: bool = True     # Enable MLA
    mla_kv_heads: int = 4    # Number of key-value heads for MLA
    mla_q_proj_dim: int = 32  # Query projection dimension
    mla_kv_proj_dim: int = 16  # Key-value projection dimension
    
    # MoE (Mixture of Experts) config
    moe_num_experts: int = 4  # Number of experts
    moe_top_k: int = 2       # Number of experts per token
    moe_expert_capacity: float = 1.25
    moe_aux_loss_coeff: float = 0.01
    
    # Multi-token prediction
    multi_token_predict: int = 2  # Predict next 2 tokens for children's stories
    
    # Quantization
    use_quantization: bool = False
    quantization_bits: int = 8


class RoPEPositionalEncoding(nn.Module):
    """Rotary Positional Encoding (RoPE) for better position understanding"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _compute_cos_sin(self, seq_len: int, device: torch.device):
        """Compute cosine and sine values for given sequence length"""
        if seq_len > self._cached_seq_len or self._cached_cos is None:
            # Create position indices
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            
            # Compute frequencies
            freqs = torch.outer(t, self.inv_freq)
            
            # Create rotation matrix components
            cos_vals = torch.cos(freqs)
            sin_vals = torch.sin(freqs)
            
            # Cache results
            self._cached_cos = cos_vals
            self._cached_sin = sin_vals
            self._cached_seq_len = seq_len
        
        return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
    
    def apply_rope(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        """Apply RoPE to input tensor"""
        batch_size, seq_len, n_heads, head_dim = x.shape
        
        # Get cos/sin values
        cos, sin = self._compute_cos_sin(seq_len, x.device)
        
        # Handle position_ids if provided
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        
        # Reshape for broadcasting
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(2)
        
        # Split x into two halves
        x1 = x[..., ::2]  # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        # Apply rotation
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Recombine
        rotated_x = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
        
        return rotated_x


class MultiheadLatentAttention(nn.Module):
    """
    Multihead Latent Attention (MLA) - DeepSeek's efficient attention mechanism
    Uses shared key-value heads with projection decomposition for efficiency
    """
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.kv_heads = config.mla_kv_heads
        self.kv_head_dim = self.head_dim
        
        # Query projection with decomposition
        self.q_a_proj = nn.Linear(config.n_embd, config.mla_q_proj_dim, bias=False)
        self.q_b_proj = nn.Linear(config.mla_q_proj_dim, config.n_embd, bias=False)
        
        # Key-Value projection with shared heads
        self.kv_a_proj = nn.Linear(config.n_embd, config.mla_kv_proj_dim, bias=False)
        self.kv_b_proj = nn.Linear(config.mla_kv_proj_dim, self.kv_heads * self.head_dim * 2, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # RoPE for positional encoding
        self.rope = RoPEPositionalEncoding(self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Scaling factor
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape
        
        # Query projection through decomposition
        q_latent = self.q_a_proj(x)  # [B, T, rank]
        q = self.q_b_proj(q_latent)  # [B, T, n_embd]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim)
        
        # Key-Value projection through shared heads
        kv_latent = self.kv_a_proj(x)  # [B, T, kv_rank]
        kv = self.kv_b_proj(kv_latent)  # [B, T, kv_heads * kv_head_dim * 2]
        kv = kv.view(batch_size, seq_len, self.kv_heads, self.head_dim, 2)
        k, v = kv.unbind(dim=-1)  # Each: [B, T, kv_heads, kv_head_dim]
        
        # Apply RoPE to queries and keys before expansion
        q = self.rope.apply_rope(q)
        k = self.rope.apply_rope(k)
        
        # Expand key-value to match query heads
        k = k.repeat_interleave(self.n_head // self.kv_heads, dim=2)
        v = v.repeat_interleave(self.n_head // self.kv_heads, dim=2)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [B, n_head, T, head_dim]
        k = k.transpose(1, 2)  # [B, n_head, T, head_dim]
        v = v.transpose(1, 2)  # [B, n_head, T, head_dim]
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is None:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            attn_scores.masked_fill_(causal_mask, float('-inf'))
        else:
            attn_scores = attn_scores + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, n_head, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class MoEExpert(nn.Module):
    """Expert network for Mixture of Experts"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class MixtureOfExperts(nn.Module):
    """Mixture of Experts (MoE) for increased model capacity"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.expert_capacity = config.moe_expert_capacity
        
        # Router
        self.router = nn.Linear(config.n_embd, config.moe_num_experts, bias=False)
        
        # Experts
        self.experts = nn.ModuleList([MoEExpert(config) for _ in range(config.moe_num_experts)])
        
        # Layer norm
        self.ln = nn.LayerNorm(config.n_embd, bias=config.bias)
    
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get router logits
        router_logits = self.router(x)  # [B, T, num_experts]
        
        # Get top-k experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [B, T]
            
            if expert_mask.any():
                # Get tokens for this expert
                expert_tokens = x[expert_mask]  # [num_tokens, hidden_dim]
                
                # Get routing weights for this expert
                expert_weights = top_k_probs[expert_mask]  # [num_tokens, top_k]
                expert_weights = expert_weights[top_k_indices[expert_mask] == expert_idx]  # [num_tokens]
                
                # Apply expert
                expert_output = self.experts[expert_idx](expert_tokens)  # [num_tokens, hidden_dim]
                
                # Weight the output
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                
                # Add to output
                output[expert_mask] += weighted_output
        
        # Apply layer norm
        output = self.ln(output)
        
        return output, router_logits
    
    def _compute_aux_loss(self, router_logits: torch.Tensor):
        """Compute auxiliary loss for load balancing"""
        router_probs = F.softmax(router_logits, dim=-1)
        mean_expert_usage = router_probs.mean(dim=[0, 1])  # [num_experts]
        target_usage = 1.0 / self.num_experts
        
        aux_loss = torch.sum((mean_expert_usage - target_usage) ** 2)
        return aux_loss


class DeepSeekBlock(nn.Module):
    """DeepSeek transformer block with MLA and MoE"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        
        # Layer norms
        self.ln1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # Attention - use MLA if enabled, otherwise use standard attention
        if config.use_mla:
            self.attn = MultiheadLatentAttention(config)
        else:
            # Standard multihead attention as fallback
            self.attn = nn.MultiheadAttention(
                config.n_embd, 
                config.n_head, 
                dropout=config.dropout,
                bias=config.bias,
                batch_first=True
            )
        
        # MoE
        self.moe = MixtureOfExperts(config)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Attention with residual connection
        if self.config.use_mla:
            x = x + self.attn(self.ln1(x), attention_mask)
        else:
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attention_mask)
            x = x + attn_out
        
        # MoE with residual connection
        moe_output, router_logits = self.moe(self.ln2(x))
        x = x + moe_output
        
        return x, router_logits


class MultiTokenPredictor(nn.Module):
    """Multi-token prediction head for improved training efficiency"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config
        self.num_tokens = config.multi_token_predict
        
        # Separate prediction heads for each future token
        self.predictors = nn.ModuleList([
            nn.Linear(config.n_embd, config.vocab_size, bias=False)
            for _ in range(config.multi_token_predict)
        ])
    
    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for multi-token prediction"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Predict multiple future tokens
        logits = []
        for i, predictor in enumerate(self.predictors):
            # Use hidden states shifted by i+1 positions
            if i + 1 < seq_len:
                token_logits = predictor(hidden_states[:, i+1:i+2, :])  # [B, 1, vocab_size]
                logits.append(token_logits)
            else:
                # Pad with zeros if not enough sequence length
                token_logits = torch.zeros(batch_size, 1, self.config.vocab_size, 
                                         device=hidden_states.device)
                logits.append(token_logits)
        
        return torch.cat(logits, dim=1)  # [B, num_tokens, vocab_size]


class DeepSeek(nn.Module):
    """DeepSeek model for children's story generation"""
    
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        assert isinstance(config, DeepSeekConfig), "config must be an instance of DeepSeekConfig"
        self.config = config
        
        # Token and position embeddings
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([DeepSeekBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        
        # Language model head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Multi-token predictor
        if config.multi_token_predict > 0:
            self.multi_token_predictor = MultiTokenPredictor(config)
        else:
            self.multi_token_predictor = None
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Setup quantization if enabled
        if config.use_quantization:
            self._setup_quantization()
    
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def _setup_quantization(self):
        """Setup quantization for the model"""
        # This would implement quantization logic
        # For now, just a placeholder
        pass
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass"""
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.block_size
        
        # Position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        
        # Token and position embeddings
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        router_logits_list = []
        for block in self.transformer.h:
            x, router_logits = block(x)
            router_logits_list.append(router_logits)
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # Training mode
            if self.multi_token_predictor is not None:
                # Multi-token prediction
                multi_logits = self.multi_token_predictor(x)
                loss = self._compute_multi_token_loss(multi_logits, targets)
            else:
                # Standard single-token prediction
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                     targets.view(-1), ignore_index=-100)
            
            # Add MoE auxiliary loss
            if router_logits_list:
                aux_loss = sum(self.transformer.h[i].moe._compute_aux_loss(router_logits_list[i])
                              for i in range(len(router_logits_list)))
                loss += self.config.moe_aux_loss_coeff * aux_loss
            
            return logits if self.multi_token_predictor is None else multi_logits, loss
        else:
            # Inference mode
            logits = self.lm_head(x[:, [-1], :])
            return logits, None
    
    def _compute_multi_token_loss(self, logits: torch.Tensor, targets: torch.Tensor):
        """Compute loss for multi-token prediction"""
        batch_size, num_tokens, vocab_size = logits.shape
        
        # Prepare targets for multi-token prediction
        # For multi-token prediction, we need targets shifted by 1, 2, ..., num_tokens positions
        multi_targets = []
        for i in range(num_tokens):
            if i + 1 < targets.size(1):
                # Take targets shifted by (i+1) positions
                shifted_targets = targets[:, i+1:i+2]  # [batch_size, 1]
                multi_targets.append(shifted_targets)
            else:
                # Pad with -100 (ignore_index) if not enough sequence length
                pad_targets = torch.full((batch_size, 1), -100, device=targets.device, dtype=targets.dtype)
                multi_targets.append(pad_targets)
        
        # Concatenate to get [batch_size, num_tokens]
        multi_targets = torch.cat(multi_targets, dim=1)
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = multi_targets.view(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
        
        return loss
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100, 
                 temperature: float = 1.0, top_k: Optional[int] = None):
        """Generate text using the model"""
        for _ in range(max_new_tokens):
            # Ensure input doesn't exceed block size
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        
        return input_ids
    
    @classmethod
    def from_pretrained(cls, model_type: str, override_args: Optional[dict] = None):
        """Load a pretrained model"""
        # This would implement loading from pretrained weights
        # For now, return a default configuration
        config = DeepSeekConfig()
        if override_args:
            for key, value in override_args.items():
                setattr(config, key, value)
        return cls(config) 