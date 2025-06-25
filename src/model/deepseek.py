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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from contextlib import nullcontext


class MOEManager:
    """
    Basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """
    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)
    
    def aggregate_aux_loss(self):
        return sum(self.aux_loss)
    
    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)


MANAGER = MOEManager()


class Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Router settings - map from your config
        self.top_k = config.moe_top_k
        self.n_exp = config.moe_num_experts
        assert self.top_k >= 1 and self.top_k <= self.n_exp
        
        # Optional features (add these to your config if needed)
        self.use_noisy_top_k = getattr(config, 'moe_use_noisy_top_k', False)
        self.train_capacity = getattr(config, 'moe_train_capacity', 1.25)
        self.eval_capacity = getattr(config, 'moe_eval_capacity', 2.0)
        self.min_capacity = getattr(config, 'moe_min_capacity', 4)
        self.router_use_full_prec = getattr(config, 'moe_router_use_full_prec', True)
        
        # Auxiliary loss settings
        self.use_aux_loss = getattr(config, 'moe_use_aux_loss', True)
        self.use_router_z_loss = getattr(config, 'moe_use_router_z_loss', True)
        
        # Linear projection for (noisy) softmax gating
        self.w_g = nn.Linear(config.n_embd, self.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, self.n_exp, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        # Optionally run router in full precision for stability
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)
        
        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T
            
            # Router logits
            logits = self.w_g(x)  # [B, T, n_exp]
            
            # Router z loss
            if self.use_router_z_loss and self.training:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)
            
            # Find top k experts
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [B, T, k]
            
            # Normalize expert probabilities over top-k only
            router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)
            
            # Auxiliary load balancing loss
            if self.use_aux_loss and self.training:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)
            
            # Compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)
            
            # Multi-hot mask of chosen experts
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2)  # [k, B * T, n_exp]
            
            # Compute cumulative sum for token ranking within experts
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]
            
            # Mask out entries beyond capacity
            exp_mask *= torch.lt(exp_rank, exp_capacity)  # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1))  # [n_exp]
            
            # Get position of each token in its expert's batch
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]
            
            # Mask probabilities to only include selected experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]  # [1, B * T, n_exp]
            exp_weights = exp_mask * router_probs  # [k, B * T, n_exp]
            
            # Convert rank to one-hot over capacity
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity)  # [k, B * T, exp_capacity]
            
            # Create weight matrix for combine operation
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool()
            
            return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """Switch Transformer auxiliary loss for load balancing"""
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))
        
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """ST-MoE router z loss to prevent logit explosion"""
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0
        return torch.mean(z_loss)
    
    def get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2  # Make even
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return int(capacity)


class MLPExperts(nn.Module):
    """
    Batched MLP experts - processes all experts in parallel using bmm
    """
    def __init__(self, config):
        super().__init__()
        self.bias = config.bias
        self.n_exp = config.moe_num_experts
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Initialize expert weights
        self.c_fc = nn.Parameter(torch.empty(self.n_exp, self.n_embd, 4 * self.n_embd))
        self.c_proj = nn.Parameter(torch.empty(self.n_exp, 4 * self.n_embd, self.n_embd))
        
        if self.bias:
            self.fc_bias = nn.Parameter(torch.empty(self.n_exp, 1, 4 * self.n_embd))
            self.proj_bias = nn.Parameter(torch.empty(self.n_exp, 1, self.n_embd))
        else:
            self.fc_bias = None
            self.proj_bias = None
        
        self.gelu = nn.GELU()
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Standard initialization for linear layers
        nn.init.normal_(self.c_fc, mean=0.0, std=0.02)
        nn.init.normal_(self.c_proj, mean=0.0, std=0.02)
    
    def forward(self, x):
        # x shape: [n_exp, exp_capacity, n_embd]
        x = torch.bmm(x, self.c_fc)
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        x = self.dropout_layer(x)
        return x


class MixtureOfExperts(nn.Module):
    """
    Efficient Mixture of Experts implementation with batched processing
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = Router(config)
        self.experts = MLPExperts(config)
        
        # Layer norm after MoE (as in your original)
        self.ln = nn.LayerNorm(config.n_embd, bias=config.bias)
    
    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size()
        num_tokens = B * T
        
        # Pass through router
        used_capacity, exp_weight, exp_mask = self.router(x)
        
        # Flatten input
        x_flat = x.view(num_tokens, n_embd)
        
        # Reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] @ [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x_flat) @ x_flat
        
        # Compute expert outputs
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, n_embd]
        
        # Aggregate expert outputs based on router weights
        exp_weight = exp_weight.view(num_tokens, -1)  # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd)  # [n_exp * exp_capacity, n_embd]
        output = exp_weight @ exp_out  # [B * T, n_embd]
        
        # Reshape and apply layer norm
        output = output.view(B, T, n_embd)
        output = self.ln(output)
        
        # Return output and router logits (for compatibility with your DeepSeekBlock)
        # Note: router logits are already used for aux losses inside the router
        return output, None  # Second return value for compatibility


class DeepSeekBlock(nn.Module):
    """DeepSeek transformer block with MLA and MoE"""
    
    def __init__(self, config):
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
        
        # MoE with efficient implementation
        self.moe = MixtureOfExperts(config)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Attention with residual connection
        if self.config.use_mla:
            x = x + self.attn(self.ln1(x), attention_mask)
        else:
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attention_mask)
            x = x + attn_out
        
        # MoE with residual connection
        moe_output, _ = self.moe(self.ln2(x))
        x = x + moe_output
        
        return x, None  # Second return for compatibility

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
    
    def _compute_multi_token_loss(self, multi_logits, targets):
        """Compute loss for multi-token prediction"""
        # Placeholder - implement based on your MultiTokenPredictor output format
        loss = 0
        for i, logits in enumerate(multi_logits):
            # Shift targets appropriately for each prediction head
            shifted_targets = targets[:, i+1:i+1+logits.size(1)]
            if shifted_targets.size(1) > 0:
                loss += F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    shifted_targets.reshape(-1),
                    ignore_index=-100
                )
        return loss / len(multi_logits)
    
    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """Forward pass"""
        device = input_ids.device
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.block_size
        
        # Reset auxiliary losses at the start of forward pass
        if self.training:
            MANAGER.reset_aux_loss()
            MANAGER.reset_router_z_loss()
        
        # Position indices
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)
        
        # Token and position embeddings
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # Forward through transformer blocks
        for block in self.transformer.h:
            x, _ = block(x)  # Router logits are now handled internally
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            # Training mode
            if self.multi_token_predictor is not None:
                # Multi-token prediction
                multi_logits = self.multi_token_predictor(x)
                loss = self._compute_multi_token_loss(multi_logits, targets)
                logits = multi_logits  # For compatibility
            else:
                # Standard single-token prediction
                logits = self.lm_head(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    targets.view(-1), 
                    ignore_index=-100
                )
            
            # Add MoE auxiliary losses from MANAGER
            if self.training:
                # Load balancing loss
                aux_loss = MANAGER.aggregate_aux_loss()
                if aux_loss > 0:
                    loss += self.config.moe_aux_loss_coeff * aux_loss
                
                # Router z-loss
                router_z_loss = MANAGER.aggregate_router_z_loss()
                if router_z_loss > 0:
                    loss += self.config.moe_router_z_loss_coeff * router_z_loss
            
            return logits, loss
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