"""
DeepSeek Trainer for Children's Stories
Advanced training with MLA, MoE, and multi-token prediction
"""

import torch
import numpy as np
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import matplotlib.pyplot as plt
import os
import datetime
import time
import shutil
import psutil
import math
import gc
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from typing import Dict, List, Optional, Tuple

class DeepSeekTrainer:
    def __init__(self, model, optimizer, device, batch_size, max_iters, eval_interval, 
                 eval_iters, learning_rate, weight_decay, warmup_iters, lr_decay_iters, 
                 min_lr, checkpoint_dir='checkpoints', use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.checkpoint_dir = checkpoint_dir
        self.use_mixed_precision = use_mixed_precision
        self.best_loss = float('inf')
        
        # Training state
        self.current_iter = 0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize gradient scaler for mixed precision training
        if use_mixed_precision and device == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Initialize training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'grad_norm': [],
            'memory_usage': [],
            'moe_aux_loss': [],
            'multi_token_loss': []
        }
        
        # Load data
        self.data = self.load_data()
        self.n = len(self.data)

    def load_data(self):
        """Load the training data"""
        try:
            data_file = os.path.join('src', 'data', 'train.bin')
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Training data file not found at {data_file}")
            
            # Load data as numpy array first
            data = np.memmap(data_file, dtype=np.uint16, mode='r')
            # Convert to tensor
            data = torch.from_numpy(data.copy())  # Make a copy to ensure it's writable
            return data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def get_batch(self, split):
        """Get a batch of data"""
        try:
            # Generate random indices
            ix = torch.randint(len(self.data) - self.model.config.block_size, (self.batch_size,))
            
            # Get input sequences
            x = torch.stack([self.data[i:i+self.model.config.block_size].long() for i in ix])
            # Get target sequences (shifted by 1)
            y = torch.stack([self.data[i+1:i+1+self.model.config.block_size].long() for i in ix])
            
            # Move to device
            x, y = x.to(self.device), y.to(self.device)
            return x, y
        except Exception as e:
            print(f"Error in get_batch: {str(e)}")
            raise

    def get_lr(self, it):
        """Get learning rate for current iteration"""
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    def estimate_loss(self):
        """Estimate loss on validation set"""
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                try:
                    X, Y = self.get_batch(split)
                    with torch.no_grad():
                        if self.scaler is not None:
                            with torch.amp.autocast('cuda'):
                                logits, loss = self.model(X, Y)
                        else:
                            logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                except Exception as e:
                    print(f"Error during evaluation: {str(e)}")
                    continue
            out[split] = losses.mean()
        self.model.train()
        return out

    def check_disk_space(self, required_space_mb=1000):
        """Check if there's enough disk space for saving the model"""
        try:
            # Get disk usage statistics
            disk_usage = psutil.disk_usage('/')
            free_space_mb = disk_usage.free / (1024 * 1024)  # Convert to MB
            
            if free_space_mb < required_space_mb:
                print(f"Warning: Low disk space. Only {free_space_mb:.2f}MB free, {required_space_mb}MB required")
                return False
            return True
        except Exception as e:
            print(f"Warning: Could not check disk space: {e}")
            return True  # Continue anyway if we can't check

    def save_checkpoint(self, iter_num, loss, is_best=False):
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iter_num': iter_num,
                'loss': loss,
                'config': self.model.config,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'metrics': self.metrics,
                'best_loss': self.best_loss
            }
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{iter_num}.pt')
            torch.save(checkpoint, checkpoint_path)
            
            if is_best:
                best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
                torch.save(checkpoint, best_path)
                print(f"Saved best model with loss {loss:.4f}")
            
            print(f"Saved checkpoint to {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            return False

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint with error handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.current_iter = checkpoint['iter_num']
            self.best_loss = checkpoint['loss']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.learning_rates = checkpoint.get('learning_rates', [])
            self.metrics = checkpoint.get('metrics', self.metrics)
            print(f"Successfully loaded checkpoint from iteration {self.current_iter}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return False

    def train(self):
        """Train the DeepSeek model"""
        print(f"DeepSeek Training started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model.config.n_layer} layers, {self.model.config.n_head} heads, {self.model.config.n_embd} dims")
        print(f"MLA: {self.model.config.mla_kv_heads} KV heads, MoE: {self.model.config.moe_num_experts} experts")
        print(f"Multi-token prediction: {self.model.config.multi_token_predict} tokens")
        start_time = time.time()
        
        try:
            # Initialize training
            X, Y = self.get_batch('train')
            best_loss = float('inf')
            current_loss = None
            
            for iter_num in range(self.current_iter, self.max_iters):
                self.current_iter = iter_num
                
                # Determine and set the learning rate for this iteration
                lr = self.get_lr(iter_num)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Forward pass with mixed precision
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        logits, loss = self.model(X, Y)
                else:
                    logits, loss = self.model(X, Y)
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Get new batch
                X, Y = self.get_batch('train')
                
                # Track metrics
                current_loss = loss.item()
                self.train_losses.append(current_loss)
                self.learning_rates.append(lr)
                
                # Update best loss
                if current_loss < best_loss:
                    best_loss = current_loss
                
                # Evaluation
                if iter_num % self.eval_interval == 0:
                    losses = self.estimate_loss()
                    self.val_losses.append(losses['val'])
                    
                    # Save checkpoint if it's the best so far
                    if losses['val'] < self.best_loss:
                        self.best_loss = losses['val']
                        self.save_checkpoint(iter_num, losses['val'], is_best=True)
                    
                    # Regular checkpoint saving
                    if iter_num % (self.eval_interval * 5) == 0:
                        self.save_checkpoint(iter_num, losses['val'])
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    print(f"iter {iter_num}: train_loss {current_loss:.4f}, val_loss {losses['val']:.4f}, "
                          f"lr {lr:.2e}, time {elapsed:.2f}s")
                    
                    # Memory usage
                    if self.device == 'cuda':
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        print(f"GPU memory: {memory_used:.2f} GB")
                
                # Memory cleanup
                if iter_num % 100 == 0:
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            # Final checkpoint
            self.save_checkpoint(self.max_iters, current_loss)
            
            # Plot training metrics
            self.plot_metrics()
            
            print(f"Training completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Save emergency checkpoint
            if current_loss is not None:
                self.save_checkpoint(self.current_iter, current_loss)
            raise

    def plot_losses(self, train_losses, val_losses):
        """Plot training and validation losses"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_metrics(self):
        """Plot comprehensive training metrics"""
        if not self.train_losses or not self.val_losses:
            print("No metrics to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.train_losses, label='Training Loss', alpha=0.7)
        axes[0, 0].plot(self.val_losses, label='Validation Loss', alpha=0.7)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.learning_rates)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True)
        
        # Memory usage
        if self.metrics['memory_usage']:
            axes[1, 0].plot(self.metrics['memory_usage'])
            axes[1, 0].set_title('GPU Memory Usage')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Memory (GB)')
            axes[1, 0].grid(True)
        
        # Gradient norm
        if self.metrics['grad_norm']:
            axes[1, 1].plot(self.metrics['grad_norm'])
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('deepseek_training_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Training metrics saved to deepseek_training_metrics.png")


def create_deepseek_trainer(model, config):
    """Create a DeepSeek trainer with the given configuration"""
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Trainer
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
    
    return trainer 