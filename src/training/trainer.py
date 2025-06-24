"""
Modern DeepSeek Trainer with DataLoader and Epochs
Memory-efficient training with proper epoch management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import os
import time
from typing import Dict, Optional
import math

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data.dataloader import create_dataloaders


class DeepSeekTrainerV2:
    """Modern trainer with DataLoader, epochs, and memory efficiency"""
    
    def __init__(self, 
                 model,
                 optimizer,
                 device: str,
                 batch_size: int = 32,
                 max_epochs: int = 10,
                 max_iters: Optional[int] = None,
                 eval_interval: int = 1000,
                 learning_rate: float = 6e-4,
                 warmup_iters: int = 2000,
                 lr_decay_iters: int = 20000,
                 min_lr: float = 6e-5,
                 checkpoint_dir: str = 'checkpoints',
                 use_mixed_precision: bool = True,
                 num_workers: int = 4,
                 streaming: bool = True,
                 gradient_accumulation_steps: int = 1):
        
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.learning_rate = learning_rate
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.checkpoint_dir = checkpoint_dir
        self.use_mixed_precision = use_mixed_precision
        self.num_workers = num_workers
        self.streaming = streaming
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Training state
        self.current_epoch = 0
        self.current_iter = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize gradient scaler for mixed precision
        if use_mixed_precision and device == 'cuda':
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Initialize metrics
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        # Create dataloaders
        print("Creating dataloaders...")
        self.dataloaders = create_dataloaders(
            batch_size=batch_size,
            max_length=model.config.block_size,
            num_workers=num_workers,
            streaming=streaming
        )
        
        print(f"DataLoaders created:")
        print(f"  - Streaming mode: {streaming}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Max length: {model.config.block_size}")
        
        # Fix the model's loss function to use correct ignore_index
        self._fix_model_ignore_index()
    
    def _fix_model_ignore_index(self):
        """Update model to use -100 as ignore_index for consistency with dataloader"""
        # This is a quick fix - ideally this should be in model config
        print("Note: The model uses ignore_index=-1, but dataloader uses -100.")
        print("Consider updating the model's loss calculation to use ignore_index=-100")
    
    def get_lr(self, it: int) -> float:
        """Get learning rate for current iteration"""
        # Linear warmup
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters
        # Cosine decay
        if it > self.lr_decay_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
    
    def evaluate(self, dataloader: DataLoader, max_batches: Optional[int] = None) -> float:
        """Evaluate the model on validation/test set"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches and batch_idx >= max_batches:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                # Note: attention_mask is available in batch['attention_mask'] if needed
                
                # Forward pass
                if self.scaler is not None and self.device == 'cuda':
                    with autocast():
                        logits, loss = self.model(input_ids, targets)
                else:
                    logits, loss = self.model(input_ids, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch: int, iteration: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'metrics': self.metrics,
            'config': self.model.config
        }
        
        # Save with descriptive filename
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch}_iter_{iteration}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Update best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 New best model saved! Val loss: {val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.current_iter = checkpoint['iteration']
        self.best_val_loss = checkpoint['val_loss']
        self.metrics = checkpoint.get('metrics', self.metrics)
        print(f"✅ Checkpoint loaded from epoch {self.current_epoch}")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        # Get data loader
        train_loader = self.dataloaders['train']
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Check if we've hit max iterations
            if self.max_iters and self.current_iter >= self.max_iters:
                print(f"Reached max iterations ({self.max_iters})")
                break
            
            # Get batch data
            input_ids = batch['input_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            # attention_mask = batch['attention_mask'].to(self.device)  # Available if model needs it
            
            # Update learning rate
            lr = self.get_lr(self.current_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            # Forward pass with gradient accumulation
            if self.scaler is not None:
                with autocast():
                    logits, loss = self.model(input_ids, targets)
                    loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                logits, loss = self.model(input_ids, targets)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            self.current_iter += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                'lr': f"{lr:.2e}",
                'iter': self.current_iter
            })
            
            # Periodic evaluation
            if self.current_iter % self.eval_interval == 0:
                val_loss = self.evaluate(self.dataloaders['validation'], max_batches=100)
                
                print(f"\n📊 Iteration {self.current_iter}:")
                print(f"   Train loss: {epoch_loss/num_batches:.4f}")
                print(f"   Val loss: {val_loss:.4f}")
                print(f"   Learning rate: {lr:.2e}")
                
                # Save metrics
                self.metrics['val_losses'].append(val_loss)
                self.metrics['learning_rates'].append(lr)
                
                # Save checkpoint
                self.save_checkpoint(epoch, self.current_iter, val_loss)
                
                # Memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Epoch statistics
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / max(num_batches, 1)
        
        print(f"\n✅ Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"   Average loss: {avg_loss:.4f}")
        print(f"   Batches processed: {num_batches}")
        
        self.metrics['train_losses'].append(avg_loss)
        self.metrics['epoch_times'].append(epoch_time)
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print("🚀 Starting training...")
        print(f"Configuration:")
        print(f"  - Max epochs: {self.max_epochs}")
        print(f"  - Max iterations: {self.max_iters}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Device: {self.device}")
        print(f"  - Mixed precision: {self.use_mixed_precision}")
        print(f"  - Gradient accumulation: {self.gradient_accumulation_steps}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.max_epochs):
                self.current_epoch = epoch
                
                print(f"\n{'='*60}")
                print(f"EPOCH {epoch+1}/{self.max_epochs}")
                print(f"{'='*60}")
                
                # Train for one epoch
                train_loss = self.train_epoch(epoch)
                
                # Final validation for the epoch
                print(f"\n🔍 Final epoch validation...")
                val_loss = self.evaluate(self.dataloaders['validation'], max_batches=200)
                
                print(f"\n📈 Epoch {epoch+1} Summary:")
                print(f"   Train loss: {train_loss:.4f}")
                print(f"   Val loss: {val_loss:.4f}")
                print(f"   Best val loss: {self.best_val_loss:.4f}")
                
                # Save epoch checkpoint
                self.save_checkpoint(epoch, self.current_iter, val_loss)
                
                # Check if we should stop
                if self.max_iters and self.current_iter >= self.max_iters:
                    print(f"Reached max iterations ({self.max_iters}), stopping...")
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        finally:
            total_time = time.time() - start_time
            print(f"\n🏁 Training completed in {total_time/60:.1f} minutes")
            print(f"   Total epochs: {self.current_epoch + 1}")
            print(f"   Total iterations: {self.current_iter}")
            print(f"   Best validation loss: {self.best_val_loss:.4f}")
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Legacy function for compatibility
def create_deepseek_trainer(*args, **kwargs):
    """Create trainer (legacy compatibility)"""
    return DeepSeekTrainerV2(*args, **kwargs)