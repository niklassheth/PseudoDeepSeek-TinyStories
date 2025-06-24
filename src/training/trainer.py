"""
Modern DeepSeek Trainer with DataLoader and Epochs
Memory-efficient training with proper epoch management
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm.auto import tqdm
import os
import time
from typing import Dict, Optional, Callable, Any
import math
# Profiling imports
import torch.profiler
try:
    import nvidia_ml_py3 as nvml
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: nvidia_ml_py3 not available. GPU monitoring will be limited.")
    NVML_AVAILABLE = False

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
            self.scaler = GradScaler('cuda')
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
            num_workers=num_workers,
            max_length=model.config.block_size
        )
        
        print(f"DataLoaders created:")
        print(f"  - Streaming mode: {streaming}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Num workers: {num_workers}")
        print(f"  - Max length: {model.config.block_size}")
        
        # Initialize profiling
        self.profiling_active = False
        
        # Initialize NVIDIA ML for GPU monitoring
        if torch.cuda.is_available() and NVML_AVAILABLE:
            try:
                nvml.nvmlInit()
                self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_monitoring = True
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
                self.gpu_monitoring = False
        else:
            self.gpu_monitoring = False
        
    
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
                    with autocast('cuda'):
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
                with autocast('cuda'):
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
    
    def run_profiling_session(self, num_steps: int = 2):
        """Run profiling session"""
        print(f"🔍 Profiling {sum(p.numel() for p in self.model.parameters()):,} parameter model")
        
        # Run performance profiling
        trace_file = self.profile_training_step(num_steps=num_steps)
        
        # Run memory profiling  
        snapshot_file = self.profile_memory_usage(num_steps=min(num_steps, 3))
        
        print(f"\n✅ Profiling complete:")
        print(f"  Chrome trace: {trace_file}")
        print(f"  Memory snapshot: {snapshot_file}")
        
        return {'trace': trace_file, 'snapshot': snapshot_file}
    
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
    
    # ==================== PROFILING METHODS ====================
    
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        stats = {}
        
        if torch.cuda.is_available():
            # PyTorch GPU stats
            stats['torch_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            stats['torch_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            stats['torch_max_allocated_gb'] = torch.cuda.max_memory_allocated() / 1e9
            
            # NVIDIA ML stats (if available)
            if self.gpu_monitoring:
                try:
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    util_info = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    stats['nvml_used_gb'] = mem_info.used / 1e9
                    stats['nvml_total_gb'] = mem_info.total / 1e9
                    stats['nvml_gpu_util_percent'] = util_info.gpu
                    stats['nvml_memory_util_percent'] = util_info.memory
                except Exception as e:
                    stats['nvml_error'] = str(e)
        
        return stats
    
    def print_memory_stats(self, prefix: str = ""):
        """Print current memory statistics"""
        stats = self.get_gpu_stats()
        if stats:
            print(f"{prefix}Memory Stats:")
            if 'torch_allocated_gb' in stats:
                print(f"  PyTorch - Allocated: {stats['torch_allocated_gb']:.2f}GB, Reserved: {stats['torch_reserved_gb']:.2f}GB")
                print(f"  PyTorch - Max Allocated: {stats['torch_max_allocated_gb']:.2f}GB")
            if 'nvml_gpu_util_percent' in stats:
                print(f"  GPU Utilization: {stats['nvml_gpu_util_percent']}%, Memory Utilization: {stats['nvml_memory_util_percent']}%")
                print(f"  NVML Memory: {stats['nvml_used_gb']:.2f}GB / {stats['nvml_total_gb']:.2f}GB")
    
    def profile_training_step(self, num_steps: int = 3, warmup_steps: int = 5) -> str:
        """Profile training steps and generate Chrome trace"""
        profile_dir = os.path.join(self.checkpoint_dir, 'profiling')
        os.makedirs(profile_dir, exist_ok=True)
        
        # Ensure we have enough steps for the schedule
        total_steps = max(num_steps, warmup_steps + 3)
        active_steps = total_steps - warmup_steps
        
        print(f"🔍 Performance profiling: {total_steps} steps ({warmup_steps} warmup + {active_steps} active)")
        
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        schedule = torch.profiler.schedule(wait=1, warmup=warmup_steps, active=active_steps, repeat=1)
        trace_file = os.path.join(profile_dir, f'trace_{int(time.time())}.json')
        
        with torch.profiler.profile(activities=activities, schedule=schedule, record_shapes=True, profile_memory=True) as prof:
            self.model.train()
            train_loader = self.dataloaders['train']
            
            for step, batch in enumerate(train_loader):
                if step >= total_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                if self.scaler is not None:
                    with autocast('cuda'):
                        _, loss = self.model(input_ids, targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _, loss = self.model(input_ids, targets)
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                prof.step()
        
        # Export Chrome trace
        try:
            prof.export_chrome_trace(trace_file)
            print(f"📁 Chrome trace: {trace_file}")
        except RuntimeError:
            print(f"⚠️  Chrome trace export issue (may be saved already)")
        
        return trace_file
    
    
    def profile_memory_usage(self, num_steps: int = 3) -> str:
        """Profile memory usage and generate memory snapshot"""
        print(f"🧠 Memory profiling: {num_steps} steps")
        
        profile_dir = os.path.join(self.checkpoint_dir, 'profiling')
        os.makedirs(profile_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history(max_entries=100000)
        
        try:
            self.model.train()
            train_loader = self.dataloaders['train']
            
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                if self.scaler is not None:
                    with autocast('cuda'):
                        _, loss = self.model(input_ids, targets)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _, loss = self.model(input_ids, targets)
                    loss.backward()
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
        
        finally:
            if torch.cuda.is_available():
                snapshot_file = os.path.join(profile_dir, f'memory_snapshot_{int(time.time())}.pkl')
                try:
                    torch.cuda.memory._dump_snapshot(snapshot_file)
                    print(f"📁 Memory snapshot: {snapshot_file}")
                    print(f"🌐 Upload to https://pytorch.org/memory_viz")
                except Exception as e:
                    print(f"⚠️  Memory snapshot failed: {e}")
                    snapshot_file = "N/A"
                
                torch.cuda.memory._record_memory_history(enabled=None)
        
        return snapshot_file
    
# Legacy function for compatibility
def create_deepseek_trainer(*args, **kwargs):
    """Create trainer (legacy compatibility)"""
    return DeepSeekTrainerV2(*args, **kwargs)