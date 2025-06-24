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
import json
from contextlib import contextmanager
from collections import defaultdict

# Profiling imports
import torch.profiler
import psutil
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
        self.timing_stats = defaultdict(list)
        self.profiler = None
        
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
        """Run a dedicated profiling session"""
        print("\n🔍 Starting dedicated profiling session...")
        
        # Print model summary
        model_summary = self.profile_model_summary()
        print(f"\n📋 Model Summary:")
        print(f"  Total parameters: {model_summary['model_info']['total_params']:,}")
        print(f"  Model size: {model_summary['model_info']['model_size_mb']:.1f}MB")
        
        # Print GPU info
        if model_summary['gpu_info']:
            print(f"\n🖥️  System Info:")
            if 'nvml_total_gb' in model_summary['gpu_info']:
                print(f"  GPU Memory: {model_summary['gpu_info']['nvml_total_gb']:.1f}GB")
            print(f"  CPU Cores: {model_summary['system_info']['cpu_count']}")
            print(f"  System Memory: {model_summary['system_info']['memory_gb']:.1f}GB")
        
        # Run comprehensive profiling
        print("\n🔍 Running performance profiling...")
        performance_analysis = self.profile_training_step(num_steps=num_steps)
        
        print("\n🧠 Running memory profiling...")
        memory_analysis = self.profile_memory_usage(num_steps=min(num_steps, 3))
        
        print("\n✅ Comprehensive profiling session complete!")
        print("\n📊 Available Analysis:")
        print("  - Performance: Chrome trace files for timeline analysis")
        print("  - Memory: PyTorch memory snapshots for detailed memory analysis")
        print("  - Reports: Text analysis files with recommendations")
        
        return {'performance': performance_analysis, 'memory': memory_analysis}
    
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
            
            # Print timing summary if available
            if self.timing_stats:
                print(f"\n⏱️  Training Timing Summary:")
                for op_name, times in self.timing_stats.items():
                    avg_time = sum(times) / len(times)
                    print(f"   {op_name}: {avg_time:.4f}s average")
            
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ==================== PROFILING METHODS ====================
    
    @contextmanager
    def timer(self, name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        yield
        elapsed = time.time() - start_time
        self.timing_stats[name].append(elapsed)
    
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
    
    def profile_training_step(self, num_steps: int = 3, warmup_steps: int = 1) -> str:
        """Profile training steps and return analysis"""
        print(f"🔍 Starting profiling for {num_steps} steps (warmup: {warmup_steps})...")
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        profile_dir = os.path.join(self.checkpoint_dir, 'profiling')
        os.makedirs(profile_dir, exist_ok=True)
        
        # Configure profiler
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=warmup_steps,
            active=num_steps - warmup_steps,
            repeat=1
        )
        
        trace_file = os.path.join(profile_dir, f'trace_{int(time.time())}.json')
        
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True
        ) as prof:
            
            self.model.train()
            train_loader = self.dataloaders['train']
            step_times = []
            
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                
                step_start = time.time()
                
                # Get batch data
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward and backward pass
                with self.timer('forward_pass'):
                    if self.scaler is not None:
                        with autocast('cuda'):
                            logits, loss = self.model(input_ids, targets)
                    else:
                        logits, loss = self.model(input_ids, targets)
                
                with self.timer('backward_pass'):
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                with self.timer('optimizer_step'):
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Print progress
                if step % 5 == 0:
                    gpu_stats = self.get_gpu_stats()
                    gpu_util = gpu_stats.get('nvml_gpu_util_percent', 'N/A')
                    print(f"  Step {step}: {step_time:.3f}s, Loss: {loss.item():.4f}, GPU: {gpu_util}%")
                
                prof.step()
        
        # Export Chrome trace
        try:
            prof.export_chrome_trace(trace_file)
            print(f"📁 Chrome trace saved to: {trace_file}")
        except RuntimeError as e:
            if "already saved" in str(e):
                print(f"⚠️  Chrome trace already saved (this is normal)")
            else:
                print(f"❌ Could not save Chrome trace: {e}")
                trace_file = "N/A"
        
        # Analyze results
        analysis = self._analyze_profiling_results(prof, step_times, trace_file)
        
        return analysis
    
    def _analyze_profiling_results(self, prof: torch.profiler.profile, step_times: list, trace_file: str) -> str:
        """Analyze profiling results and generate report"""
        analysis_lines = []
        analysis_lines.append("\n" + "="*60)
        analysis_lines.append("🔍 PROFILING ANALYSIS REPORT")
        analysis_lines.append("="*60)
        
        # Basic timing stats
        avg_step_time = sum(step_times) / len(step_times)
        analysis_lines.append(f"\n📊 Basic Performance:")
        analysis_lines.append(f"  Average step time: {avg_step_time:.3f}s")
        analysis_lines.append(f"  Steps per second: {1/avg_step_time:.2f}")
        analysis_lines.append(f"  Tokens per second: {self.batch_size * self.model.config.block_size / avg_step_time:.0f}")
        
        # Memory stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            analysis_lines.append(f"\n💾 Memory Usage:")
            if 'torch_max_allocated_gb' in gpu_stats:
                analysis_lines.append(f"  Peak GPU memory: {gpu_stats['torch_max_allocated_gb']:.2f}GB")
            if 'nvml_gpu_util_percent' in gpu_stats:
                analysis_lines.append(f"  GPU utilization: {gpu_stats['nvml_gpu_util_percent']}%")
                analysis_lines.append(f"  Memory utilization: {gpu_stats['nvml_memory_util_percent']}%")
        
        # Top CPU operations
        analysis_lines.append(f"\n🖥️  Top CPU Operations:")
        cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        analysis_lines.append(cpu_table)
        
        # Top CUDA operations (if available)
        if torch.cuda.is_available():
            analysis_lines.append(f"\n🚀 Top CUDA Operations:")
            cuda_table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
            analysis_lines.append(cuda_table)
        
        # Timing breakdown
        if self.timing_stats:
            analysis_lines.append(f"\n⏱️  Operation Timing Breakdown:")
            for op_name, times in self.timing_stats.items():
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                analysis_lines.append(f"  {op_name}: {avg_time:.4f}s avg, {total_time:.4f}s total ({len(times)} calls)")
        
        # Memory efficiency analysis
        analysis_lines.append(f"\n🧠 Memory Efficiency Analysis:")
        if 'nvml_gpu_util_percent' in gpu_stats:
            gpu_util = gpu_stats['nvml_gpu_util_percent']
            if gpu_util < 70:
                analysis_lines.append(f"  ⚠️  LOW GPU utilization ({gpu_util}%) - possible CPU bottleneck")
            elif gpu_util > 95:
                analysis_lines.append(f"  ✅ HIGH GPU utilization ({gpu_util}%) - good compute efficiency")
            else:
                analysis_lines.append(f"  ✅ MODERATE GPU utilization ({gpu_util}%) - room for improvement")
        
        # Recommendations
        analysis_lines.append(f"\n💡 Optimization Recommendations:")
        analysis_lines.append(f"  1. Check Chrome trace file: {trace_file}")
        analysis_lines.append(f"  2. Open in Chrome at: chrome://tracing/")
        analysis_lines.append(f"  3. Look for:")
        analysis_lines.append(f"     - Long gaps between GPU kernels (data loading bottleneck)")
        analysis_lines.append(f"     - Inefficient attention patterns (bright yellow blocks)")
        analysis_lines.append(f"     - Memory allocation spikes (red memory timeline)")
        analysis_lines.append(f"     - CPU-GPU synchronization issues")
        
        analysis_text = "\n".join(analysis_lines)
        
        # Save analysis to file
        analysis_file = os.path.join(os.path.dirname(trace_file), f'analysis_{int(time.time())}.txt')
        with open(analysis_file, 'w') as f:
            f.write(analysis_text)
        
        print(analysis_text)
        print(f"\n📁 Analysis saved to: {analysis_file}")
        
        return analysis_text
    
    def profile_model_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive model profiling summary"""
        print("\n🔍 Generating model profiling summary...")
        
        summary = {
            'model_info': {
                'total_params': sum(p.numel() for p in self.model.parameters()),
                'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1e6,
                'architecture': type(self.model).__name__,
                'config': self.model.config.__dict__ if hasattr(self.model, 'config') else {}
            },
            'gpu_info': self.get_gpu_stats(),
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1e9,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            }
        }
        
        # Model layer analysis
        layer_info = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                params = sum(p.numel() for p in module.parameters())
                if params > 0:
                    layer_info.append({
                        'name': name,
                        'type': type(module).__name__,
                        'params': params,
                        'size_mb': sum(p.numel() * p.element_size() for p in module.parameters()) / 1e6
                    })
        
        # Sort by parameter count
        layer_info.sort(key=lambda x: x['params'], reverse=True)
        summary['layer_analysis'] = layer_info[:20]  # Top 20 layers
        
        return summary
    
    def profile_memory_usage(self, num_steps: int = 3) -> str:
        """Detailed memory profiling with PyTorch memory snapshots"""
        print(f"\n🧠 Starting memory profiling for {num_steps} steps...")
        
        # Create profiling directory
        profile_dir = os.path.join(self.checkpoint_dir, 'profiling')
        os.makedirs(profile_dir, exist_ok=True)
        
        # Reset memory stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        # Start memory history recording
        if torch.cuda.is_available():
            torch.cuda.memory._record_memory_history(max_entries=100000)
        
        memory_stats = []
        
        try:
            self.model.train()
            train_loader = self.dataloaders['train']
            
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                
                step_stats = {'step': step}
                
                # Get batch data
                input_ids = batch['input_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Before forward pass
                step_stats['before_forward'] = self.get_gpu_stats()
                
                # Forward pass
                with self.timer('forward_pass'):
                    if self.scaler is not None:
                        with autocast('cuda'):
                            logits, loss = self.model(input_ids, targets)
                    else:
                        logits, loss = self.model(input_ids, targets)
                
                step_stats['after_forward'] = self.get_gpu_stats()
                
                # Backward pass
                with self.timer('backward_pass'):
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                
                step_stats['after_backward'] = self.get_gpu_stats()
                
                # Optimizer step
                with self.timer('optimizer_step'):
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)  # Memory efficient
                
                step_stats['after_optimizer'] = self.get_gpu_stats()
                step_stats['loss'] = loss.item()
                
                memory_stats.append(step_stats)
                
                # Print progress
                if step % 2 == 0:
                    current_mem = step_stats['after_optimizer'].get('torch_allocated_gb', 0)
                    peak_mem = step_stats['after_optimizer'].get('torch_max_allocated_gb', 0)
                    print(f"  Step {step}: Loss {loss.item():.4f}, Current: {current_mem:.2f}GB, Peak: {peak_mem:.2f}GB")
        
        finally:
            # Save memory snapshot
            if torch.cuda.is_available():
                snapshot_file = os.path.join(profile_dir, f'memory_snapshot_{int(time.time())}.pkl')
                try:
                    torch.cuda.memory._dump_snapshot(snapshot_file)
                    print(f"\n📁 Memory snapshot saved: {snapshot_file}")
                    print(f"🌐 Upload to https://pytorch.org/memory_viz for interactive analysis")
                except Exception as e:
                    print(f"⚠️  Could not save memory snapshot: {e}")
                    snapshot_file = "N/A"
                
                torch.cuda.memory._record_memory_history(enabled=None)
        
        # Analyze memory patterns
        analysis = self._analyze_memory_patterns(memory_stats, snapshot_file)
        return analysis
    
    def _analyze_memory_patterns(self, memory_stats: list, snapshot_file: str) -> str:
        """Analyze memory usage patterns"""
        analysis_lines = []
        analysis_lines.append("\n" + "="*60)
        analysis_lines.append("🧠 MEMORY ANALYSIS REPORT")
        analysis_lines.append("="*60)
        
        if not memory_stats:
            analysis_lines.append("No memory statistics collected.")
            return "\n".join(analysis_lines)
        
        # Extract memory trends
        peak_memories = []
        forward_memories = []
        backward_memories = []
        
        for stats in memory_stats:
            if 'after_optimizer' in stats and 'torch_max_allocated_gb' in stats['after_optimizer']:
                peak_memories.append(stats['after_optimizer']['torch_max_allocated_gb'])
            
            if 'after_forward' in stats and 'torch_allocated_gb' in stats['after_forward']:
                forward_memories.append(stats['after_forward']['torch_allocated_gb'])
            
            if 'after_backward' in stats and 'torch_allocated_gb' in stats['after_backward']:
                backward_memories.append(stats['after_backward']['torch_allocated_gb'])
        
        # Memory statistics
        if peak_memories:
            max_peak = max(peak_memories)
            avg_forward = sum(forward_memories) / len(forward_memories) if forward_memories else 0
            avg_backward = sum(backward_memories) / len(backward_memories) if backward_memories else 0
            
            analysis_lines.append(f"\n📊 Memory Usage Statistics:")
            analysis_lines.append(f"  Peak GPU memory: {max_peak:.2f}GB")
            analysis_lines.append(f"  Average after forward: {avg_forward:.2f}GB")
            analysis_lines.append(f"  Average after backward: {avg_backward:.2f}GB")
            analysis_lines.append(f"  Memory increase (forward → backward): {avg_backward - avg_forward:.2f}GB")
        
        # Memory growth analysis
        if len(peak_memories) > 1:
            initial_peak = peak_memories[0]
            final_peak = peak_memories[-1]
            memory_growth = final_peak - initial_peak
            
            analysis_lines.append(f"\n📈 Memory Growth Analysis:")
            analysis_lines.append(f"  Initial peak: {initial_peak:.2f}GB")
            analysis_lines.append(f"  Final peak: {final_peak:.2f}GB")
            analysis_lines.append(f"  Memory growth: {memory_growth:.2f}GB")
            
            if memory_growth > 0.1:
                analysis_lines.append(f"  ⚠️  Significant memory growth detected - possible memory leak")
            else:
                analysis_lines.append(f"  ✅ Stable memory usage - no significant growth")
        
        # Memory efficiency recommendations
        analysis_lines.append(f"\n💡 Memory Optimization Recommendations:")
        
        if max_peak > 0:
            if max_peak > 20:
                analysis_lines.append(f"  🔴 HIGH memory usage ({max_peak:.1f}GB) - Consider:")
                analysis_lines.append(f"     - Reduce batch size")
                analysis_lines.append(f"     - Enable gradient checkpointing")
                analysis_lines.append(f"     - Use activation checkpointing for transformer blocks")
            elif max_peak > 10:
                analysis_lines.append(f"  🟡 MODERATE memory usage ({max_peak:.1f}GB) - Consider:")
                analysis_lines.append(f"     - Optimize attention mechanisms")
                analysis_lines.append(f"     - Reduce MoE experts if possible")
            else:
                analysis_lines.append(f"  🟢 GOOD memory usage ({max_peak:.1f}GB) - Room for:")
                analysis_lines.append(f"     - Larger batch sizes")
                analysis_lines.append(f"     - Longer context lengths")
        
        # Analysis files
        analysis_lines.append(f"\n📁 Memory Analysis Files:")
        analysis_lines.append(f"  Memory snapshot: {snapshot_file}")
        analysis_lines.append(f"  Interactive viewer: https://pytorch.org/memory_viz")
        
        analysis_text = "\n".join(analysis_lines)
        
        # Save analysis
        analysis_file = os.path.join(os.path.dirname(snapshot_file) if snapshot_file != "N/A" else self.checkpoint_dir, 
                                   f'memory_analysis_{int(time.time())}.txt')
        try:
            with open(analysis_file, 'w') as f:
                f.write(analysis_text)
            print(f"\n📄 Memory analysis saved to: {analysis_file}")
        except Exception as e:
            print(f"⚠️  Could not save memory analysis: {e}")
        
        print(analysis_text)
        return analysis_text


# Legacy function for compatibility
def create_deepseek_trainer(*args, **kwargs):
    """Create trainer (legacy compatibility)"""
    return DeepSeekTrainerV2(*args, **kwargs)