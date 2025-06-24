"""
Simple and Fast DataLoader for TinyStories Dataset
Maximum simplicity with native multiprocessing for speed
"""

from multiprocessing import cpu_count
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, Dataset
import tiktoken
from typing import Dict, List, Any, cast


def fast_collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """Fast collate function using PyTorch's optimized pad_sequence"""
    # Convert to tensors
    input_ids_list = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    
    # Pad sequences efficiently
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    
    # Create attention mask
    attention_mask = pad_sequence(
        [torch.ones(len(seq), dtype=torch.long) for seq in input_ids_list],
        batch_first=True, 
        padding_value=0
    )
    
    # Create targets (shifted by 1 for language modeling)
    targets = input_ids.clone()
    targets[:, :-1] = targets[:, 1:]
    targets[:, -1] = -100
    targets[attention_mask == 0] = -100  # Mask padding
    
    return {
        "input_ids": input_ids,
        "targets": targets,
        "attention_mask": attention_mask
    }


def create_dataloaders(
    batch_size: int = 32,
    num_workers: int = 2,
    dataset_name: str = "roneneldan/TinyStories",
    max_length: int = 1024
) -> Dict[str, DataLoader]:
    """
    Create simple, fast DataLoaders with batch tokenization
    
    Args:
        batch_size: Batch size for training
        num_workers: DataLoader workers (keep low, 1-2)
        dataset_name: HuggingFace dataset name
    """
    
    print(f"Loading {dataset_name} dataset...")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    def tokenize_batch(examples: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        """Simple batch tokenization with max length truncation"""
        tokenized = tokenizer.encode_batch(examples["text"])
        # Truncate sequences that are too long
        truncated = [seq[:max_length] for seq in tokenized]
        return {"input_ids": truncated}
    
    dataloaders = {}
    
    # Process both splits
    for split_name in ["train", "validation"]:
        print(f"Processing {split_name} split...")
        
        # Load dataset
        dataset = cast(Dataset, load_dataset(dataset_name, split=split_name))
        
        # Fast batch tokenization
        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            batch_size=1000,
            num_proc=cpu_count(),
            remove_columns=["text"]
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            cast(Any, tokenized_dataset),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            collate_fn=fast_collate_fn,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False
        )
        
        dataloaders[split_name] = dataloader
        print(f"✅ {split_name} DataLoader ready")
    
    return dataloaders


if __name__ == "__main__":
    print("Testing Simple Fast DataLoader...")
    
    # Test with optimal settings
    dataloaders = create_dataloaders(
        batch_size=8,
        num_workers=1,
        max_length=1024
    )
    
    train_loader = dataloaders['train']
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Quick test
    import time
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        if i == 0:
            print(f"Batch shape: {batch['input_ids'].shape}")
            # Decode sample
            real_tokens = batch['input_ids'][0][batch['attention_mask'][0] == 1]
            text = tokenizer.decode(real_tokens[:30].tolist())
            print(f"Sample: '{text}...'")
        
        if i >= 4:
            break
    
    elapsed = time.time() - start_time
    print(f"Processed 5 batches in {elapsed:.2f}s ({5/elapsed:.1f} batches/sec)")
    print("✅ Test complete!")