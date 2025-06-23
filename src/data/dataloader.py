"""
Simple DataLoader for TinyStories Dataset
Using official HuggingFace patterns with minimal preprocessing
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import tiktoken
from typing import Dict, Optional


def tokenize_function(examples, tokenizer, max_length=1024):
    """Simple tokenization function"""
    # Tokenize the texts
    tokenized = tokenizer.encode_batch(examples["text"])
    
    # Truncate to max_length
    input_ids = []
    for tokens in tokenized:
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        input_ids.append(tokens)
    
    return {"input_ids": input_ids}


def collate_fn(batch):
    """Simple collate function for language modeling"""
    # Get all input_ids from the batch
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    
    # Find max length in batch
    max_len = max(len(seq) for seq in input_ids)
    
    # Pad sequences to same length
    padded_input_ids = []
    for seq in input_ids:
        if len(seq) < max_len:
            # Pad with zeros (we'll mask these in loss calculation)
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            padded_seq = torch.cat([seq, padding])
        else:
            padded_seq = seq
        padded_input_ids.append(padded_seq)
    
    # Stack into batch tensor
    input_ids = torch.stack(padded_input_ids)
    
    # Create targets (shifted by 1 for language modeling)
    targets = torch.zeros_like(input_ids)
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = -1  # Special token for padding/end
    
    return {
        "input_ids": input_ids,
        "targets": targets
    }


def create_simple_dataloaders(
    batch_size: int = 32,
    max_length: int = 1024,
    num_workers: int = 4,
    streaming: bool = True,
    dataset_name: str = "roneneldan/TinyStories"
) -> Dict[str, DataLoader]:
    """
    Create simple DataLoaders for TinyStories dataset
    
    Args:
        batch_size: Batch size
        max_length: Maximum sequence length
        num_workers: Number of data loading workers
        streaming: Whether to use streaming mode
        dataset_name: HuggingFace dataset name
    
    Returns:
        Dictionary with train and validation dataloaders
    """
    
    print(f"Loading {dataset_name} dataset...")
    print(f"Streaming: {streaming}, Batch size: {batch_size}, Max length: {max_length}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load dataset
    if streaming:
        dataset = load_dataset(dataset_name, streaming=True)
    else:
        dataset = load_dataset(dataset_name)
    
    # Simple tokenization using map
    def simple_tokenize(examples):
        # Just tokenize the text directly
        texts = examples["text"]
        tokenized_texts = []
        
        for text in texts:
            # Simple tokenization
            tokens = tokenizer.encode(text)
            
            # Truncate if too long
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            # Skip if too short
            if len(tokens) < 10:
                tokens = []
            
            tokenized_texts.append(tokens)
        
        return {"input_ids": tokenized_texts}
    
    dataloaders = {}
    
    # Process train and validation splits
    for split_name in ["train", "validation"]:
        print(f"Processing {split_name} split...")
        
        # Get the split
        split_dataset = dataset[split_name]
        
        # Apply tokenization
        tokenized_dataset = split_dataset.map(
            simple_tokenize,
            batched=True,
            remove_columns=["text"]  # Remove original text column
        )
        
        # Filter out empty sequences
        if not streaming:
            tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) > 0)
        
        # Set format to PyTorch
        tokenized_dataset = tokenized_dataset.with_format("torch")
        
        # Create DataLoader
        if streaming:
            # For streaming datasets
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True
            )
        else:
            # For regular datasets
            dataloader = DataLoader(
                tokenized_dataset,
                batch_size=batch_size,
                shuffle=(split_name == "train"),
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                drop_last=True
            )
        
        dataloaders[split_name] = dataloader
        print(f"✅ {split_name} DataLoader created")
    
    return dataloaders


# Backward compatibility function
def create_dataloaders(*args, **kwargs):
    """Backward compatibility wrapper"""
    return create_simple_dataloaders(*args, **kwargs)


if __name__ == "__main__":
    # Test the simple dataloader
    print("Testing Simple TinyStories DataLoader...")
    
    # Test with small batch for quick verification
    dataloaders = create_simple_dataloaders(
        batch_size=4, 
        max_length=512, 
        streaming=True,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )
    
    print("Testing train dataloader...")
    train_loader = dataloaders['train']
    
    # Test a few batches
    for i, batch in enumerate(train_loader):
        print(f"Batch {i+1}:")
        print(f"  Input shape: {batch['input_ids'].shape}")
        print(f"  Target shape: {batch['targets'].shape}")
        print(f"  Sample tokens: {batch['input_ids'][0][:20].tolist()}")
        
        if i >= 2:  # Test first few batches
            break
    
    print("✅ Simple DataLoader test completed successfully!")