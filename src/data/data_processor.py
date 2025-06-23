"""
Data Processor for DeepSeek Children's Stories Model
Handles dataset loading, preprocessing, and tokenization for children's story generation
"""

import tiktoken
import os
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from typing import Dict, List, Optional

def load_encoder_decoder():
    """Load the encoder and decoder for text processing"""
    enc = tiktoken.get_encoding("gpt2")
    return enc, enc

class DeepSeekDataProcessor:
    def __init__(self, config=None):
        # Initialize tokenizer with GPT-2 encoding
        self.enc = tiktoken.get_encoding("gpt2")
        
        # Special tokens for story structure (optimized for children's stories)
        self.special_tokens = {
            "story_start": "<|story|>",
            "story_end": "</|story|>",
            "prompt_start": "<|prompt|>",
            "prompt_end": "</|prompt|>",
            "moral_start": "<|moral|>",
            "moral_end": "</|moral|>",
            "character_start": "<|character|>",
            "character_end": "</|character|>"
        }
        
        # Ensure data directory exists
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Data directory: {self.data_dir}")
        
        # Configuration for processing
        self.max_length = 1024  # DeepSeek context window
        self.min_length = 50    # Minimum story length
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for children's stories"""
        # Basic text cleaning
        text = text.lower()  # Convert to lowercase for consistency
        text = text.replace('\n', ' ')  # Replace newlines with spaces
        text = ' '.join(text.split())  # Normalize whitespace
        
        # Ensure minimum length
        if len(text) < self.min_length:
            return ""
            
        return text
        
    def extract_story_elements(self, example: Dict) -> Dict:
        """Extract story elements for better structure"""
        prompt = self.preprocess_text(example.get('prompt', ''))
        story = self.preprocess_text(example.get('text', ''))
        
        # Extract potential moral or lesson
        moral = ""
        if 'moral' in example:
            moral = self.preprocess_text(example['moral'])
        elif 'lesson' in example:
            moral = self.preprocess_text(example['lesson'])
        
        # Extract main character if available
        character = ""
        if 'character' in example:
            character = self.preprocess_text(example['character'])
        
        return {
            'prompt': prompt,
            'story': story,
            'moral': moral,
            'character': character
        }
        
    def process(self, example: Dict) -> Dict:
        """Process a single example for DeepSeek model"""
        # Extract story elements
        elements = self.extract_story_elements(example)
        
        # Skip if no valid content
        if not elements['story'] or not elements['prompt']:
            return {'ids': [], 'len': 0}
        
        # Create structured text with special tokens
        full_text = (
            f"{self.special_tokens['prompt_start']} {elements['prompt']} {self.special_tokens['prompt_end']} "
        )
        
        # Add character information if available
        if elements['character']:
            full_text += f"{self.special_tokens['character_start']} {elements['character']} {self.special_tokens['character_end']} "
        
        # Add the main story
        full_text += f"{self.special_tokens['story_start']} {elements['story']} {self.special_tokens['story_end']}"
        
        # Add moral if available
        if elements['moral']:
            full_text += f" {self.special_tokens['moral_start']} {elements['moral']} {self.special_tokens['moral_end']}"
        
        # Tokenize with error handling
        try:
            ids = self.enc.encode_ordinary(full_text)
            
            # Ensure the sequence isn't too long
            if len(ids) > self.max_length:
                ids = ids[:self.max_length]
            
            # Skip if too short
            if len(ids) < 20:
                return {'ids': [], 'len': 0}
                
            out = {'ids': ids, 'len': len(ids)}
            return out
            
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            return {'ids': [], 'len': 0}
        
    def prepare_dataset(self) -> Dict:
        """Prepare the Children Stories Collection dataset for DeepSeek training"""
        # Load the Children Stories Collection dataset
        print("Loading Children Stories Collection dataset...")
        ds = load_dataset("ajibawa-2023/Children-Stories-Collection")
        
        train_bin_path = os.path.join(self.data_dir, "train.bin")
        val_bin_path = os.path.join(self.data_dir, "validation.bin")
        finetune_bin_path = os.path.join(self.data_dir, "finetune.bin")
        
        print(f"Checking for existing processed files...")
        
        # Check if all files exist
        if (os.path.exists(train_bin_path) and 
            os.path.exists(val_bin_path) and 
            os.path.exists(finetune_bin_path)):
            
            print("Found existing processed files!")
            print(f"Train file: {os.path.getsize(train_bin_path) / (1024*1024):.2f} MB")
            print(f"Validation file: {os.path.getsize(val_bin_path) / (1024*1024):.2f} MB")
            print(f"Finetune file: {os.path.getsize(finetune_bin_path) / (1024*1024):.2f} MB")
            
            return {
                "train": train_bin_path,
                "validation": val_bin_path,
                "finetune": finetune_bin_path
            }
        
        print("Processing dataset...")
        
        # Filter out examples that are too short or too long
        def filter_by_length(example):
            text_length = len(example.get('text', ''))
            return self.min_length <= text_length <= 2000  # Reasonable length for children's stories
        
        ds = ds.filter(filter_by_length)
        print(f"After filtering: {len(ds['train'])} examples")
        
        # Split the dataset into train, validation, and finetune sets
        train_val_test = ds["train"].train_test_split(test_size=0.2, seed=42)
        val_finetune = train_val_test["test"].train_test_split(test_size=0.5, seed=42)
        
        # Create a new dataset dictionary with all splits
        ds = {
            "train": train_val_test["train"],
            "validation": val_finetune["train"],
            "finetune": val_finetune["test"]
        }
        
        print(f"Dataset split sizes:")
        print(f"Training set: {len(ds['train'])} examples")
        print(f"Validation set: {len(ds['validation'])} examples")
        print(f"Finetune set: {len(ds['finetune'])} examples")
        
        # Process each split
        for split_name, split_data in ds.items():
            print(f"\nProcessing {split_name} split...")
            
            # Process the data
            tokenized = split_data.map(
                self.process,
                remove_columns=['text', 'prompt', 'text_token_length'],
                desc=f"tokenizing {split_name} split",
                num_proc=8,
            )
            
            # Filter out empty sequences
            tokenized = tokenized.filter(lambda x: x['len'] > 0)
            print(f"After processing: {len(tokenized)} valid examples")
            
            # Save to binary file
            filename = os.path.join(self.data_dir, f"{split_name}.bin")
            print(f"Saving {split_name} split to: {filename}")
            
            # Calculate total length
            arr_len = np.sum(tokenized['len'], dtype=np.uint64)
            dtype = np.uint16
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
            
            # Verify file was created
            if os.path.exists(filename):
                print(f"Successfully created {filename}")
                print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
            else:
                raise RuntimeError(f"Failed to create {filename}")
        
        return {
            "train": train_bin_path,
            "validation": val_bin_path,
            "finetune": finetune_bin_path
        }
    
    def load_binary_data(self, filepath: str) -> torch.Tensor:
        """Load binary data file as tensor"""
        try:
            data = np.memmap(filepath, dtype=np.uint16, mode='r')
            return torch.from_numpy(data.copy())
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            raise
    
    def get_batch(self, data: torch.Tensor, batch_size: int, block_size: int) -> tuple:
        """Get a batch of data for training"""
        # Generate random indices
        ix = torch.randint(len(data) - block_size, (batch_size,))
        
        # Get input sequences
        x = torch.stack([data[i:i+block_size].long() for i in ix])
        # Get target sequences (shifted by 1)
        y = torch.stack([data[i+1:i+1+block_size].long() for i in ix])
        
        return x, y
    
    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        try:
            return self.enc.decode(token_ids)
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            return ""
    
    def encode_text(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        try:
            return self.enc.encode_ordinary(text)
        except Exception as e:
            print(f"Error encoding text: {e}")
            return []


def main():
    """Main function to process the dataset"""
    print("DeepSeek Children's Stories Data Processor")
    print("=" * 50)
    
    processor = DeepSeekDataProcessor()
    processor.prepare_dataset()
    
    print("\nData processing completed successfully!")
    print("Files created:")
    print("- src/data/train.bin")
    print("- src/data/validation.bin") 
    print("- src/data/finetune.bin")


if __name__ == "__main__":
    main() 