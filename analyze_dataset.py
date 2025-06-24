#!/usr/bin/env python3
"""
Dataset Statistical Analysis for TinyStories
Analyzes sequence lengths and token distributions efficiently
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import tiktoken
from tqdm import tqdm
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data.dataloader import create_dataloaders


def analyze_dataset(max_samples: int = 50000, batch_size: int = 64):
    """
    Analyze tokenized dataset statistics efficiently
    
    Args:
        max_samples: Maximum number of samples to analyze (for speed)
        batch_size: Batch size for data loading
    """
    print("🔍 Starting TinyStories Dataset Analysis...")
    print(f"📊 Analyzing up to {max_samples:,} samples with batch size {batch_size}")
    
    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    vocab_size = tokenizer.n_vocab
    print(f"📝 Vocabulary size: {vocab_size:,} tokens")
    
    # Create dataloader
    print("\n🚀 Loading dataset...")
    dataloaders = create_dataloaders(
        batch_size=batch_size,
        num_workers=2,
        max_length=512  # Use current config
    )
    
    train_loader = dataloaders['train']
    
    # Statistics containers
    sequence_lengths = []
    token_counts = Counter()
    total_tokens = 0
    samples_processed = 0
    
    print(f"\n📈 Processing batches...")
    start_time = time.time()
    
    # Process batches
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing")):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        
        # Process each sequence in batch
        for seq_idx in range(input_ids.size(0)):
            # Get actual sequence length (non-padded)
            seq_len = attention_mask[seq_idx].sum().item()
            sequence_lengths.append(seq_len)
            
            # Get actual tokens (excluding padding)
            tokens = input_ids[seq_idx][:seq_len].tolist()
            
            # Count tokens
            for token in tokens:
                token_counts[token] += 1
                total_tokens += 1
            
            samples_processed += 1
            
            # Stop if we've hit our sample limit
            if samples_processed >= max_samples:
                break
        
        if samples_processed >= max_samples:
            break
    
    elapsed = time.time() - start_time
    print(f"\n✅ Processed {samples_processed:,} samples in {elapsed:.1f}s")
    print(f"⚡ Processing speed: {samples_processed/elapsed:.0f} samples/sec")
    print(f"🎯 Total tokens analyzed: {total_tokens:,}")
    
    # Analyze sequence lengths
    print("\n" + "="*50)
    print("📏 SEQUENCE LENGTH ANALYSIS")
    print("="*50)
    
    seq_lengths = np.array(sequence_lengths)
    print(f"Mean length: {seq_lengths.mean():.1f}")
    print(f"Median length: {np.median(seq_lengths):.1f}")
    print(f"Min length: {seq_lengths.min()}")
    print(f"Max length: {seq_lengths.max()}")
    print(f"Std deviation: {seq_lengths.std():.1f}")
    
    # Sequence length histogram
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.hist(seq_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Analyze token usage
    print("\n" + "="*50)
    print("🎭 TOKEN USAGE ANALYSIS")
    print("="*50)
    
    # Find unused tokens
    used_tokens = set(token_counts.keys())
    all_tokens = set(range(vocab_size))
    unused_tokens = all_tokens - used_tokens
    
    print(f"Tokens used: {len(used_tokens):,} / {vocab_size:,} ({len(used_tokens)/vocab_size*100:.1f}%)")
    print(f"Unused tokens: {len(unused_tokens):,} ({len(unused_tokens)/vocab_size*100:.1f}%)")
    
    # Token frequency analysis
    token_freqs = np.array(list(token_counts.values()))
    print(f"\nToken frequency stats:")
    print(f"Mean frequency: {token_freqs.mean():.1f}")
    print(f"Median frequency: {np.median(token_freqs):.1f}")
    print(f"Max frequency: {token_freqs.max():,}")
    print(f"Min frequency: {token_freqs.min()}")
    
    # Most common tokens
    print(f"\n🏆 Top 20 most common tokens:")
    most_common = token_counts.most_common(20)
    for i, (token_id, count) in enumerate(most_common):
        try:
            token_str = repr(tokenizer.decode([token_id]))
            print(f"{i+1:2d}. Token {token_id:5d}: {count:8,} times - {token_str}")
        except:
            print(f"{i+1:2d}. Token {token_id:5d}: {count:8,} times - [DECODE ERROR]")
    
    # Rare tokens (bottom 20)
    print(f"\n🦄 Bottom 20 rarest used tokens:")
    least_common = token_counts.most_common()[:-21:-1]  # Get last 20
    for i, (token_id, count) in enumerate(least_common):
        try:
            token_str = repr(tokenizer.decode([token_id]))
            print(f"{i+1:2d}. Token {token_id:5d}: {count:8,} times - {token_str}")
        except:
            print(f"{i+1:2d}. Token {token_id:5d}: {count:8,} times - [DECODE ERROR]")
    
    # Find outliers (very rare tokens)
    print(f"\n🔍 Outlier Analysis:")
    freq_threshold = 10  # Tokens used less than 10 times
    rare_tokens = [(token_id, count) for token_id, count in token_counts.items() if count < freq_threshold]
    print(f"Tokens used < {freq_threshold} times: {len(rare_tokens):,}")
    
    # Token frequency distribution
    plt.subplot(2, 2, 2)
    # Use log scale for better visualization
    plt.hist(token_freqs, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Token Frequency Distribution')
    plt.xlabel('Token Frequency')
    plt.ylabel('Number of Tokens')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # Token frequency vs rank (Zipf's law)
    plt.subplot(2, 2, 3)
    sorted_freqs = sorted(token_freqs, reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    plt.loglog(ranks, sorted_freqs, 'o-', markersize=2, alpha=0.7, color='green')
    plt.title("Token Frequency vs Rank (Zipf's Law)")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Cumulative token coverage
    plt.subplot(2, 2, 4)
    cumsum = np.cumsum(sorted_freqs)
    coverage = cumsum / total_tokens * 100
    plt.plot(ranks, coverage, color='purple', linewidth=2)
    plt.title('Cumulative Token Coverage')
    plt.xlabel('Number of Most Frequent Tokens')
    plt.ylabel('Coverage (%)')
    plt.grid(True, alpha=0.3)
    
    # Find how many tokens cover 90% and 99% of usage
    idx_90 = np.where(coverage >= 90)[0][0] + 1
    idx_99 = np.where(coverage >= 99)[0][0] + 1
    plt.axhline(y=90, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=99, color='orange', linestyle='--', alpha=0.7)
    plt.axvline(x=idx_90, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=idx_99, color='orange', linestyle='--', alpha=0.7)
    
    print(f"\n📊 Coverage Analysis:")
    print(f"Top {idx_90:,} tokens cover 90% of all usage")
    print(f"Top {idx_99:,} tokens cover 99% of all usage")
    print(f"Effective vocabulary size (90% coverage): {idx_90:,} / {vocab_size:,} ({idx_90/vocab_size*100:.1f}%)")
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Plots saved to 'dataset_analysis.png'")
    
    # Summary statistics
    print(f"\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    print(f"Samples analyzed: {samples_processed:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average sequence length: {seq_lengths.mean():.1f}")
    print(f"Vocabulary utilization: {len(used_tokens)/vocab_size*100:.1f}%")
    print(f"Unused tokens: {len(unused_tokens):,}")
    print(f"Rare tokens (< {freq_threshold} uses): {len(rare_tokens):,}")
    print(f"Effective vocab (90% coverage): {idx_90:,}")
    
    return {
        'sequence_lengths': seq_lengths,
        'token_counts': token_counts,
        'unused_tokens': unused_tokens,
        'rare_tokens': rare_tokens,
        'vocab_utilization': len(used_tokens) / vocab_size,
        'effective_vocab_90': idx_90,
        'effective_vocab_99': idx_99
    }


if __name__ == "__main__":
    # Run analysis
    results = analyze_dataset(max_samples=50000, batch_size=64)
    print("\n✅ Analysis complete!")