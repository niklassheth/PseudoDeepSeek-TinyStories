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
        max_length=4096  # Allow longer sequences for analysis
    )
    
    train_loader = dataloaders['train']
    
    # Statistics containers
    sequence_lengths = []
    token_counts = Counter()
    total_tokens = 0
    samples_processed = 0
    extreme_sequences = {'very_long': [], 'very_short': []}  # Store extreme examples
    non_ascii_sequences = []  # Store sequences with non-ASCII characters
    max_examples = 3
    
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
            
            # Capture extreme sequence examples
            if seq_len > 1000 and len(extreme_sequences['very_long']) < max_examples:
                try:
                    decoded_text = tokenizer.decode(tokens)
                    extreme_sequences['very_long'].append({
                        'length': seq_len,
                        'text': decoded_text[:500] + "..." if len(decoded_text) > 500 else decoded_text
                    })
                except:
                    pass
            elif seq_len < 50 and len(extreme_sequences['very_short']) < max_examples:
                try:
                    decoded_text = tokenizer.decode(tokens)
                    extreme_sequences['very_short'].append({
                        'length': seq_len,
                        'text': decoded_text
                    })
                except:
                    pass
            
            # Check for non-ASCII characters
            if len(non_ascii_sequences) < max_examples:
                try:
                    decoded_text = tokenizer.decode(tokens)
                    # Check if text contains non-ASCII characters
                    if any(ord(char) > 127 for char in decoded_text):
                        non_ascii_sequences.append({
                            'length': seq_len,
                            'text': decoded_text[:300] + "..." if len(decoded_text) > 300 else decoded_text,
                            'non_ascii_chars': [char for char in decoded_text if ord(char) > 127][:20]  # Show first 20
                        })
                except:
                    pass
            
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
    
    # Focus on outliers with percentiles
    percentiles = [1, 5, 10, 90, 95, 99, 99.9]
    print(f"\n🎯 Outlier Analysis (Percentiles):")
    for p in percentiles:
        val = np.percentile(seq_lengths, p)
        print(f"  {p:4.1f}th percentile: {val:.0f}")
    
    # Count extreme sequences
    very_short = np.sum(seq_lengths < 50)
    very_long = np.sum(seq_lengths > 1000)
    super_long = np.sum(seq_lengths > 2000)
    print(f"\n📊 Extreme Sequence Counts:")
    print(f"  Very short (< 50 tokens): {very_short:,} ({very_short/len(seq_lengths)*100:.2f}%)")
    print(f"  Very long (> 1000 tokens): {very_long:,} ({very_long/len(seq_lengths)*100:.2f}%)")
    print(f"  Super long (> 2000 tokens): {super_long:,} ({super_long/len(seq_lengths)*100:.2f}%)")
    
    # Plotting section - Focus on outliers
    plt.figure(figsize=(16, 12))
    
    # 1. Full sequence length distribution with outlier highlights
    plt.subplot(3, 3, 1)
    plt.hist(seq_lengths, bins=100, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(512, color='red', linestyle='--', alpha=0.8, label='Current Context (512)')
    plt.axvline(np.percentile(seq_lengths, 99), color='orange', linestyle='--', alpha=0.8, label='99th percentile')
    plt.title('Full Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Zoomed view: Only very long sequences (>500)
    plt.subplot(3, 3, 2) 
    long_seqs = seq_lengths[seq_lengths > 500]
    if len(long_seqs) > 0:
        plt.hist(long_seqs, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.axvline(512, color='blue', linestyle='--', alpha=0.8, label='Context Limit')
        plt.title(f'Long Sequences (>500 tokens)\n{len(long_seqs):,} sequences')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No sequences > 500 tokens', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Long Sequences (>500 tokens)')
    plt.grid(True, alpha=0.3)
    
    # 3. Zoomed view: Very short sequences (<100)
    plt.subplot(3, 3, 3)
    short_seqs = seq_lengths[seq_lengths < 100]
    if len(short_seqs) > 0:
        plt.hist(short_seqs, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title(f'Short Sequences (<100 tokens)\n{len(short_seqs):,} sequences')
        plt.xlabel('Sequence Length')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No sequences < 100 tokens', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Short Sequences (<100 tokens)')
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
    
    # Find outliers (very rare tokens) - Multiple thresholds
    print(f"\n🔍 Rare Token Analysis:")
    thresholds = [1, 2, 5, 10, 50, 100]
    rare_token_counts = {}
    for threshold in thresholds:
        rare_tokens_t = [(token_id, count) for token_id, count in token_counts.items() if count <= threshold]
        rare_token_counts[threshold] = len(rare_tokens_t)
        print(f"  Tokens used ≤ {threshold:3d} times: {len(rare_tokens_t):6,} ({len(rare_tokens_t)/len(used_tokens)*100:5.1f}% of used tokens)")
    
    # Show some examples of single-occurrence tokens
    single_tokens = [(token_id, count) for token_id, count in token_counts.items() if count == 1]
    print(f"\n🦄 Single-occurrence tokens ({len(single_tokens):,} total):")
    for i, (token_id, count) in enumerate(single_tokens[:10]):
        try:
            token_str = tokenizer.decode([token_id])
            print(f"  {i+1:2d}. Token {token_id:5d}: '{token_str}' (repr: {repr(token_str)})")
        except:
            print(f"  {i+1:2d}. Token {token_id:5d}: [DECODE ERROR]")
    
    rare_tokens = single_tokens  # For backward compatibility
    
    # 4. Rare token frequency focus (≤100 occurrences)
    plt.subplot(3, 3, 4)
    rare_freqs = token_freqs[token_freqs <= 100]
    if len(rare_freqs) > 0:
        plt.hist(rare_freqs, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.title(f'Rare Token Frequencies (≤100)\n{len(rare_freqs):,} tokens')
        plt.xlabel('Token Frequency')
        plt.ylabel('Number of Tokens')
        plt.yscale('log')
    else:
        plt.text(0.5, 0.5, 'No rare tokens ≤100', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Rare Token Frequencies (≤100)')
    plt.grid(True, alpha=0.3)
    
    # 5. Ultra-rare tokens (≤10 occurrences)
    plt.subplot(3, 3, 5)
    ultra_rare_freqs = token_freqs[token_freqs <= 10]
    if len(ultra_rare_freqs) > 0:
        plt.hist(ultra_rare_freqs, bins=np.arange(1, 12), alpha=0.7, color='darkred', edgecolor='black')
        plt.title(f'Ultra-Rare Tokens (≤10)\n{len(ultra_rare_freqs):,} tokens')
        plt.xlabel('Token Frequency')
        plt.ylabel('Number of Tokens')
        plt.xticks(range(1, 11))
    else:
        plt.text(0.5, 0.5, 'No ultra-rare tokens ≤10', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Ultra-Rare Tokens (≤10)')
    plt.grid(True, alpha=0.3)
    
    # 6. Token frequency thresholds bar chart
    plt.subplot(3, 3, 6)
    threshold_labels = ['≤1', '≤2', '≤5', '≤10', '≤50', '≤100']
    threshold_counts = [rare_token_counts[t] for t in [1, 2, 5, 10, 50, 100]]
    bars = plt.bar(threshold_labels, threshold_counts, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Tokens by Frequency Threshold')
    plt.xlabel('Frequency Threshold')
    plt.ylabel('Number of Tokens')
    plt.yscale('log')
    # Add value labels on bars
    for bar, count in zip(bars, threshold_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 7. Zipf's law - focus on long tail
    plt.subplot(3, 3, 7)
    sorted_freqs = sorted(token_freqs, reverse=True)
    ranks = np.arange(1, len(sorted_freqs) + 1)
    plt.loglog(ranks, sorted_freqs, 'o-', markersize=1, alpha=0.7, color='green')
    plt.title("Zipf's Law - Full Distribution")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    # Highlight the long tail
    tail_start = len(sorted_freqs) // 2
    plt.loglog(ranks[tail_start:], sorted_freqs[tail_start:], 'o-', markersize=1, alpha=0.9, color='red', label='Long tail')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Cumulative coverage with rare token focus
    plt.subplot(3, 3, 8)
    cumsum = np.cumsum(sorted_freqs)
    coverage = cumsum / total_tokens * 100
    plt.plot(ranks, coverage, color='purple', linewidth=2)
    plt.title('Cumulative Token Coverage')
    plt.xlabel('Number of Most Frequent Tokens')
    plt.ylabel('Coverage (%)')
    
    # Find coverage thresholds
    idx_90 = np.where(coverage >= 90)[0][0] + 1
    idx_99 = np.where(coverage >= 99)[0][0] + 1
    plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='90%')
    plt.axhline(y=99, color='orange', linestyle='--', alpha=0.7, label='99%')
    plt.axvline(x=idx_90, color='red', linestyle='--', alpha=0.7)
    plt.axvline(x=idx_99, color='orange', linestyle='--', alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Vocabulary utilization pie chart
    plt.subplot(3, 3, 9)
    unused_count = len(unused_tokens)
    used_count = len(used_tokens)
    single_count = len(single_tokens)
    rare_count = rare_token_counts[10] - single_count  # 2-10 occurrences
    common_count = used_count - rare_count - single_count
    
    sizes = [unused_count, single_count, rare_count, common_count]
    labels = [f'Unused\n{unused_count:,}', f'Single-use\n{single_count:,}', 
              f'Rare (2-10)\n{rare_count:,}', f'Common (>10)\n{common_count:,}']
    colors = ['lightgray', 'red', 'orange', 'green']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Vocabulary Utilization')
    
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
    print(f"Single-occurrence tokens: {len(single_tokens):,}")
    print(f"Effective vocab (90% coverage): {idx_90:,}")
    
    # Show extreme sequence examples
    print(f"\n" + "="*50)
    print("📏 EXTREME SEQUENCE EXAMPLES")
    print("="*50)
    
    if extreme_sequences['very_long']:
        print(f"\n🔍 Very Long Sequences (>1000 tokens):")
        for i, seq in enumerate(extreme_sequences['very_long']):
            print(f"\n  Example {i+1} (Length: {seq['length']} tokens):")
            print(f"  {'-'*60}")
            print(f"  {seq['text']}")
            print(f"  {'-'*60}")
    else:
        print(f"\n📏 No sequences >1000 tokens found in sample")
    
    if extreme_sequences['very_short']:
        print(f"\n🔍 Very Short Sequences (<50 tokens):")
        for i, seq in enumerate(extreme_sequences['very_short']):
            print(f"\n  Example {i+1} (Length: {seq['length']} tokens):")
            print(f"  {'-'*60}")
            print(f"  {seq['text']}")
            print(f"  {'-'*60}")
    else:
        print(f"\n📏 No sequences <50 tokens found in sample")
    
    # Show non-ASCII character examples
    if non_ascii_sequences:
        print(f"\n" + "="*50)
        print("🌍 NON-ASCII CHARACTER EXAMPLES")
        print("="*50)
        print(f"\n🔍 Sequences with Non-ASCII Characters:")
        for i, seq in enumerate(non_ascii_sequences):
            print(f"\n  Example {i+1} (Length: {seq['length']} tokens):")
            print(f"  Non-ASCII chars found: {seq['non_ascii_chars']}")
            print(f"  {'-'*60}")
            print(f"  {seq['text']}")
            print(f"  {'-'*60}")
    else:
        print(f"\n" + "="*50)
        print("🌍 NON-ASCII CHARACTER ANALYSIS")
        print("="*50)
        print(f"\n✅ No non-ASCII characters found in analyzed samples")
    
    return {
        'sequence_lengths': seq_lengths,
        'token_counts': token_counts,
        'unused_tokens': unused_tokens,
        'rare_tokens': rare_tokens,
        'single_tokens': single_tokens,
        'rare_token_counts': rare_token_counts,
        'extreme_sequences': extreme_sequences,
        'non_ascii_sequences': non_ascii_sequences,
        'vocab_utilization': len(used_tokens) / vocab_size,
        'effective_vocab_90': idx_90,
        'effective_vocab_99': idx_99
    }


if __name__ == "__main__":
    # Run analysis
    results = analyze_dataset(max_samples=float("inf"), batch_size=64)
    print("\n✅ Analysis complete!")