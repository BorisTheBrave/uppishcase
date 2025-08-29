#!/usr/bin/env python3
"""
Script to load a HuggingFace dataset and model, then loop over the dataset 
to record average token activations using TransformerLens.
"""

# %%
import argparse
import dataclasses
import os
from transformer_lens import HookedTransformer
import torch as t
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from common import DEFAULT_MODEL, SteeringVector

# %%
def extract_text_from_batch(dataset, start_idx, end_idx):
    """
    Extract text from a batch of dataset samples, handling different dataset formats.
    
    Args:
        dataset: HuggingFace dataset
        start_idx: Starting index for the batch
        end_idx: Ending index for the batch
        
    Returns:
        list of text strings
    """
    batch_texts = []
    
    for j in range(start_idx, end_idx):
        sample = dataset[j]
        # Handle different dataset formats
        if 'text' in sample:
            text = sample['text']
        elif 'content' in sample:
            text = sample['content']
        elif 'instruction' in sample:
            text = sample['instruction']
        elif isinstance(sample, str):
            text = sample
        else:
            # Try to find any string field
            text_fields = [v for v in sample.values() if isinstance(v, str)]
            text = text_fields[0] if text_fields else str(sample)
        
        batch_texts.append(text)
    
    return batch_texts

# %%
def process_batch(model, batch_texts: list[str], layer_num: int):
    """
    Process a batch of texts by selecting random words, truncating, and capitalizing.
    
    Args:
        model: TransformerLens HookedTransformer model
        batch_texts: List of text strings to process
        layer_num: Layer number to analyze
        
    Returns:
        Tensor of activation differences between uppercase and normal versions
    """
    import random
    
    final_batch = []
    
    for text in batch_texts:
        # Split text by spaces to get words
        words = text.split(" ")
        
        # Skip empty texts or texts with no words
        if not words or len(words) == 0:
            continue
            
        # Pick a random word index
        random_word_idx = random.randint(0, len(words) - 1)
        
        # Truncate everything after the random word (inclusive of the word)
        truncated_text = " ".join(words[:random_word_idx + 1])

        # Capitalize the selected word and create second version
        capitalized_words = words.copy()
        capitalized_words[random_word_idx] = words[random_word_idx].upper()
        capitalized_text = " ".join(capitalized_words[:random_word_idx + 1])

        if truncated_text == capitalized_text:
            continue

        final_batch.append(truncated_text)
        final_batch.append(capitalized_text)

    if not final_batch:
        return t.empty(0, model.cfg.d_model)

    # Tokenize the batch
    tokens = model.to_tokens(final_batch, prepend_bos=True)
    
    # Run the model to get activations at the specified layer
    _, cache = model.run_with_cache(tokens)
    
    # Get activations from the specified layer
    layer_key = f"blocks.{layer_num}.hook_resid_post"
    layer_activations = cache[layer_key]  # Shape: [batch_size, seq_len, d_model]
    
    # Find the last non-padding token for each sequence
    last_nonzero = (tokens != model.tokenizer.pad_token_id).sum(dim=1) - 1
    
    # For pairs, use the maximum sequence length between normal and uppercase versions
    last_nonzero = t.minimum(last_nonzero[::2], last_nonzero[1::2])

    activations = []
    for i in range(len(final_batch) // 2):
        normal_act = layer_activations[i*2, last_nonzero[i]]  # Normal version
        upper_act = layer_activations[i*2+1, last_nonzero[i]]  # Uppercase version
        activations.append(upper_act - normal_act)

    if activations:
        activations = t.stack(activations)
    else:
        activations = t.empty(0, model.cfg.d_model)
    
    return activations

# %%
def process_dataset_activations(dataset, model, layer_num=3, max_samples=None, batch_size=8):
    """
    Process dataset samples and compute average token activations.
    
    Args:
        dataset: HuggingFace dataset
        model: TransformerLens HookedTransformer model
        layer_num: Layer number to analyze
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Batch size for processing
        
    Returns:
        dict with activation statistics
    """
    
    total_samples = 0
    activation_sums = None
    num_batches = 0
    
    # Determine number of samples to process
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Processing {num_samples} samples from dataset using layer {layer_num}...")
    
    with t.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, num_samples)
            
            # Extract text from batch using the dedicated function
            batch_texts = extract_text_from_batch(dataset, i, batch_end)

            try:
                # Process batch to get activation differences
                activation_diffs = process_batch(model, batch_texts, layer_num)
                
                if activation_diffs.numel() > 0:
                    # Compute activation statistics
                    batch_samples = activation_diffs.shape[0]
                    
                    total_samples += batch_samples
                    
                    # Accumulate activation sums for computing mean direction
                    batch_activation_sum = activation_diffs.sum(dim=0)  # Sum over batch
                    
                    if activation_sums is None:
                        activation_sums = batch_activation_sum
                    else:
                        activation_sums += batch_activation_sum
                    
                    num_batches += 1
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {e}")
                continue
    
    # Compute final statistics
    mean_embedding = activation_sums / total_samples
    
    results = {
        'total_samples': total_samples,
        'dir': mean_embedding,
        'num_batches': num_batches,
        'embedding_dim': mean_embedding.shape[0]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Load HF dataset and model, compute average token activations"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL,
        help=f"Model name to analyze (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="HuggingFace dataset name (default: wikitext)"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration (default: databricks/databricks-dolly-15k)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to process (default: 1000)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=3,
        help="Layer number to analyze (default: 3)"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    
    # Load model using TransformerLens
    device = "cuda" if t.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(args.model, device=device)
    
    print(f"Loading dataset: {args.dataset}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Process dataset and compute activations
    results = process_dataset_activations(
        dataset, 
        model, 
        layer_num=args.layer,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Print results
    print(f"\nActivation Analysis Results (Layer {args.layer}):")
    print(f"Total samples processed: {results['total_samples']:,}")
    print(f"Number of batches: {results['num_batches']}")
    print(f"Embedding dimension: {results['embedding_dim']}")
    print(f"Mean activation magnitude: {results['dir'].norm().item():.6f}")
    print(f"Mean activation shape: {results['dir'].shape}")
    
    # Save results to cache
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    model_name_safe = args.model.replace('/', '_')
    dataset_name_safe = f"{args.dataset}_{args.dataset_config}".replace('/', '_')
    
    cache_path = f"{cache_dir}/layer{args.layer}_activations_{model_name_safe}_{dataset_name_safe}.pt"
    
    # Save comprehensive results
    save_data = SteeringVector(
        dir=results['dir'],
        layer=args.layer,
        model_name = args.model,
        src = {
            'total_samples': results['total_samples'],
            'dataset_name': args.dataset,
            'dataset_config': args.dataset_config,
            'max_samples': args.max_samples
        })
    
    t.save( dataclasses.asdict(save_data), cache_path)
    print(f"\nSaved results to: {cache_path}")


# %%
if __name__ == "__main__":
    main()


# %%
import sys
sys.exit()


# %%

from common import DEFAULT_MODEL
# %%

model_name = DEFAULT_MODEL
device = "cuda" if t.cuda.is_available() else "cpu"
PER_TOK = False
# %%

model = HookedTransformer.from_pretrained(model_name, device=device)

# %%

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
batch_texts = extract_text_from_batch(dataset, 0, 1)
# %%

x = process_batch(model, batch_texts, 3)
# %%

x.shape
# %%

x.norm(dim=1)