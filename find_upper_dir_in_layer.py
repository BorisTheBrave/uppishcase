#!/usr/bin/env python3
"""
Script to load a HuggingFace dataset and model, then loop over the dataset 
to record average token activations.
"""

# %%
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
import torch as t
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

from common import DEFAULT_MODEL, get_layer

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
def process_batch(tokenizer, model, batch_texts: list[str], layer_num: int):
    """
    Process a batch of texts by selecting random words, truncating, and capitalizing.
    
    Args:
        tokenizer: Model tokenizer
        model: Model (not used in current implementation)
        batch_texts: List of text strings to process
        
    Returns:
        list of processed text strings
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
        final_batch.append(truncated_text)
        
        # Capitalize the selected word and create second version
        capitalized_words = words.copy()
        capitalized_words[random_word_idx] = words[random_word_idx].upper()
        capitalized_text = " ".join(capitalized_words[:random_word_idx + 1])
        final_batch.append(capitalized_text)

    

    encoded = tokenizer(
        batch_texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        # max_length=512
    )

    device = model.device
    
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)



    
    device = model.device
    target_layer = get_layer(model, layer_num)
    layer_activations = []
    
    def activation_hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        layer_activations.append(output.detach().cpu())
    
    hook_handle = target_layer.register_forward_hook(activation_hook)
    
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        hook_handle.remove()

    last_nonzero = (attention_mask != 0).max(dim=1).indices
    last_nonzero = t.maximum(last_nonzero[::2], last_nonzero[1::2])

    activations = []
    for i in range(len(final_batch)):
        upper_act = layer_activations[i*2][last_nonzero[i]]
        normal_act = layer_activations[i*2+1][last_nonzero[i]]
        activations.append(upper_act - normal_act)

    activations = t.stack(activations)
    
    return activations

# %%
def process_dataset_activations(dataset, tokenizer, model, layer_num=3, max_samples=None, batch_size=8):
    """
    Process dataset samples and compute average token activations.
    
    Args:
        dataset: HuggingFace dataset
        tokenizer: Model tokenizer
        model: Model for computing activations
        layer_num: Layer number to analyze
        max_samples: Maximum number of samples to process (None for all)
        batch_size: Batch size for processing
        
    Returns:
        dict with activation statistics
    """
    device = model.device
    target_layer = get_layer(model, layer_num)
    
    total_activation = 0.0
    total_tokens = 0
    activation_sums = None
    num_batches = 0
    layer_activations = []
    
    # Set up hook to capture layer activations
    def activation_hook(module, input, output):
        layer_activations.append(output.detach().cpu())
    
    hook_handle = target_layer.register_forward_hook(activation_hook)
    
    # Determine number of samples to process
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    print(f"Processing {num_samples} samples from dataset using layer {layer_num}...")
    
    model.eval()
    try:
        with t.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, num_samples)
                
                # Extract text from batch using the dedicated function
                batch_texts = extract_text_from_batch(dataset, i, batch_end)

                x = process_batch(tokenizer, model, batch_texts, layer_num)
                
                # Tokenize batch
                try:
                    encoded = tokenizer(
                        batch_texts, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512
                    )
                    
                    input_ids = encoded.input_ids.to(device)
                    attention_mask = encoded.attention_mask.to(device)
                    
                    # Clear previous activations
                    layer_activations.clear()
                    
                    # Forward pass to trigger hook
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    
                    # Get the captured activations
                    if layer_activations:
                        activations = layer_activations[0].to(device)
                        
                        # Apply attention mask to exclude padding tokens
                        masked_activations = activations * attention_mask.unsqueeze(-1)
                        
                        # Compute activation statistics
                        batch_activation = masked_activations.sum().item()
                        batch_tokens = attention_mask.sum().item()
                        
                        total_activation += batch_activation
                        total_tokens += batch_tokens
                        
                        # Accumulate activation sums for computing mean direction
                        batch_activation_sum = masked_activations.sum(dim=(0, 1))  # Sum over batch and sequence
                        
                        if activation_sums is None:
                            activation_sums = batch_activation_sum
                        else:
                            activation_sums += batch_activation_sum
                        
                        num_batches += 1
                    
                except Exception as e:
                    print(f"Error processing batch {i//batch_size}: {e}")
                    continue
    finally:
        # Always remove the hook
        hook_handle.remove()
    
    if total_tokens == 0:
        print("Warning: No tokens processed successfully")
        # Try to get embedding dimension from model
        try:
            from common import get_embed_layer
            embed_layer = get_embed_layer(model)
            embedding_dim = embed_layer.embedding_dim
        except:
            embedding_dim = 768  # fallback default
        
        return {
            'average_activation': 0.0,
            'total_tokens': 0,
            'mean_embedding': t.zeros(embedding_dim),
            'num_batches': num_batches,
            'embedding_dim': embedding_dim
        }
    
    # Compute final statistics
    average_activation = total_activation / total_tokens
    mean_embedding = activation_sums / total_tokens
    
    results = {
        'average_activation': average_activation,
        'total_tokens': total_tokens,
        'mean_embedding': mean_embedding,
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
        default="databricks/databricks-dolly-15k",
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
    
    # Load tokenizer and model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    device = "cuda" if t.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"Loading dataset: {args.dataset}")
    if args.dataset_config:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Process dataset and compute activations
    results = process_dataset_activations(
        dataset, 
        tokenizer, 
        model, 
        layer_num=args.layer,
        max_samples=args.max_samples,
        batch_size=args.batch_size
    )
    
    # Print results
    print(f"\nActivation Analysis Results (Layer {args.layer}):")
    print(f"Total tokens processed: {results['total_tokens']:,}")
    print(f"Number of batches: {results['num_batches']}")
    print(f"Embedding dimension: {results['embedding_dim']}")
    print(f"Average token activation: {results['average_activation']:.6f}")
    print(f"Mean activation magnitude: {results['mean_embedding'].norm().item():.6f}")
    print(f"Mean activation shape: {results['mean_embedding'].shape}")
    
    # Save results to cache
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    model_name_safe = args.model.replace('/', '_')
    dataset_name_safe = f"{args.dataset}_{args.dataset_config}".replace('/', '_')
    
    cache_path = f"{cache_dir}/layer{args.layer}_activations_{model_name_safe}_{dataset_name_safe}.pt"
    
    # Save comprehensive results
    save_data = {
        'average_activation': results['average_activation'],
        'mean_embedding': results['mean_embedding'],
        'total_tokens': results['total_tokens'],
        'embedding_dim': results['embedding_dim'],
        'layer_num': args.layer,
        'model_name': args.model,
        'dataset_name': args.dataset,
        'dataset_config': args.dataset_config,
        'max_samples': args.max_samples
    }
    
    t.save(save_data, cache_path)
    print(f"\nSaved results to: {cache_path}")


if __name__ == "__main__":
    pass
    # main()


# %%
import sys
sys.exit()


# %%

from common import get_embed_layer, DEFAULT_MODEL
# %%

model_name = DEFAULT_MODEL
device = "cuda" if t.cuda.is_available() else "cpu"
PER_TOK = False
# %%

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# %%

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
batch_texts = extract_text_from_batch(dataset, 0, 1)
# %%

x = process_batch(tokenizer, model, batch_texts, 3)
# %%

x.shape
# %%

x.norm(dim=1)