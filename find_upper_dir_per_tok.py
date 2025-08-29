#!/usr/bin/env python3
"""
Script to analyze case differences in token embeddings and record per-token directions.
"""

import argparse
import dataclasses
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
import torch as t
import numpy as np

from common import SteeringVector, get_embed_layer, DEFAULT_MODEL


def find_case_paired_tokens(tokenizer):
    """Find tokens that have both lowercase and uppercase versions."""
    # Loop through all tokens in the vocabulary
    lowercase_tokens = set()
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if re.match(r"^[Ġa-z0-9 ]+$", token):
            lowercase_tokens.add(token)

    paired_tokens = []
    for token_lower in lowercase_tokens:
        token_upper = token_lower.upper()
        token_lower_id = tokenizer.convert_tokens_to_ids(token_lower)
        token_upper_id = tokenizer.convert_tokens_to_ids(token_upper)
        if (token_lower_id != tokenizer.unk_token_id and 
            token_upper_id != tokenizer.unk_token_id and
            token_lower_id != token_upper_id):
            paired_tokens.append((token_lower, token_lower_id, token_upper, token_upper_id))

    return paired_tokens


def main():
    parser = argparse.ArgumentParser(
        description="Analyze case differences in token embeddings and record per-token directions"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL,
        help=f"Model name to analyze (default: {DEFAULT_MODEL})"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    
    # Load tokenizer and model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    embed_in = get_embed_layer(model)
    
    # Find case-paired tokens
    paired_tokens = find_case_paired_tokens(tokenizer)

    print(f"Found {len(paired_tokens)} paired tokens")
    
    if not paired_tokens:
        print("No case-paired tokens found. Exiting.")
        return
    
    # Extract embeddings for paired tokens
    lower_ids = t.tensor([token_lower_id for token_lower, token_lower_id, token_upper, token_upper_id in paired_tokens])
    upper_ids = t.tensor([token_upper_id for token_lower, token_lower_id, token_upper, token_upper_id in paired_tokens])

    lower_x = embed_in(lower_ids)
    upper_x = embed_in(upper_ids)
    diff = upper_x - lower_x
    
    # Create full vocabulary tensor filled with zeros
    vocab_size = len(tokenizer)
    embedding_dim = lower_x.shape[1]
    full_dir = t.zeros(vocab_size, embedding_dim)
    
    # Fill in the directions for paired tokens
    for i, (token_lower, token_lower_id, token_upper, token_upper_id) in enumerate(paired_tokens):
        # Store the direction from lowercase to uppercase at the lowercase token's position
        full_dir[token_lower_id] = diff[i]
    
    # Print debugging information
    print(f"\nEmbedding analysis:")
    print(f"Number of paired tokens: {len(paired_tokens)}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Full direction tensor shape: {full_dir.shape}")
    print(f"Non-zero rows: {(full_dir != 0).any(dim=1).sum().item()}")
    print(f"Mean difference magnitude: {diff.abs().mean():.6f}")
    print(f"Std difference magnitude: {diff.abs().std():.6f}")
    print(f"Max difference magnitude: {diff.abs().max():.6f}")
    
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)

    save_data = SteeringVector(
        dir=full_dir,
        model_name = args.model,
        hook='embed',
        per_tok=True,
        src = {
            'paired_tokens_count': len(paired_tokens),
        })
    
    # Save full vocabulary directions to cache file
    cache_path = f"cache/upper_dir_per_tok_{args.model.replace('/', '_')}.pt"
    save_data.save(cache_path)
    print(f"Saved full vocabulary directions to: {cache_path}")


if __name__ == "__main__":
    main() 