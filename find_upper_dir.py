#!/usr/bin/env python3
"""
Script to analyze case differences in token embeddings and perform hypothesis testing.
"""

import argparse
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
import torch as t
import numpy as np
from scipy import stats
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from common import get_embed_layer, DEFAULT_MODEL


def find_case_paired_tokens(tokenizer):
    """Find tokens that have both lowercase and uppercase versions."""
    # Loop through all tokens in the vocabulary
    alphanum_tokens = []
    for token_id in range(len(tokenizer)):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if re.match(r"^[a-zA-Z0-9 ]+$", token):
            alphanum_tokens.append(token)

    paired_tokens = []
    lowercase_tokens = set(map(lambda x: x.lower(), alphanum_tokens))
    for token_lower in lowercase_tokens:
        token_upper = token_lower.upper()
        token_lower_id = tokenizer.convert_tokens_to_ids(token_lower)
        token_upper_id = tokenizer.convert_tokens_to_ids(token_upper)
        if (token_lower_id != tokenizer.unk_token_id and 
            token_upper_id != tokenizer.unk_token_id and
            token_lower_id != token_upper_id):
            paired_tokens.append((token_lower, token_lower_id, token_upper, token_upper_id))

    print(f"Found {len(paired_tokens)} paired tokens")
    return paired_tokens


def plot_tsne(lower_x, upper_x):
    """Create t-SNE visualization of case-paired token embeddings."""
    # Combine embeddings and create labels
    all_embeddings = t.cat([lower_x, upper_x], dim=0).detach().numpy()
    labels = ['lowercase'] * len(lower_x) + ['uppercase'] * len(upper_x)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Split back into lower and upper for plotting
    lower_2d = embeddings_2d[:len(lower_x)]
    upper_2d = embeddings_2d[len(lower_x):]

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(lower_2d[:, 0], lower_2d[:, 1], c='blue', label='lowercase', alpha=0.6)
    plt.scatter(upper_2d[:, 0], upper_2d[:, 1], c='red', label='uppercase', alpha=0.6)
    plt.title('t-SNE visualization of case-paired token embeddings')
    plt.legend()
    plt.show()


def multivariate_hotelling_test(diff_tensor, alpha=0.05):
    """
    Perform Hotelling's T-squared test to test if the mean of diff_tensor is 0.
    
    Args:
        diff_tensor: tensor of shape (samples, dim)
        alpha: significance level
    
    Returns:
        dict with test results
    """
    # Convert to numpy array
    X = diff_tensor.detach().numpy()
    n, p = X.shape
    
    print(f"Sample size: {n}, Dimensions: {p}")
    
    # Check if we have enough samples relative to dimensions
    if n <= p:
        print(f"Warning: Sample size ({n}) is less than or equal to dimensions ({p})")
        print("Hotelling's T-squared test requires n > p. Using alternative approach.")
        
        # Use univariate t-tests on each dimension instead
        p_values = []
        significant_dims = []
        for i in range(p):
            t_stat, p_val = stats.ttest_1samp(X[:, i], 0)
            p_values.append(p_val)
            if p_val < alpha:
                significant_dims.append(i)
        
        # Bonferroni correction for multiple comparisons
        significant_after_correction = [i for i, p_val in enumerate(p_values) if p_val < alpha / p]
        
        results = {
            'test_type': 'univariate_t_tests',
            'p_values': p_values,
            'significant_dims': significant_dims,
            'significant_after_bonferroni': significant_after_correction,
            'sample_mean': np.mean(X, axis=0),
            'sample_cov': np.cov(X, rowvar=False)
        }
        
        print(f"\nUnivariate t-test Results (Bonferroni-corrected α={alpha/p:.6f}):")
        print(f"Significant dimensions: {len(significant_after_correction)}/{p}")
        if significant_after_correction:
            print(f"Significant dimension indices: {significant_after_correction[:10]}{'...' if len(significant_after_correction) > 10 else ''}")
        else:
            print("No dimensions are significantly different from 0 after correction")
            
        return results
    
    # Calculate sample mean and covariance
    sample_mean = np.mean(X, axis=0)
    sample_cov = np.cov(X, rowvar=False)
    
    print(f"Sample mean shape: {sample_mean.shape}")
    print(f"Sample covariance shape: {sample_cov.shape}")
    
    # Check condition number of covariance matrix
    eigenvals = np.linalg.eigvals(sample_cov)
    condition_number = np.max(np.abs(eigenvals)) / np.min(np.abs(eigenvals))
    print(f"Covariance matrix condition number: {condition_number:.2e}")
    
    if condition_number > 1e12:
        print("Warning: Covariance matrix is ill-conditioned. Adding regularization.")
        # Add small regularization to diagonal
        sample_cov += np.eye(p) * 1e-8 * np.trace(sample_cov) / p
    
    try:
        # Invert covariance matrix
        cov_inv = np.linalg.inv(sample_cov)
        
        # Calculate T-squared statistic
        T_squared = n * sample_mean.T @ cov_inv @ sample_mean
        
        # Check for numerical issues
        if not np.isfinite(T_squared) or T_squared < 0:
            print("Warning: T-squared statistic is invalid. Using alternative approach.")
            return multivariate_hotelling_test(diff_tensor, alpha)  # Recursive call with univariate approach
        
        # Convert to F-statistic
        F_stat = (n - p) / (p * (n - 1)) * T_squared
        
        # Check if F-statistic is valid
        if not np.isfinite(F_stat) or F_stat < 0:
            print("Warning: F-statistic is invalid. Using alternative approach.")
            return multivariate_hotelling_test(diff_tensor, alpha)  # Recursive call with univariate approach
        
        # Degrees of freedom
        df1, df2 = p, n - p
        
        # Calculate p-value
        p_value = 1 - stats.f.cdf(F_stat, df1, df2)
        
        # Critical value
        critical_value = stats.f.ppf(1 - alpha, df1, df2)
        
        # Decision
        reject_null = F_stat > critical_value
        
        results = {
            'test_type': 'hotelling_t_squared',
            'T_squared': T_squared,
            'F_statistic': F_stat,
            'p_value': p_value,
            'critical_value': critical_value,
            'reject_null': reject_null,
            'sample_mean': sample_mean,
            'sample_cov': sample_cov,
            'condition_number': condition_number
        }
        
        print(f"\nHotelling's T-squared Test Results:")
        print(f"T-squared statistic: {T_squared:.4f}")
        print(f"F-statistic: {F_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Critical value (α={alpha}): {critical_value:.4f}")
        print(f"Reject null hypothesis (μ=0): {reject_null}")
        
        if reject_null:
            print("✓ Reject H₀: The mean difference is significantly different from 0")
        else:
            print("✗ Fail to reject H₀: No significant evidence that mean difference differs from 0")
            
        return results
        
    except np.linalg.LinAlgError:
        print("Error: Covariance matrix is singular. Using alternative approach.")
        return multivariate_hotelling_test(diff_tensor, alpha)  # Recursive call with univariate approach


def main():
    parser = argparse.ArgumentParser(
        description="Analyze case differences in token embeddings and perform hypothesis testing"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=DEFAULT_MODEL,
        help=f"Model name to analyze (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Perform hypothesis testing on the case differences"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    
    # Load tokenizer and model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    embed_in = get_embed_layer(model)
    
    # Find case-paired tokens
    paired_tokens = find_case_paired_tokens(tokenizer)
    
    if not paired_tokens:
        print("No case-paired tokens found. Exiting.")
        return
    
    # Extract embeddings for paired tokens
    lower_ids = t.tensor([token_lower_id for token_lower, token_lower_id, token_upper, token_upper_id in paired_tokens])
    upper_ids = t.tensor([token_upper_id for token_lower, token_lower_id, token_upper, token_upper_id in paired_tokens])

    lower_x = embed_in(lower_ids)
    upper_x = embed_in(upper_ids)
    diff = upper_x - lower_x
    
    # Print debugging information
    print(f"\nEmbedding analysis:")
    print(f"Number of paired tokens: {len(paired_tokens)}")
    print(f"Embedding dimension: {lower_x.shape[1]}")
    print(f"Lower embeddings shape: {lower_x.shape}")
    print(f"Upper embeddings shape: {upper_x.shape}")
    print(f"Difference embeddings shape: {diff.shape}")
    print(f"Mean difference magnitude: {diff.abs().mean():.6f}")
    print(f"Std difference magnitude: {diff.abs().std():.6f}")
    print(f"Max difference magnitude: {diff.abs().max():.6f}")
    
    # Perform hypothesis testing if requested
    if args.test:
        print("\nPerforming multivariate hypothesis test on diff tensor...")
        test_results = multivariate_hotelling_test(diff, alpha=0.05)
    
    # Save mean differences to cache file
    cache_path = f"cache/upper_dir_{args.model.replace('/', '_')}.pt"
    
    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    
    # Calculate and save mean differences
    mean_diff = diff.mean(dim=0)
    t.save(mean_diff, cache_path)
    print(f"Saved mean differences to: {cache_path}")


if __name__ == "__main__":
    main()
