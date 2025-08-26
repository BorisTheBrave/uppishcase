# %%
import bisect
from functools import partial
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
import torch as t
import re
import os
from eval import generate_text, generate_logits, evaluate_preference

from common import get_embed_layer, DEFAULT_MODEL
# %%

model_name = DEFAULT_MODEL
device = "cuda" if t.cuda.is_available() else "cpu"
PER_TOK = False
# %%

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# %%
# Load directions from cache
if PER_TOK:
    cache_path = f"cache/upper_dir_per_tok_{model_name.replace('/', '_')}.pt"
else:
    cache_path = f"cache/upper_dir_{model_name.replace('/', '_')}.pt"
upper_dir = t.load(cache_path).to(device)


# %%
generated_text = generate_text(
    "*The meaning of life is*",
    tokenizer,
    model,
    upper_dir,
    max_new_tokens=20,
    do_sample=False,
    # temperature=0.7,
    steering_scale=2,
    transform="uppish"
)
print(generated_text)


# %%


generated_logits = generate_logits(
    "*The meaning of life is*",
    tokenizer,
    model,
    upper_dir,
    steering_scale=2,
    transform="uppish"
)

print(generated_logits)


# %%
pref = evaluate_preference(
    tokenizer,
    model,
    upper_dir,
    per_tok=PER_TOK,
    steering_scale=2,
    transform="uppish"
)
print(pref)
# %%