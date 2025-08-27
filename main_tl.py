# %%
import bisect
from functools import partial
from typing import Literal
import transformer_lens as tl
from transformer_lens import HookedTransformer
import torch as t
import re
import os
from eval_tl import generate_text, generate_logits, evaluate_preference

# %%

device = "cuda" if t.cuda.is_available() else "cpu"
# %%
# Load directions from cache
cache_path = f"cache/layer3_activations_google_gemma-2-2b-it_wikitext_wikitext-2-raw-v1.pt"
data = t.load(cache_path)
layer = data["layer"]
per_tok = False
upper_dir = data["dir"].to(device)
model_name = data["model_name"]

# %%

# Load model using TransformerLens
model = HookedTransformer.from_pretrained(model_name, device=device)



# %%
generated_text = generate_text(
    "*The meaning of life is*",
    model,
    layer,
    upper_dir,
    per_tok=per_tok,
    max_new_tokens=20,
    do_sample=False,
    # temperature=0.7,
    steering_scale=1,
    transform="uppish"
)
print(generated_text)
# %%


generated_logits = generate_logits(
    "*The meaning of life is*",
    model,
    layer,
    upper_dir,
    steering_scale=2,
    transform="uppish"
)

print(generated_logits)


# %%
pref = evaluate_preference(
    model,
    layer,
    upper_dir,
    per_tok=per_tok,
    steering_scale=-0.1,
    transform="none"
)
print(pref)
# %%

