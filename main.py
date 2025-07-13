# %%
import bisect
from functools import partial
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
import torch as t
import re
import os

from common import get_embed_layer
# %%
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load mean differences from cache
cache_path = f"cache/upper_dir_{model_name.replace('/', '_')}.pt"
upper_dir = t.load(cache_path)

# %%
# All transforms return a tuple of (tokens, mask, steering)

def none_transform(texts: list[str]):
    toks = tokenizer(texts, return_tensors="pt")
    
    return (
        toks.input_ids,
        toks.attention_mask,
        t.zeros_like(toks.input_ids)
    )

def UPPER_transform(texts: list[str]):
    """Returns a tuple of (tokens, mask, upper_dir_steering)"""
    toks = tokenizer(list(map(lambda x: x.upper(), texts)), return_tensors="pt")
    
    return (
        toks.input_ids,
        toks.attention_mask,
        t.zeros_like(toks.input_ids)
    )

def uppish_transform(
        texts: list[str], 
        case: Literal["lower", "upper"] = "lower",
        multiplier: float = 1.0):
    
    lower_texts = []
    star_indexes = []
    for text in texts:
        l = text.lower() if case == "lower" else text.upper()
        arr = l.split("*")
        l = ""
        indexes = []
        i = 0
        for a in arr:
            l += a
            i += len(a)
            indexes.append(i)

        lower_texts.append(l)
        star_indexes.append(indexes)

    toks = tokenizer(lower_texts, return_tensors="pt", return_offsets_mapping=True)

    uppish_steering = t.zeros_like(toks.input_ids, dtype=t.float32)
    for b in range(len(texts)):
        for i in range(len(toks[b])):
            start, stop = toks.offset_mapping[b][i]
            # Count number of starred characters in tok_text
            # We filter to only alpha numeric
            is_star_count = 0
            count = 0
            for j in range(start, stop):
                if not lower_texts[b][j].isalnum():
                    continue
                # Find the position of j in star_indexes[b] using bisect
                pos = bisect.bisect_left(star_indexes[b], j+1)
                # If pos is even, this character is not starred
                if pos % 2 != 0:
                    is_star_count += 1
                count += 1
            if count > 0:
                uppish_steering[b, i] = is_star_count / count
            else:
                uppish_steering[b, i] = 0

    return (
        toks.input_ids,
        toks.attention_mask,
        uppish_steering * multiplier
    )

Transform = Literal["none", "UPPER", "uppish", "UPPISH"]

transforms = {
    "none": none_transform,
    "UPPER": UPPER_transform,
    "uppish": partial(uppish_transform, case="lower", multiplier=1.0),
    "UPPISH": partial(uppish_transform, case="upper", multiplier=1.0),
}

#%%

def generate_text(text: str, transform: Transform = "none", steering_scale: float = 1.0, **kwargs):

    transform_fn = transforms[transform] if transform is not None else none_transform

    inputs, mask, steering = transform_fn([text])

    is_first = True
    def hook(module, input, output):
        nonlocal is_first
        if is_first:
            assert input[0].shape == inputs.shape
            is_first = False
            output += steering.unsqueeze(-1) * upper_dir * steering_scale
        return output
    
    handle = get_embed_layer(model).register_forward_hook(hook)

    try:
        with t.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=mask,
                **kwargs,
                pad_token_id=tokenizer.pad_token_id
            )
    finally:
        handle.remove()

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


generated_text = generate_text(
    "*The meaning of life is*",
    max_new_tokens=20,
    do_sample=False,
    # temperature=0.7,
    steering_scale=2,
    transform="uppish"
)
print(generated_text)

# %%



# %%
