# %%
import bisect
from functools import partial
from typing import Literal
import transformer_lens as tl
from transformer_lens import HookedTransformer
import torch as t
import re
import os
from transformer_lens.utils import get_act_name
# %%
# All transforms return a tuple of (tokens, mask, steering)

def none_transform(texts: list[str], tokenizer) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    toks = tokenizer(texts, return_tensors="pt")
    
    return (
        toks.input_ids,
        toks.attention_mask,
        t.zeros_like(toks.input_ids)
    )

def UPPER_transform(texts: list[str], tokenizer) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    toks = tokenizer(list(map(lambda x: x.upper(), texts)), return_tensors="pt")
    
    return (
        toks.input_ids,
        toks.attention_mask,
        t.zeros_like(toks.input_ids)
    )

def uppish_transform(
        texts: list[str], 
        tokenizer,
        case: Literal["lower", "upper"] = "lower",
        multiplier: float = 1.0,
        ) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """
    Identifies any text marked between * characters as steering 1.0, otherwise 0.0.
    The text itself is split of * characters and lowercased/uppercased.
    """
    
    lower_texts = []
    star_indexes = []
    for text in texts:
        arr = text.split("*")
        l = ""
        indexes = []
        i = 0
        for a in arr:
            if len(indexes) % 2 == 1:
                a = a.lower() if case == "lower" else a.upper()
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
    "UPPER": partial(uppish_transform, case="upper", multiplier=0.0),
    "uppish": partial(uppish_transform, case="lower", multiplier=1.0),
    "UPPISH": partial(uppish_transform, case="upper", multiplier=1.0),
}

#%%

def generate_text(text: str,
    model: HookedTransformer,
    layer: int,
    upper_dir: t.Tensor,
    per_tok: bool = False,
    transform: Transform = "none",
    steering_scale: float = 1.0,
    **kwargs
):
    device = model.cfg.device

    transform_fn = transforms[transform] if transform is not None else none_transform

    inputs, mask, steering = transform_fn([text], model.tokenizer)

    inputs = inputs.to(device)
    mask = mask.to(device)
    steering = steering.to(device)

    is_first = True
    def hook(resid_post, hook):
        nonlocal is_first
        if is_first:
            is_first = False
            if per_tok:
                # upper_dir has shape (vocab_size, embedding_dim)
                # Index it with the input token IDs
                token_dirs = upper_dir[inputs]
                resid_post += steering.unsqueeze(-1) * token_dirs * steering_scale
            else:
                # upper_dir has shape (embedding_dim,)
                resid_post = resid_post + steering.unsqueeze(-1) * upper_dir * steering_scale
        return resid_post
    
    # Use TransformerLens hook system
    act_layer = get_act_name("resid_post", layer)
    # act_layer = "hook_embed"

    inputs[mask == 0] = model.tokenizer.pad_token_id

    with model.hooks(fwd_hooks=[(act_layer, hook)]):
        with t.no_grad():
            outputs = model.generate(
                inputs,
                **kwargs,
            )

    # Decode the generated text
    generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def generate_logits(text: str,
    model: HookedTransformer,
    layer: int,
    upper_dir: t.Tensor,
    per_tok: bool = False,
    transform: Transform = "none",
    steering_scale: float = 1.0):

    device = model.cfg.device

    transform_fn = transforms[transform] if transform is not None else none_transform

    inputs, mask, steering = transform_fn([text], model.tokenizer)

    inputs = inputs.to(device)
    mask = mask.to(device)
    steering = steering.to(device)

    is_first = True
    def hook(resid_post, hook):
        nonlocal is_first
        if is_first:
            is_first = False
            if per_tok:
                # upper_dir has shape (vocab_size, embedding_dim)
                # Index it with the input token IDs
                token_dirs = upper_dir[inputs]
                resid_post += steering.unsqueeze(-1) * token_dirs * steering_scale
            else:
                # upper_dir has shape (embedding_dim,)
                resid_post += steering.unsqueeze(-1) * upper_dir * steering_scale
        return resid_post
    
    # Use TransformerLens hook system   
    act_layer = get_act_name("resid_post", layer)
    # act_layer = "hook_embed"

    inputs[mask == 0] = model.tokenizer.pad_token_id

    with model.hooks(fwd_hooks=[(act_layer, hook)]):
        with t.no_grad():
            outputs = model(inputs)

    return outputs  # Return logits from the cache
    
def evaluate_preference(
    model: HookedTransformer,
    layer: int,
    upper_dir: t.Tensor,
    per_tok: bool = False,
    steering_scale: float = -1.0,
    transform: str = "uppish"):
    """
    Evaluate preference between A and B options using steering.
    
    Args:
        steering_scale: Scale factor for steering direction
        transform: Transform type to apply
        
    Returns:
        List of preference scores (A preference minus B preference)
    """
    messages = [
        [{"role": "user", "content": """Please choose one of A or B, abiding by the following rules to the best of your ability.
A *is the correct answer*.
B is the correct answer.

Respond in the following format:
ANSWER: <answer>
"""}, {"role": "assistant", "content": "ANSWER:"}],
        [{"role": "user", "content": """Please choose one of A or B, abiding by the following rules to the best of your ability.
A is the correct answer.
B *is the correct answer*.

Respond in the following format:
ANSWER: <answer>
"""}, {"role": "assistant", "content": "ANSWER:"}],
    ]

    a_token_id = model.tokenizer.encode(" A", add_special_tokens=False)[0]
    b_token_id = model.tokenizer.encode(" B", add_special_tokens=False)[0]

    preferences = []
    for message_list in messages:
        formatted = model.tokenizer.apply_chat_template(message_list, tokenize=False)
        assert isinstance(formatted, str)
        logits = generate_logits(
            formatted,
            model=model,
            layer=layer,
            upper_dir=upper_dir,
            per_tok=per_tok,
            steering_scale=steering_scale,
            transform=transform
        )
        a_pref = logits[0, -1, a_token_id] - logits[0, -1, b_token_id]
        preferences.append(a_pref.item())
    
    return preferences

# %%
