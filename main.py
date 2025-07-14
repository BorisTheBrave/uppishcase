# %%
import bisect
from functools import partial
from typing import Literal
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.tokenization_utils import PreTrainedTokenizer
import torch as t
import re
import os

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
# All transforms return a tuple of (tokens, mask, steering)

def none_transform(texts: list[str]) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    toks = tokenizer(texts, return_tensors="pt")
    
    return (
        toks.input_ids,
        toks.attention_mask,
        t.zeros_like(toks.input_ids)
    )

def UPPER_transform(texts: list[str]) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    toks = tokenizer(list(map(lambda x: x.upper(), texts)), return_tensors="pt")
    
    return (
        toks.input_ids,
        toks.attention_mask,
        t.zeros_like(toks.input_ids)
    )

def uppish_transform(
        texts: list[str], 
        case: Literal["lower", "upper"] = "lower",
        multiplier: float = 1.0) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
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

def generate_text(text: str, transform: Transform = "none", steering_scale: float = 1.0, **kwargs):

    transform_fn = transforms[transform] if transform is not None else none_transform

    inputs, mask, steering = transform_fn([text])

    inputs = inputs.to(device)
    mask = mask.to(device)
    steering = steering.to(device)

    is_first = True
    def hook(module, input, output):
        nonlocal is_first
        if is_first:
            assert input[0].shape == inputs.shape
            is_first = False
            if PER_TOK:
                # upper_dir has shape (vocab_size, embedding_dim)
                # Index it with the input token IDs
                token_dirs = upper_dir[inputs]
                output += steering.unsqueeze(-1) * token_dirs * steering_scale
            else:
                # upper_dir has shape (embedding_dim,)
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

def generate_logits(text: str, transform: Transform = "none", steering_scale: float = 1.0):

    transform_fn = transforms[transform] if transform is not None else none_transform

    inputs, mask, steering = transform_fn([text])

    inputs = inputs.to(device)
    mask = mask.to(device)
    steering = steering.to(device)

    is_first = True
    def hook(module, input, output):
        nonlocal is_first
        if is_first:
            assert input[0].shape == inputs.shape
            is_first = False
            if PER_TOK:
                # upper_dir has shape (vocab_size, embedding_dim)
                # Index it with the input token IDs
                token_dirs = upper_dir[inputs]
                output += steering.unsqueeze(-1) * token_dirs * steering_scale
            else:
                # upper_dir has shape (embedding_dim,)
                output += steering.unsqueeze(-1) * upper_dir * steering_scale
        return output
    
    handle = get_embed_layer(model).register_forward_hook(hook)

    try:
        with t.no_grad():
            outputs = model(inputs, attention_mask=mask)
    finally:
        handle.remove()

    return outputs.logits

# %%
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


generated_logits = generate_logits(
    "*The meaning of life is*",
    steering_scale=2,
    transform="uppish"
)

print(generated_logits)


# %%

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

a_token_id = tokenizer.encode(" A")[0]
b_token_id = tokenizer.encode(" B")[0]

for message_list in messages:
    formatted = tokenizer.apply_chat_template(message_list, tokenize=False)
    assert isinstance(formatted, str)
    logits = generate_logits(
        formatted,
        steering_scale=-1.0,
        transform="uppish"
    )
    a_pref = logits[0, -1, a_token_id] - logits[0, -1, b_token_id]
    print(a_pref.item())


# %%
