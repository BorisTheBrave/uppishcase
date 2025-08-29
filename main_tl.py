# %%
import glob
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer
import torch as t
from common import SteeringVector
from eval_tl import generate_text, generate_logits, evaluate_preference

# %%

device = "cuda" if t.cuda.is_available() else "cpu"
# %%
# Load directions from cache
cache_path = f"cache/layer0_activations_google_gemma-2-2b-it_wikitext_wikitext-2-raw-v1.pt"
# cache_path = f"cache/upper_dir_google_gemma-2-2b-it.pt"
sv = SteeringVector.load(cache_path)
sv.dir = sv.dir.to(device)
model_name = sv.model_name

# %%

# Load model using TransformerLens
model = HookedTransformer.from_pretrained(model_name, device=device)



# %%
generated_text = generate_text(
    "*The meaning of life is*",
    model,
    sv,
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
    model,
    sv,
    steering_scale=2,
    transform="uppish"
)

print(generated_logits)


# %%
pref = evaluate_preference(
    model,
    sv,
    steering_scale=-0.1,
    transform="none"
)
print(pref)
# %%

# Initialize a list to store the preference scores

def evaluate_preference_at_scale(model, sv, steering_scales):
    a_preference_scores = []
    b_preference_scores = []
    baseline = None

    for scale in tqdm(steering_scales):
        # The transform is set to "none" as in the example
        pref = evaluate_preference(
            model,
            sv,
            steering_scale=scale,
            transform="uppish"
        )
        # evaluate_preference returns a list of preference scores.
        # We'll store the first one (corresponding to the first message in the prompt).
        a_preference_scores.append(pref[0])
        b_preference_scores.append(pref[1])
        if t.isclose(t.tensor(scale), t.tensor(0.0), atol=1e-6):
            baseline = pref[0]

    if baseline is not None:
        for i in range(len(a_preference_scores)):
            a_preference_scores[i] = a_preference_scores[i] - baseline
            b_preference_scores[i] = b_preference_scores[i] - baseline

    return a_preference_scores, b_preference_scores

# %%
import matplotlib.pyplot as plt

def plot_preference_at_scale(a_preference_scores, b_preference_scores, steering_scales):
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(steering_scales, a_preference_scores, marker='o', linestyle='-', color='skyblue', label='A')
    plt.plot(steering_scales, b_preference_scores, marker='o', linestyle='-', color='orange', label='B')
    plt.xlabel("Steering Scale")
    plt.ylabel("Logits")
    plt.title("Model Preference for 'A' vs 'B' Across Steering Scales - " + sv.src.get('description', ''))
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8) # Add a line at y=0 for reference
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8) # Add a line at x=0 for reference
    plt.show()

def plot_preference_at_scale_all(model, sv):

    # Loop through each steering_scale and run evaluate_preference
    if sv.hook == 'embed':
        steering_scales = t.linspace(-200.0, 200.0, 51).tolist()
    else:
        steering_scales = t.linspace(-2.0, 2.0, 51).tolist()

    a_preference_scores, b_preference_scores = evaluate_preference_at_scale(model, sv, steering_scales)
    plot_preference_at_scale(a_preference_scores, b_preference_scores, steering_scales)

plot_preference_at_scale_all(model, sv)

# %%
caches = glob.glob("cache/layer*.pt")
from natsort import natsorted
caches = natsorted(caches)




for path in tqdm(caches):
    sv = SteeringVector.load(path)
    sv.dir = sv.dir.to(device)
    if model_name != sv.model_name:
        model_name = sv.model_name
        model = HookedTransformer.from_pretrained(model_name, device=device)
    plot_preference_at_scale_all(model, sv)

# %%
