DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

def get_embed_layer(model):
    if hasattr(model, "base_model"):
        model = model.base_model

    for prop in ["embed_in", "embed_tokens"]:
        if hasattr(model, prop):
            return getattr(model, prop)
    raise ValueError("Model does not have an embed_in layer")