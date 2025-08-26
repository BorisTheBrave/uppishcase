DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"

def get_embed_layer(model):
    if hasattr(model, "base_model"):
        model = model.base_model

    for prop in ["embed_in", "embed_tokens"]:
        if hasattr(model, prop):
            return getattr(model, prop)
    raise ValueError("Model does not have an embed_in layer")


def get_layer(model, layer_num):
    """
    Get a specific layer from the model.
    
    Args:
        model: The transformer model
        layer_num: Layer number to access
        
    Returns:
        The specified layer module
    """
    if hasattr(model, "base_model"):
        model = model.base_model
    
    # Handle different model architectures
    if hasattr(model, "layers"):
        layers = model.layers
    elif hasattr(model, "h"):  # GPT-style
        layers = model.h
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):  # BERT-style
        layers = model.encoder.layer
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2 style
        layers = model.transformer.h
    else:
        raise ValueError("Could not find layers in model")
    
    if layer_num >= len(layers):
        raise ValueError(f"Layer {layer_num} does not exist. Model has {len(layers)} layers.")
    
    return layers[layer_num]