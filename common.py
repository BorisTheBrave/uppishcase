# DEFAULT_MODEL = "Qwen/Qwen2-7B-Instruct"
from dataclasses import dataclass
from typing import Literal, Optional
from torch import Tensor


DEFAULT_MODEL = "google/gemma-2-2b-it"
DEFAULT_LAYER = 3

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


def get_embed_layer_tl(model):
    """Get the embedding layer from a TransformerLens model."""
    return model.embed

def get_layer_tl(model, layer_num):
    """
    Get a specific layer from the TransformerLens model.
    
    Args:
        model: The TransformerLens HookedTransformer model
        layer_num: Layer number to access
        
    Returns:
        The specified layer module
    """
    if layer_num >= model.cfg.n_layers:
        raise ValueError(f"Layer {layer_num} does not exist. Model has {model.cfg.n_layers} layers.")
    
    return model.blocks[layer_num]

@dataclass
class SteeringVector:
    dir: Tensor
    model_name: str
    hook: Literal['embed', 'resid_post'] = "resid_post"
    layer: Optional[int] = None
    per_tok: bool = False
    src: object = None

