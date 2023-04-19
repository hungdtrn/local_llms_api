from .types import *
from .huggingface import Model as Model

def get_default_weight(model):
    if model == "llama":
        return "chavinlo/gpt4-x-alpaca", ""
    elif model == "alpacalora":
        return "decapoda-research/llama-13b-hf", "chansung/alpaca-lora-13b"
    elif model == "huggingface":
        return "eachadea/legacy-vicuna-13b", ""
    else:
        raise Exception("Model not implemented")

def create_model(model, weight, lora_weights, load8bit, **kwargs):
    """ The currently support models are llama, alpacalora, and generic huggingface models.
    Both models are loaded from huggingface. 
    """
    if model not in ["alpacalora", "llama", "huggingface"]:
        raise Exception("Model not implemented")
    default_weight, default_lora_weights = get_default_weight(model)
    weight = weight if weight is not None else default_weight
    lora_weights = lora_weights if lora_weights is not None else default_lora_weights
    
    return Model(model_name=model, model_path=weight, lora_path=lora_weights,
                 load_in_8bit=load8bit)
