from .types import *
from .alpaca_lora import Model as AlpacaLora
from .alpaca import Model as Alpaca
from .vicuna import Model as Vicuna


# BASE_MODEL = "decapoda-research/llama-13b-hf"

# # This model has error
# LORA_WEIGHTS = "baseten/alpaca-30b"
# LORA_WEIGHTS = "chansung/alpaca-lora-13b"
# model = AlpacaLora(base_model=BASE_MODEL, load_in_8bit=True,
#                lora_weights=LORA_WEIGHTS)

# BASE_MODEL = "eachadea/legacy-vicuna-13b"
# model = Vicuna(base_model=BASE_MODEL)

# BASE_MODEL = "chavinlo/gpt4-x-alpaca"
# model = Alpaca(base_model=BASE_MODEL)

def get_default_weight(model):
    
    if model == "alpaca":
        return "chavinlo/gpt4-x-alpaca", ""
    elif model == "alpacalora":
        return "decapoda-research/llama-13b-hf", "chansung/alpaca-lora-13b"
    elif model == "vicuna":
        return "eachadea/legacy-vicuna-13b", ""
    else:
        raise Exception("Model not implemented")


def create_model(model, weight, lora_weights):
    default_weight, default_lora_weights = get_default_weight(model)
    if model == "alpaca":
        return Alpaca(base_model=weight if weight is not None else default_weight)
    elif model == "vicuna":            
        return Vicuna(base_model=weight if weight is not None  else default_weight)
    elif model == "alpacalora":
        return AlpacaLora(base_model=weight if weight is not None  else default_weight, 
                          lora_weights=lora_weights if lora_weights is not None  else default_lora_weights,
                          load_in_8bit=True)
    else:
        raise Exception("Model not implemented")