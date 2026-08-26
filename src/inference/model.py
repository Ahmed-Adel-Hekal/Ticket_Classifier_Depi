from transformers import AutoTokenizer, AutoModelForCausalLM
from .config import MODEL_NAME

"""
File Function 
Function input - output 
"""


def load_model() : 
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME,device_map='auto',torch_dtype='auto')
    return tokenizer, model


def get_model_device(model) :
    return model.get_input_embeddings().weight.device
