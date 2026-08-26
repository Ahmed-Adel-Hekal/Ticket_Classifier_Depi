import torch

def tokenize(tokenizer, text, device) :
    input = tokenizer(text, return_tensors='pt')

    return {key :value.to(device) for key, value in input.items() }
