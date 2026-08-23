import torch

def apply_temperature(logit, temp) : 
    if temp <=0 :
        return logit
    return logit/temp

def apply_top_k(logits, top_k) :
    if top_k is None  or top_k <= 0 :
        return logits

    top_k = min(top_k, logits.size(-1))

    values, _ = torch.topk(logits, top_k)
    #  [[9,8,7]]
    threshold = values[:,-1].unsqueeze(-1)

    logits = logits.masked_fill(
        logits < threshold,
        float('-inf')
    )

    return logits

def apply_top_p(logits, top_p) :

    if top_p is None or top_p>= 1.0 : 
        return logits

    if not 0.0 < top_p <= 1.0 : 
        raise ValueError("top_p must be between 0 and 1.")

    # [batch_size, seq_len, vocab_size]

    # [i , love , to , eat]
    # [.1, .7,   .3  , .6]
    # [.7  .6 .3 .1]  --> sorted logits
    # [1, 3, 2, 0]   --> sorted indices
    # [.6  .3 .1  .1 ]
    # [.6  .9  1.0  1.1]

    sorted_logits , sorted_indices = torch.sort(logits, descending=True, dim=-1)

    sorted_prob = torch.softmax(sorted_logits, dim=-1)
# [4]
    cumulative_prop = torch.cumsum(sorted_prob, dim= -1)

    sorted_indices_to_remove = cumulative_prop > top_p

    sorted_indices_to_remove[...,1:] = (sorted_indices_to_remove[...,:-1].clone())

    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(
        logits,
        dtype= torch.bool
    )

    indices_to_remove.scatter_(
        dim=-1,
        index = sorted_indices,
        src = sorted_indices_to_remove
    )

    logits = logits.masked_fill(
        indices_to_remove,
        float('-inf')
    )

    return logits

def select_next_token(logits, temp) : 
    if temp == 0 :
        return torch.argmax(logits, dim=-1, keepdim=True)

    prob = torch.softmax(logits, dim = -1)

    return torch.multinomial(prob, num_samples=1)


# unbelivable



def apply_logit_bias(logits: torch.tensor,
                     token_bias:dict[int,float]) ->torch.Tensor : 


    logits = logits.clone()

    for token_id, bias in token_bias.items()  :
        logits[token_id] += bias
    return logits