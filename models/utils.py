import random
import numpy as np
import torch
from torch.nn import functional as F

import wandb
from pathlib import Path
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


# based on https://github.com/karpathy/minGPT/blob/master/mingpt/utils.py
@torch.no_grad()
def sample(model, batch, steps, tokenizer, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    model.eval()
    x_in = batch['source_ids']
    x_mask = batch['source_mask']
    for k in range(steps):
        logits, _ = model(input_ids=x_in, attention_mask=x_mask, return_dict=False)
        # pluck the logits at the final step and scale by temperature
        logits = logits[0, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x_in = torch.cat((x_in, ix.unsqueeze(0)), dim=1)
        x_mask = torch.cat([x_mask, torch.ones((1,1))],1)
        if ix-1 >= len(tokenizer)-2:  # pure laziness. Should be more careful on eos/pad tokens
            break
    return x_in


def initialize_wandb(config):
    """Initializes the wandb module."""
    with open(str(Path(os.getcwd()).parent.absolute())+'/wandb_key.txt') as key_f:
        wandbkey = key_f.read()
    wandb.login(key=wandbkey)
    wandb.init(project="promptgpt", entity="nits")

    wandb.config = {
        "learning_rate": config.learning_rate,
        "epochs": config.max_epochs,
        "batch_size": config.batch_size
    }
    return wandb
