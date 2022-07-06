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
    # out[out < v[:, [-1]]] = -float('Inf')
    out[out < min(v)] = -float('Inf')
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
        model_output = model(input_ids=x_in, attention_mask=x_mask, return_dict=False)
        logits = model_output[0]
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
        x_mask = torch.cat([x_mask, torch.ones((1, 1))], 1)
        if ix - 1 >= len(tokenizer) - 2:  # pure laziness. Should be more careful on eos/pad tokens
            break
    return x_in


def initialize_wandb(config):
    """Initializes the wandb module."""
    with open(str(Path(os.getcwd()).parent.absolute()) + '/wandb_key.txt') as key_f:
        wandbkey = key_f.read()
    wandb.login(key=wandbkey)
    wandb.init(project="promptgpt", entity="nits")

    wandb.config = {
        "learning_rate": config.learning_rate,
        "epochs": config.max_epochs,
        "batch_size": config.batch_size
    }
    return wandb


def train_model_db(model, name, train_config, ds_name, tokenizer):
    """
    Train the model on the dataset
    """
    # Initialize wandb
    wandb = initialize_wandb(train_config)

    # Create the datasets and data-loaders
    set_seed(1)
    train_ds = CustomDataset(ds_name,
                             tokenizer,
                             num_examples=train_config.num_examples_per_ds,
                             split_type='train',
                             max_len=train_config.max_tokenized)
    train_dataloader = DataLoader(train_ds,
                                  shuffle=True,
                                  batch_size=train_config.batch_size,
                                  num_workers=train_config.num_workers)
    test_ds = CustomDataset(ds_name,
                            tokenizer,
                            num_examples=train_config.num_examples_per_test_ds,
                            split_type='test',
                            max_len=train_config.max_tokenized)
    test_dataloader = DataLoader(test_ds,
                                 shuffle=True,
                                 batch_size=train_config.batch_size,
                                 num_workers=train_config.num_workers)
    # Create the trainer
    model_train = Trainer(model, train_dataloader, test_dataloader, train_conf,
                          wandb=wandb)  # the None is for test_dataset
    # model train
    model_train.train()

    torch.save(model.state_dict(), 'model_weights/' + name + '_' + ds_name + '_state_dict.pt')


def get_concat_dl(ds_names, train_config):
    set_seed(1)

    train_ds = CustomDataset(ds_names[0],
                             tokenizer,
                             num_examples=train_config.num_examples_per_ds,
                             split_type='train',
                             max_len=train_config.max_tokenized)
    test_ds = CustomDataset(ds_names[0],
                            tokenizer,
                            num_examples=train_config.num_examples_per_ds,
                            split_type='test',
                            max_len=train_config.max_tokenized)
    for i in range(1, len(ds_names)):
        train_ds_i = CustomDataset(ds_names[i],
                                   tokenizer,
                                   num_examples=train_config.num_examples_per_ds,
                                   split_type='train',
                                   max_len=train_config.max_tokenized)
        test_ds_i = CustomDataset(ds_names[i],
                                  tokenizer,
                                  num_examples=train_config.num_examples_per_ds,
                                  split_type='test',
                                  max_len=train_config.max_tokenized)
        train_ds = ConcatDataset([train_ds, train_ds_i])
        test_ds = ConcatDataset([test_ds, test_ds_i])

        train_dataloader = DataLoader(train_ds, shuffle=True,
                                      batch_size=train_config.batch_size, num_workers=train_config.num_workers)
        test_dataloader = DataLoader(test_ds, shuffle=True,
                                     batch_size=train_config.batch_size, num_workers=train_config.num_workers)
        return train_dataloader, test_dataloader


def train_model_concat(model, name, train_config, ds_name, tokenizer):
    # Initialize wandb
    wandb = initialize_wandb(train_config)

    train_dataloader, test_dataloader = get_concat_dl(ds_names, train_config)
    # Create the trainer
    model_train = Trainer(model, train_dataloader, test_dataloader, train_conf,
                          wandb=wandb)  # the None is for test_dataset
    # model train
    model_train.train()

    torch.save(model.state_dict(), 'model_weights/' + name + '_' + ds_name + '_state_dict.pt')
