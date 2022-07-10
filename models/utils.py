import random
import numpy as np
import torch
from torch.nn import functional as F

import wandb
from pathlib import Path
import os

from models.trainer import Trainer

from models.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset


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
def sample(model, source_ids, source_mask, steps, tokenizer, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    model.eval()
    x_in = source_ids
    x_mask = source_mask
    for k in range(steps):
        print(x_in)
        print(x_mask)
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

        # set probs to zero at the location of the padding token
        probs[tokenizer.pad_token_id] = 0.0

        # repetition of the last token is not allowed
        if k > 0:
            probs[ix] = 0.0

        # sampling
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=3, dim=-1)
        # append to the sequence and continue
        x_in = torch.cat((x_in, ix.unsqueeze(0)), dim=1)
        x_mask = torch.cat([x_mask, torch.ones((1, 1))], 1)
        if ix == tokenizer.eos_token_id:
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
                             max_tok_len=train_config.max_tokenized,
                             max_char_len=train_config.max_char_len)
    train_dataloader = DataLoader(train_ds,
                                  shuffle=True,
                                  batch_size=train_config.batch_size,
                                  num_workers=train_config.num_workers)
    test_ds = CustomDataset(ds_name,
                            tokenizer,
                            num_examples=train_config.num_examples_per_test_ds,
                            split_type='test',
                            max_tok_len=train_config.max_tokenized,
                            max_char_len=train_config.max_char_len)
    test_dataloader = DataLoader(test_ds,
                                 shuffle=True,
                                 batch_size=train_config.batch_size,
                                 num_workers=train_config.num_workers)
    # Create the trainer
    model_train = Trainer(model, train_dataloader, test_dataloader, train_config,
                          wandb=wandb)  # the None is for test_dataset
    # model train
    model_train.train()

    torch.save(model.state_dict(), 'model_weights/' + name + '_' + ds_name + '_state_dict.pt')


def get_concat_dl(ds_names, train_config, tokenizer):
    set_seed(1)

    train_ds = CustomDataset(ds_names[0],
                             tokenizer,
                             num_examples=train_config.num_examples_per_ds,
                             split_type='train',
                             max_tok_len=train_config.max_tokenized,
                             max_char_len=train_config.max_char_len)
    test_ds = CustomDataset(ds_names[0],
                            tokenizer,
                            num_examples=train_config.num_examples_per_ds,
                            split_type='test',
                            max_tok_len=train_config.max_tokenized,
                            max_char_len=train_config.max_char_len)
    for i in range(1, len(ds_names)):
        train_ds_i = CustomDataset(ds_names[i],
                                   tokenizer,
                                   num_examples=train_config.num_examples_per_ds,
                                   split_type='train',
                                   max_tok_len=train_config.max_tokenized,
                                   max_char_len=train_config.max_char_len)
        test_ds_i = CustomDataset(ds_names[i],
                                  tokenizer,
                                  num_examples=train_config.num_examples_per_ds,
                                  split_type='test',
                                  max_tok_len=train_config.max_tokenized,
                                  max_char_len=train_config.max_char_len)
        train_ds = ConcatDataset([train_ds, train_ds_i])
        test_ds = ConcatDataset([test_ds, test_ds_i])

        train_dataloader = DataLoader(train_ds, shuffle=True,
                                      batch_size=train_config.batch_size, num_workers=train_config.num_workers)
        test_dataloader = DataLoader(test_ds, shuffle=True,
                                     batch_size=train_config.batch_size, num_workers=train_config.num_workers)
        return train_dataloader, test_dataloader


def get_concat_dl_val(ds_names, train_config, tokenizer):
    set_seed(1)

    val_ds = CustomDataset(ds_names[0],
                           tokenizer,
                           num_examples=train_config.num_examples_per_test_ds,
                           split_type='validation',
                           max_tok_len=train_config.max_tokenized,
                           max_char_len=train_config.max_char_len)
    for i in range(1, len(ds_names)):
        val_ds_i = CustomDataset(ds_names[i],
                                 tokenizer,
                                 num_examples=train_config.num_examples_per_test_ds,
                                 split_type='train',
                                 max_tok_len=train_config.max_tokenized,
                                 max_char_len=train_config.max_char_len)
        val_ds = ConcatDataset([val_ds, val_ds_i])

        val_dataloader = DataLoader(val_ds, shuffle=True,
                                    batch_size=train_config.batch_size, num_workers=train_config.num_workers)
        return val_dataloader


def train_model_concat(model, name, train_config, ds_names, path_name, tokenizer):
    # Initialize wandb
    wandb = initialize_wandb(train_config)

    train_dataloader, test_dataloader = get_concat_dl(ds_names, train_config, tokenizer)
    # Create the trainer
    model_train = Trainer(model, train_dataloader, test_dataloader, train_config,
                          wandb=wandb)  # the None is for test_dataset
    # model train
    model_train.train()

    torch.save(model.state_dict(), 'model_weights/' + name + '_' + path_name + '_state_dict.pt')


def get_val_dl(ds_name, tokenizer, infer_config):
    set_seed(1)
    # check if ds_name is list of strings or single string
    if isinstance(ds_name, list):
        val_dataloader = get_concat_dl_val(ds_name, infer_config, tokenizer)
    else:
        val_ds = CustomDataset(ds_name,
                               tokenizer,
                               num_examples=infer_config.num_examples_per_test_ds,
                               split_type='validation',
                               max_tok_len=infer_config.max_tokenized,
                               max_char_len=infer_config.max_char_len)
        val_dataloader = DataLoader(val_ds,
                                    shuffle=True,
                                    batch_size=infer_config.batch_size,
                                    num_workers=infer_config.num_workers)
    return val_dataloader
