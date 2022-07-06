"""
    This module was written in a rush, so it's not pretty and some stuff are hard-coded.
    We would like to compare the performance of ~3 models on ~3 datasets (or a combination of them).
    We first train all three models on all three datasets + a combination of the datasets -- resulting in 12 models.
    We save the output models to a folder "model_weights".
"""

from models.trainer import TrainerConfig, Trainer
from models.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch

from models.utils import initialize_wandb, train_model_db, train_model_concat

from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from models.models import GPT2TrainedPrompt, GPT2TrainedPromptX

base_model_name = 'distilgpt2'

# define tokenizer, extend with pad token
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})


# general config for the models
class Config:
    model_name = base_model_name
    n_tokens_p0 = 20
    last_tokens_to_keep = 20
    vocab_size = len(tokenizer)


model_config = Config()

# training configuration
train_conf = TrainerConfig(max_epochs=1,
                           batch_size=16,
                           learning_rate=0.012,
                           lr_decay=True,
                           warmup_tokens=10,  # 512*20,
                           final_tokens=5000,  # 2*len(train_dataset)*block_size,
                           num_workers=1,
                           num_examples_per_ds=2500,
                           num_examples_per_test_ds=300,
                           max_tokenized=200)

# databases to train on
ds_names = ["wiki_qa", "wiki_bio", "samsum"]

# train all 3 models on the datasets separately
for ds_name_i in ds_names:
    # model 1: "ultra"-baseline - a simple GPT2 with LM head with no prompt tuning
    model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
    model0.resize_token_embeddings(len(tokenizer))
    train_model_db(model0, 'model0', train_conf, ds_name_i, tokenizer)

    # model 2: baseline - a GPT2 with trained prompt
    model_p = GPT2TrainedPrompt(model_config)
    train_model_db(model_p, 'model_p', train_conf, ds_name_i, tokenizer)

    # model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
    model_px = GPT2TrainedPromptX(model_config)
    train_model_db(model_px, 'model_px', train_conf, ds_name_i, tokenizer)

# another run: train on the concatenated datasets
model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
model0.resize_token_embeddings(len(tokenizer))
train_model_concat(model0, 'model0', train_conf, 'combined', tokenizer)

model_p = GPT2TrainedPrompt(model_config)
train_model_concat(model_p, 'model_p', train_conf, 'combined', tokenizer)

model_px = GPT2TrainedPromptX(model_config)
train_model_concat(model_px, 'model_px', train_conf, 'combined', tokenizer)
