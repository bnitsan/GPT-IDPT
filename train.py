'''
    This module was written in a rush, so it's not pretty.
    We would like to compare the performance of ~3 models on ~3 datasets (or a combination of them).
    We first train all three models on all three datasets + a combination of the datasets -- resulting in 12 models.
'''


from models.trainer import TrainerConfig, Trainer
from models.dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
import torch

from models.utils import initialize_wandb
from models.utils import set_seed

from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from models.models import GPT2TrainedPrompt, GPT2TrainedPromptX


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
    model_train = Trainer(model, train_dataloader, test_dataloader, train_conf, wandb=wandb)  # the None is for test_dataset
    # model train
    model_train.train()

    torch.save(model.state_dict(), name+'_'+ds_name+'_state_dict.pt')


base_model_name = 'distilgpt2'

# define tokenizer, extend with pad token
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


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
                           warmup_tokens=10,  #512*20,
                           final_tokens=5000,  # 2*len(train_dataset)*block_size,
                           num_workers=1,
                           num_examples_per_ds=2500,
                           num_examples_per_test_ds=300,
                           max_tokenized=200)

ds_names = ["wiki_qa", "wiki_bio", "samsum"]

# train all models on the datasets separately
for ds_name_i in ds_names:
    # define models
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
    model_train = Trainer(model, train_dataloader, test_dataloader, train_conf, wandb=wandb)  # the None is for test_dataset
    # model train
    model_train.train()

    torch.save(model.state_dict(), name+'_'+ds_name+'_state_dict.pt')


model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
model0.resize_token_embeddings(len(tokenizer))
train_model_concat(model0, 'model0', train_conf, 'combined', tokenizer)

# model 2: baseline - a GPT2 with trained prompt
model_p = GPT2TrainedPrompt(model_config)
train_model_concat(model_p, 'model_p', train_conf, 'combined', tokenizer)

# model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
model_px = GPT2TrainedPromptX(model_config)
train_model_concat(model_px, 'model_px', train_conf, 'combined', tokenizer)


