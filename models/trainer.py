"""
taken from https://github.com/karpathy/minGPT

Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from transformers import get_scheduler
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters

    weight_decay = 0.01
    learning_rate = 0.01  # could also be within the 1e-5 - 1e-3 range
    lr_scheduler_type = "linear"
    num_warmup_steps = 100
    max_train_steps = 10000
    max_epochs = 10
    batch_size = 64
    # betas = (0.9, 0.95)  # currently not using betas for optimizers
    grad_norm_clip = 1.0
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 1e4  # 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 1e8  # 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    wandb_flag = True
    num_examples_per_ds = 100

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def configure_optimizers(model, args):
    # Only update parameters not set to `require_grad=False`.
    optimizer_grouped_parameters = [
        {
            "params": [p for (n, p), param in zip(model.named_parameters(), model.parameters()) if param.requires_grad],
            "weight_decay": args.weight_decay,
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    return optimizer, lr_scheduler


class Trainer:

    def __init__(self, model, train_dataloader, test_dataloader, train_config, wandb=None, debug_flag=False):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = train_config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        # wandb logging
        if debug_flag:
            self.config.wandb_flag = False
        self.wandb = wandb
        if not self.wandb:
            self.config.wandb_flag = False

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        # raw_model = model.module if hasattr(self.model, "module") else model
        optimizer, lr_scheduler = configure_optimizers(model, config)

        def run_epoch(split):
            is_train = split == 'train'

            model.train(is_train)

            loader = self.train_dataloader if is_train else self.test_dataloader

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

            for it, batch_dat in pbar:

                input_ids = batch_dat['input_ids'].to(self.device)
                attention_mask = batch_dat['attention_mask'].to(self.device)
                labels_ids = batch_dat['input_ids'].to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    model_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_ids)

                    loss = model_output[0]
                    logits = model_output[1]
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:
                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    lr_scheduler.step()
                    lr = optimizer.param_groups[0]["lr"]

                    if self.config.wandb_flag:
                        self.wandb.log({"train_loss": loss.item()})

                    """
                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (labels_ids >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    """
                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataloader is not None:
                test_loss = run_epoch('test')

                if self.config.wandb_flag:
                    self.wandb.log({"test_loss": test_loss})

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataloader is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
