"""
taken from https://github.com/karpathy/minGPT

Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import logging

from tqdm import tqdm
import numpy as np

import torch
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from torch.optim import AdamW

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    weight_decay = 0.1
    learning_rate = 0.01  # could also be within the 1e-5 - 1e-3 range
    betas = (0.9, 0.95)
    epsilon = 1e-6
    lr_scheduler_type = "linear"
    num_warmup_steps = 100
    max_train_steps = 10000
    max_epochs = 10
    batch_size = 64
    grad_norm_clip = 1.0
    lr_decay = False
    ''' We currently do not use the following: warmup_tokens and final_tokens
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere.
    final_tokens = 260e9  # (at what point we reach 10% of original LR); not used
    '''

    # saving, loading and tracking parameters
    ckpt_path = None  # checkpoint settings
    num_workers = 1  # for DataLoader
    wandb_flag = True
    num_examples_per_ds = 100
    max_char_len = 1000

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def configure_optimizers(model, args):
    # Only update parameters not set to `require_grad=False`.
    optimizer_grouped_parameters = [
        {
            "params": [p for (n, p), param in zip(model.named_parameters(), model.parameters()) if param.requires_grad],
            "weight_decay": args.weight_decay,
            "epsilon": args.epsilon,
            "betas": args.betas,
        }
    ]
    ''' Original optimizer parameters used for the prompt tuning class we use:
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n == "soft_prompt.weight"],
            "weight_decay": args.weight_decay,
        }
    ]
    '''

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
            self.model.to(self.device)
            print('Setting device to cuda')

        # wandb logging
        if debug_flag:
            self.config.wandb_flag = False
        else:
            self.wandb = wandb
            if not self.wandb:
                self.config.wandb_flag = False

        if hasattr(self.model, "n_tokens"):
            self.shift_loss = self.model.n_tokens
        elif hasattr(self.model, "n_tokens_IDPT"):
            self.shift_loss = self.model.n_tokens_IDPT
        else:
            self.shift_loss = None

    def get_shifted_loss(self, logits, labels):
        assert self.shift_loss is not None
        n_tokens = logits.shape[1] - labels.shape[1]

        logits = logits[:, n_tokens:, :]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shifted_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return shifted_loss

    def save_checkpoint(self):
        logger.info("Saving checkpoint to... %s", self.config.ckpt_path)
        self.model.save_unfrozen_parts(self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        optimizer, lr_scheduler = configure_optimizers(model, config)

        def run_epoch(split):
            is_train = split == 'train'
            model.train(is_train)

            loader = self.train_dataloader if is_train else self.test_dataloader

            losses = []
            shifted_losses = []
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

                    if self.shift_loss is not None:
                        shifted_loss = self.get_shifted_loss(logits, labels_ids)
                        shifted_loss = shifted_loss.mean()
                        shifted_losses.append(shifted_loss.item())
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
                        if self.shift_loss is not None:
                            self.wandb.log({"train_shift_loss": shifted_loss.item()})

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if is_train:
                mean_epoch_train_loss = float(np.mean(shifted_losses)) if self.shift_loss is not None else float(np.mean(losses))
                if self.shift_loss is not None:
                    self.wandb.log({"mean_epoch_train_loss": mean_epoch_train_loss})

            if not is_train:
                test_loss_i = float(np.mean(losses))
                logger.info("test_loss: %f", test_loss_i)
                if self.shift_loss is not None:
                    self.wandb.log({"test_shift_loss": shifted_loss.item()})
                    self.wandb.log({"mean_epoch_test_loss": float(np.mean(shifted_losses))})
                return test_loss_i

        best_loss = float('inf')
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
