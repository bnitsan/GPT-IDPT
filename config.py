from models.trainer import TrainerConfig
from transformers import GPT2TokenizerFast

base_model_name = 'distilgpt2'


def get_tokenizer():
    tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    return tokenizer


def get_train_conf(batch_size=8, epochs=3, lr=0.0004, train_num=4000, test_num=500, wandb_name=None):
    train_conf = TrainerConfig(max_epochs=epochs,
                               batch_size=batch_size,
                               learning_rate=lr,  # 0.0003,
                               lr_decay=True,
                               num_workers=1,
                               num_examples_per_ds=train_num,  # 4000,
                               num_examples_per_test_ds=test_num,  # 300,
                               max_tokenized=400,
                               max_char_len=1000,
                               ckpt_path=None,
                               wandb_name=wandb_name)

    return train_conf


class InferConfig:
    batch_size = 1  # this should not be changed
    num_workers = 1
    max_tokenized = 512
    num_examples_per_test_ds = 10
    max_char_len = 1000
    max_steps_infer = 100


class ModelConfig:
    model_name = base_model_name
    n_tokens_0 = 20
    n_tokens_IDPT = 20
    init_from_vocab = True

    def __init__(self, vocab_size, pad_token_id, **kwargs):
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        for k, v in kwargs.items():
            setattr(self, k, v)
