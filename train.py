"""
    We would like to compare the performance of ~3 models on ~3 datasets (or a combination of them).
    We first train all three models on all three datasets + a combination of the datasets -- resulting in 12 models.
    We save the output models to a folder "model_weights".
"""

from models.utils import train_model_db, train_model_concat

from transformers import GPT2LMHeadModel
from models.model_prompt_tuning import GPT2PromptTuningLM, GPT2IDPTLM
from config import ModelConfig, get_train_conf, get_tokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=True)
parser.add_argument('--epochs', type=int, required=True)
parser.add_argument('--lr', type=float, required=True)
parser.add_argument('--train_num', type=int, required=True)
parser.add_argument('--test_num', type=int, required=True)
parser.add_argument('--tokens0', type=int, default=20)
parser.add_argument('--tokensIDPT', type=int, default=20)
args = parser.parse_args()


tokenizer = get_tokenizer()

model_config = ModelConfig(vocab_size=len(tokenizer),
                           pad_token_id=tokenizer.eos_token_id)  # pad_token_id)
model_config.n_tokens_0 = args.tokens0
model_config.n_tokens_IDPT = args.tokensIDPT

# training configuration
train_conf = get_train_conf(batch_size=args.batch,
                            epochs=args.epochs,
                            lr=args.lr,
                            train_num=args.train_num,
                            test_num=args.test_num,
                            wandb_name=str(args)[9:])


# databases to train on
ds_names = ["quartz", "rotten_tomatoes", "samsum"]  # other possible: "wiki_qa", "wiki_bio"

# train all (3) models on the datasets separately
for ds_name_i in ds_names:
    print("-----------------------------------------------------")
    print("Training models on dataset: " + ds_name_i)
    print("-----------------------------------------------------")

    # model 1: "ultra"-baseline - a simple GPT2 with LM head with no prompt tuning
    model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
    # model0.resize_token_embeddings(len(tokenizer))
    train_model_db(model0, 'model0', train_conf, ds_name_i, tokenizer)

    # model 2: baseline - a GPT2 with trained prompt
    model_p = GPT2PromptTuningLM.from_pretrained(
        model_config.model_name,
        n_tokens=model_config.n_tokens_0,
        initialize_from_vocab=model_config.init_from_vocab,
        vocab_size=model_config.vocab_size
    )
    train_model_db(model_p, 'model_p', train_conf, ds_name_i, tokenizer)

    # model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
    model_px = GPT2IDPTLM.from_pretrained(
        model_config.model_name,
        n_tokens_0=model_config.n_tokens_0,
        n_tokens_IDPT=model_config.n_tokens_IDPT,
        initialize_from_vocab=model_config.init_from_vocab,
        vocab_size=model_config.vocab_size
    )
    train_model_db(model_px, 'model_px', train_conf, ds_name_i, tokenizer)


print("-----------------------------------------------------")
print("Training models on a combined dataset")
print("-----------------------------------------------------")

# another run: train on the concatenated datasets
model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
# model0.resize_token_embeddings(len(tokenizer))
train_model_concat(model0, 'model0', train_conf, ds_names, 'combined', tokenizer)

model_p = GPT2PromptTuningLM.from_pretrained(
    model_config.model_name,
    n_tokens=model_config.n_tokens_0,
    initialize_from_vocab=model_config.init_from_vocab,
    vocab_size=model_config.vocab_size
)
train_model_concat(model_p, 'model_p', train_conf, ds_names, 'combined', tokenizer)

model_px = GPT2IDPTLM.from_pretrained(
    model_config.model_name,
    n_tokens_0=model_config.n_tokens_0,
    n_tokens_IDPT=model_config.n_tokens_IDPT,
    initialize_from_vocab=model_config.init_from_vocab,
    vocab_size=model_config.vocab_size
)
train_model_concat(model_px, 'model_px', train_conf, ds_names, 'combined', tokenizer)
