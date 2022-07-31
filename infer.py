from models.utils import sample
from datasets import load_metric
import torch
from transformers import GPT2LMHeadModel
from models.model_prompt_tuning import GPT2PromptTuningLM, GPT2IDPTLM
from models.utils import get_val_dl
from config import ModelConfig, InferConfig, get_tokenizer

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--val_num', type=int, required=True)
parser.add_argument('--tokens0', type=int, default=20)
parser.add_argument('--tokensIDPT', type=int, default=20)
args = parser.parse_args()


print_flag = False


def get_score(model, model_name, ds_name, tokenizer, infer_conf, metric):
    model.eval()
    print('Now evaluating model: ' + model_name)
    print('On dataset: ' + ds_name) if not isinstance(ds_name, list) else print('On combined dataset')
    print('-----------------------------------------------------')

    val_dataloader = get_val_dl(ds_name, tokenizer, infer_conf)
    max_steps_infer = infer_conf.max_steps_infer
    sum_f1 = 0
    for i, batch in enumerate(val_dataloader):
        input_ids = batch['source_ids']
        attention_mask = batch['source_mask']

        # find last location in input_ids where attention_mask is 1
        last_loc = torch.where(attention_mask.squeeze() == 1)[-1]
        input_ids = input_ids[:, :last_loc[-1] + 1]
        attention_mask = attention_mask[:, :last_loc[-1] + 1]

        # sample from model
        x_out = sample(model, input_ids, attention_mask, steps=max_steps_infer, tokenizer=tokenizer, temperature=0.8,
                       sample_flag=True, top_k=10)
        # remove the input tokens
        x_out = x_out[:, (last_loc[-1] + 1):]

        # convert prediction to string
        str_pred = ''.join(
            [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in x_out])

        # get target string
        last_loc_target = torch.where(batch['target_mask'].squeeze() == 1)[-1]
        target_ids = batch['target_ids'][0][:last_loc_target[-1] + 1]
        str_ref = ''.join([tokenizer.decode(g, clean_up_tokenization_spaces=True) for g in target_ids])

        # get target string output
        last_loc_source = torch.where(batch['source_mask'].squeeze() == 1)[-1]
        source_ids = batch['source_ids'][0][:last_loc_source[-1] + 1]
        str_ref_in = ''.join([tokenizer.decode(g, clean_up_tokenization_spaces=True) for g in source_ids])

        if print_flag:
            print('input: ' + str_ref_in + str_ref)
            print('pred: ' + str_pred)
            print('-----------------------------------------------------')

        scores_gen_true = metric.compute(predictions=[str_pred], references=[str_ref])
        sum_f1 += scores_gen_true['rouge1'].mid.fmeasure

    return sum_f1 / (i + 1)


tokenizer = get_tokenizer()
infer_config = InferConfig()
infer_config.num_examples_per_test_ds = args.val_num
model_config = ModelConfig(vocab_size=len(tokenizer),
                           pad_token_id=tokenizer.pad_token_id)
model_config.n_tokens_0 = args.tokens0
model_config.n_tokens_IDPT = args.tokensIDPT

ds_names = ["rotten_tomatoes", "quartz", "samsum"]  # other possible: "wiki_qa", "wiki_bio"
max_steps_per_db = [5, 4, 100]

# We adopt for now the ROUGE metric for scoring. It is not conventional, but it is easy to implement.
# In some cases, researchers even use "exact match" score.
rouge_score = load_metric("rouge")

for i, ds_name_i in enumerate(ds_names):
    # every ds should be evaluated with different max steps in inference, for cleaner results
    infer_config.max_steps_infer = max_steps_per_db[i]

    # model 1: "ultra"-baseline - a simple GPT2 with LM head with no prompt tuning
    model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
    # model0.resize_token_embeddings(len(tokenizer))
    model0.load_state_dict(torch.load('model_weights/model0_' + ds_name_i + '_state_dict.pt'))
    score0 = get_score(model0, 'model0', ds_name_i, tokenizer, infer_config, rouge_score)

    # model 2: baseline - a GPT2 with trained prompt
    model_p = GPT2PromptTuningLM.from_pretrained(
        model_config.model_name,
        weights_path='model_p_' + ds_name_i + '_state_dict.pt',
        n_tokens=model_config.n_tokens_0,
        initialize_from_vocab=model_config.init_from_vocab,
        vocab_size=model_config.vocab_size
    )
    score_p = get_score(model_p, 'model_p', ds_name_i, tokenizer, infer_config, rouge_score)

    # model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
    model_px = GPT2IDPTLM.from_pretrained(
        model_config.model_name,
        weights_path='model_px_' + ds_name_i + '_state_dict.pt',
        n_tokens_0=model_config.n_tokens_0,
        n_tokens_IDPT=model_config.n_tokens_IDPT,
        initialize_from_vocab=model_config.init_from_vocab,
        vocab_size=model_config.vocab_size
    )
    score_px = get_score(model_px, 'model_px', ds_name_i, tokenizer, infer_config, rouge_score)

    print('ds_name: ' + ds_name_i)
    print('score0: ' + str(score0) + '\nscore_p: ' + str(score_p) + '\nscore_px: ' + str(score_px))


# Combined datasets
infer_config.max_steps_infer = 5  # restrict to 4 steps, use only two datasets
ds_names = ["quartz", "rotten_tomatoes"]

# combined datasets case
model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
# model0.resize_token_embeddings(len(tokenizer))
model0.load_state_dict(torch.load('model_weights/model0_combined_state_dict.pt'))
score0 = get_score(model0, 'model0', ds_names, tokenizer, infer_config, rouge_score)

# model 2: baseline - a GPT2 with trained prompt
model_p = GPT2PromptTuningLM.from_pretrained(
    model_config.model_name,
    weights_path='model_p_combined_state_dict.pt',
    n_tokens=model_config.n_tokens_0,
    initialize_from_vocab=model_config.init_from_vocab,
    vocab_size=model_config.vocab_size
)
score_p = get_score(model_p, 'model_p', ds_names, tokenizer, infer_config, rouge_score)

# model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
model_px = GPT2IDPTLM.from_pretrained(
    model_config.model_name,
    weights_path='model_px_combined_state_dict.pt',
    n_tokens_0=model_config.n_tokens_0,
    n_tokens_IDPT=model_config.n_tokens_IDPT,
    initialize_from_vocab=model_config.init_from_vocab,
    vocab_size=model_config.vocab_size
)
score_px = get_score(model_px, 'model_px', ds_names, tokenizer, infer_config, rouge_score)

print('COMBINED SET:\nscore0: ' + str(score0) + '\nscore_p: ' + str(score_p) + '\nscore_px: ' + str(score_px))
