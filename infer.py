from models.utils import sample
from datasets import load_metric
import torch
from models.trainer import TrainerConfig
from models.dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from models.models import GPT2TrainedPrompt, GPT2TrainedPromptX
from models.utils import get_val_dl


def get_score(model, model_name, ds_name, tokenizer, infer_conf, metric, max_steps_infer=10):
    # load state dict to model
    if isinstance(ds_name, list):
        model.load_state_dict(torch.load('model_weights/' + model_name + '_combined_state_dict.pt'))
    else:
        model.load_state_dict(torch.load('model_weights/' + model_name + '_' + ds_name + '_state_dict.pt'))
    print(ds_name)
    val_dataloader = get_val_dl(ds_name, tokenizer, infer_conf)
    sum_f1 = 0
    for i, batch in enumerate(val_dataloader):
        input_ids = batch['source_ids']
        attention_mask = batch['source_mask']

        # find last location in input_ids where attention_mask is 1
        last_loc = torch.where(attention_mask.squeeze() == 1)[-1]
        input_ids = input_ids[:, :last_loc[-1] + 1]
        attention_mask = attention_mask[:, :last_loc[-1] + 1]

        x_out = sample(model, batch, steps=max_steps_infer, tokenizer=tokenizer, temperature=1, sample=True, top_k=10)
        x_out = x_out[0, len(batch['source_ids'][0]):]
        str_pred = ' '.join(
            [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in x_out])

        last_loc_target = torch.where(batch['target_mask'].squeeze() == 1)[-1]
        target_ids = batch['target_ids'][0][:last_loc_target[-1] + 1]
        str_ref = ' '.join([tokenizer.decode(g) for g in target_ids])
        print('pred: ' + str_pred + '\nref:' + str_ref)

        scores_gen_true = metric.compute(predictions=[str_pred], references=[str_ref])
        sum_f1 += scores_gen_true['rouge1'].mid.fmeasure

    return sum_f1 / (i + 1)


base_model_name = 'distilgpt2'

# define tokenizer, extend with pad token
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})


class InferConf:
    batch_size = 1  #
    num_workers = 1
    max_tokenized = 512
    num_examples_per_test_ds = 4


class ModelConfig:
    model_name = base_model_name
    n_tokens_p0 = 20
    last_tokens_to_keep = 20
    vocab_size = len(tokenizer)


infer_config = InferConf()
model_config = ModelConfig()

ds_names = ["wiki_qa", "wiki_bio", "samsum"]

# We adopt for now the ROUGE metric for scoring. It is not conventional, but it is easy to implement.
# In some cases, researchers even use "exact match" score.
rouge_score = load_metric("rouge")

for ds_name_i in ds_names:
    # model 1: "ultra"-baseline - a simple GPT2 with LM head with no prompt tuning
    model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
    model0.resize_token_embeddings(len(tokenizer))
    score0 = get_score(model0, 'model0', ds_name_i, tokenizer, infer_config, rouge_score, max_steps_infer=10)

    # model 2: baseline - a GPT2 with trained prompt
    model_p = GPT2TrainedPrompt(model_config)
    score_p = get_score(model_p, 'model_p', ds_name_i, tokenizer, infer_config, rouge_score, max_steps_infer=10)

    # model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
    model_px = GPT2TrainedPromptX(model_config)
    score_px = get_score(model_px, 'model_px', ds_name_i, tokenizer, infer_config, rouge_score, max_steps_infer=10)

    print('ds_name: ' + ds_name_i)
    print('score0: ' + str(score0) + '\nscore_p: ' + str(score_p) + '\nscore_px: ' + str(score_px))


# combined datasets case
model0 = GPT2LMHeadModel.from_pretrained(model_config.model_name)
model0.resize_token_embeddings(len(tokenizer))
score0 = get_score(model0, 'model0', ds_names, tokenizer, infer_config, rouge_score, max_steps_infer=10)

# model 2: baseline - a GPT2 with trained prompt
model_p = GPT2TrainedPrompt(model_config)
score_p = get_score(model_p, 'model_p', ds_names, tokenizer, infer_config, rouge_score, max_steps_infer=10)

# model 3: "ID-PT"-style - a GPT2 with input-dependent trained prompt
model_px = GPT2TrainedPromptX(model_config)
score_px = get_score(model_px, 'model_px', ds_names, tokenizer, infer_config, rouge_score, max_steps_infer=10)

print('COMBINED SET:\nscore0: ' + str(score0) + '\nscore_p: ' + str(score_p) + '\nscore_px: ' + str(score_px))
