from models.utils import sample
from datasets import load_metric
import torch
from models.trainer import TrainerConfig
from models.dataset import CustomDataset
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from models.models import GPT2TrainedPrompt, GPT2TrainedPromptX

base_model_name = 'distilgpt2'

# define tokenizer, extend with pad token
tokenizer = GPT2TokenizerFast.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def get_val_dl(ds_name, tokenizer, infer_config):
    val_ds = CustomDataset(ds_name,
                           tokenizer,
                           num_examples=infer_config.num_examples_per_test_ds,
                           split_type='test',
                           max_len=infer_config.max_tokenized)
    val_dataloader = DataLoader(val_ds,
                                shuffle=True,
                                batch_size=infer_config.batch_size,
                                num_workers=infer_config.num_workers)
    return val_dataloader


class InferConf:
    batch_size = 1
    num_workers = 1
    max_tokenized = 512
    num_examples_per_test_ds = 800


infer_conf = InferConf()


class ModelConfig:
    model_name = base_model_name
    n_tokens_p0 = 20
    last_tokens_to_keep = 20
    vocab_size = len(tokenizer)


model_config = ModelConfig()

# model = GPT2LMHeadModel.from_pretrained(model_config.model_name)
# model.resize_token_embeddings(len(tokenizer))

model_px = GPT2TrainedPromptX(model_config)

# load state dict to model
model_px.load_state_dict(torch.load('model_px_wiki_qa_state_dict.pt'))

model = model_px

ds_names = ["wiki_qa", "wiki_bio", "samsum"]

ds_name_i = ds_names[0]

val_dataloader = get_val_dl(ds_name_i, tokenizer, infer_conf)
sum_f1 = 0
for i, batch in enumerate(val_dataloader):
    input_ids = batch['source_ids']
    attention_mask = batch['source_mask']

    # find last location in input_ids where attention_mask is 1
    last_loc = torch.where(attention_mask.squeeze() == 1)[-1]
    input_ids = input_ids[:, :last_loc[-1] + 1]
    attention_mask = attention_mask[:, :last_loc[-1] + 1]

    x_out = sample(model, batch, steps=10, tokenizer=tokenizer, temperature=1, sample=True, top_k=10)
    x_out = x_out[0, len(batch['source_ids'][0]):]
    str_pred = ' '.join([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in x_out])

    last_loc_target = torch.where(batch['target_mask'].squeeze() == 1)[-1]
    target_ids = batch['target_ids'][0][:last_loc_target[-1] + 1]
    str_ref = ' '.join([tokenizer.decode(g) for g in target_ids])
    print('pred: ' + str_pred + '\nref:' + str_ref)

    rouge_score = load_metric("rouge")

    scores_gen_true = rouge_score.compute(predictions=[str_pred], references=[str_ref])
    sum_f1 += scores_gen_true['rouge1'].mid.fmeasure

print(sum_f1 / (i + 1))
