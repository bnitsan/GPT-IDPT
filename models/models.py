import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import GPT2Model, GPT2LMHeadModel


# based on https://github.com/kipgparker/soft-prompt-tuning/blob/main/soft_embedding.py
class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 20,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends soft learned embedding to input of a decoder model
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 20.
            random_range (float, optional): numeric range to init embedding (if not init. from vocab). Default: 0.5.
            initialize_from_vocab (bool, optional): initializes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.random_range = random_range
        self.initialize_from_vocab = initialize_from_vocab
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding())

    def initialize_embedding(self):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if self.initialize_from_vocab:
            return self.wte.weight[:self.n_tokens].clone().detach()
        return torch.FloatTensor(self.n_tokens, self.wte.weight.size(1)).uniform_(-self.random_range, self.random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        # find first pad token index in tokens torch tensor
        pad_index = tokens.eq(self.padding_idx).nonzero().item()

        print(tokens.shape)
        input_embedding = self.wte(tokens[:, self.n_tokens:])  # here we run over the first n_tokens
        print(input_embedding.shape)
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        print(learned_embedding.shape)
        print(torch.cat([learned_embedding, input_embedding], 1).shape)
        return torch.cat([learned_embedding, input_embedding], 1)


"""
    A module of a GPT2 LM model with a soft learned prompt
    Written by NB 4/7/22
"""


class GPT2TrainedPrompt(nn.Module):
    def __init__(self, config):
        """appends learned embedding to a GPT2LMHeadModel
        Args:
            Config
        Returns:
            Model class
        """
        super(GPT2TrainedPrompt, self).__init__()
        self.config = config

        self.model = GPT2LMHeadModel.from_pretrained(config.model_name)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.resize_token_embeddings(self.config.vocab_size)

        self.model_dim = self.model.get_input_embeddings().weight.size(1)

        self.soft_embed = SoftEmbedding(self.model.get_input_embeddings(),
                                        n_tokens=config.n_tokens_p0,
                                        initialize_from_vocab=False)
        self.model.set_input_embeddings(self.soft_embed)

    def initialize_embedding(self):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if self.initialize_from_vocab:
            return self.wte.weight[:self.n_tokens].clone().detach()
        return torch.FloatTensor(self.n_tokens, self.wte.weight.size(1)).uniform_(-self.random_range, self.random_range)

    def forward(self, input_ids, attention_mask=None, labels=None, return_dict=False):
        """run forward pass
        Args:
            x (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        lm_logits = transformer_outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            raise NotImplementedError("return_dict=True is not implemented.")


"""
    A module of a GPT2 LM model with a tuned prompt p(x), similar to the one discussed in Levine+(2022).
    Here, in the generation of p(x) we use the same model as the general LM model we use.

    Written by NB 4/7/22
"""


class GPT2TrainedPromptX(nn.Module):
    def __init__(self, config):
        super(GPT2TrainedPromptX, self).__init__()
        self.config = config

        # "last tokens to keep" - amount of tokens to pass from initial LM pass in estimating p(x)
        self.last_tokens_to_keep = self.config.last_tokens_to_keep

        # load GPT2 models and freeze all their parameters
        self.model_prompt = GPT2Model.from_pretrained(config.model_name)  # model used in the prompt generation
        self.model = GPT2LMHeadModel.from_pretrained(config.model_name)  # model used after prompt
        for param in self.model_prompt.parameters():
            param.requires_grad = False
        for param in self.model.parameters():
            param.requires_grad = False

        # resize tokenizer to include the new tokens (pad token by default)
        self.model.resize_token_embeddings(self.config.vocab_size)
        self.model_prompt.resize_token_embeddings(self.config.vocab_size)

        self.model_dim = self.model_prompt.get_input_embeddings().weight.size(1)

        # define a soft trainable embedding, acting as a learned prompt
        self.soft_embed = SoftEmbedding(self.model_prompt.get_input_embeddings(),
                                        n_tokens=config.n_tokens_p0,
                                        initialize_from_vocab=False)
        self.model_prompt.set_input_embeddings(self.soft_embed)

        # single attention layer; may extend later for sequential, multiple heads, etc.
        self.self_att = nn.MultiheadAttention(self.model_dim, num_heads=1, dropout=0.1)

    def initialize_embedding(self):
        """initializes learned embeddings
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if self.initialize_from_vocab:
            return self.wte.weight[:self.n_tokens].clone().detach()
        return torch.FloatTensor(self.n_tokens, self.wte.weight.size(1)).uniform_(-self.random_range, self.random_range)

    def forward(self, input_ids, attention_mask, labels=None, return_dict=False):
        """run forward pass
        Args:
            x (torch.long): input tokens before encoding
        Returns:
            logits (and loss if provided labels) of GPT2LMHeadModel prompted with 
            an input-dependent prompt p(x)
        """

        '''
            First part: generate p(x)
        '''
        prompt_transformer_outputs = self.model_prompt(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
        hidden_states = prompt_transformer_outputs[0]

        p_x = hidden_states[:, -self.last_tokens_to_keep:, :]  # Batch X (Tokens + last_tokens_to_keep) X n_embd

        '''
            Second part: embed x and concatenate the embedding with p(x): [p(x) x]
        '''
        input_ids_emb = self.model.transformer.wte(input_ids)  # embed input

        new_input_emb = torch.cat([p_x, input_ids_emb], 1)  # Batch X (Tokens + last_tokens_to_keep) X n_embd

        att_mask_px = torch.ones(attention_mask.shape[0], self.last_tokens_to_keep).to(
            attention_mask.device)  # self.last_tokens_to_keep

        new_attention_mask = torch.cat([att_mask_px, attention_mask], 1)

        transformer_outputs = self.model(
            attention_mask=new_attention_mask,
            inputs_embeds=new_input_emb,
            return_dict=return_dict,
        )

        '''
            Third part: compare logits to labels if existing, compute loss
        '''

        lm_logits = transformer_outputs[0]

        # Set device for model parallelism
        # if self.model_parallel:
        #     torch.cuda.set_device(self.transformer.first_device)
        #     hidden_states = hidden_states.to(self.lm_head.weight.device)

        loss = None
        if labels is not None:
            # extend labels to account for p(x) tokens
            labels_px = torch.zeros(labels.shape[0], self.last_tokens_to_keep).to(labels.device) + 0.5
            labels = torch.cat([labels_px, labels], 1)
            labels = labels.long()

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,)  # + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            raise NotImplementedError("return_dict=True is not implemented.")
