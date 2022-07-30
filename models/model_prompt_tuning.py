import os
from pathlib import Path

from transformers import GPT2LMHeadModel, GPTNeoForCausalLM, GPT2Model
import torch
import torch.nn as nn


class GPTPromptTuningMixin:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            weights_path: str = None,
            soft_prompt_path: str = None,
            n_tokens: int = None,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
            vocab_size: int = None,
            **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # resize vocabulary, typically to account for padding token
        # if vocab_size is not None:
        #     model.resize_token_embeddings(vocab_size)

        # Freeze the transformer model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        if weights_path is not None:
            print("Loading weights...")
            model.load_unfrozen_parts(path='model_weights', filename=weights_path)

        return model

    def set_soft_prompt_embeds(
            self,
            soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device(self.device)  # torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
            self,
            n_tokens: int = 20,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.transformer.wte.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, self.config.n_embd)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]

        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")

    def load_unfrozen_parts(self, path: str, filename: str = "unfroze_model.model"):
        self.set_soft_prompt_embeds(soft_prompt_path=os.path.join(path, filename))

    def save_unfrozen_parts(self, path: str, filename: str = "unfrozen_model.model"):
        self.save_soft_prompt(path, filename)

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(self.device)

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class GPTPromptTuningModelMixin:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            soft_prompt_path: str = None,
            n_tokens: int = None,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
            vocab_size: int = None,
            **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        # if vocab_size is not None:
        #     model.resize_token_embeddings(vocab_size)

        # Make sure to freeze transformer model
        for param in model.parameters():
            param.requires_grad = False

        if soft_prompt_path is not None:
            model.set_soft_prompt_embeds(soft_prompt_path)
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )

        return model

    def set_soft_prompt_embeds(
            self,
            soft_prompt_path: str,
    ) -> None:
        """
        Args:
            soft_prompt_path: torch soft prompt file path
        """
        self.soft_prompt = torch.load(
            soft_prompt_path, map_location=torch.device(self.device)  # torch.device("cpu")
        )
        self.n_tokens = self.soft_prompt.num_embeddings
        print(f"Set soft prompt! (n_tokens: {self.n_tokens})")

    def initialize_soft_prompt(
            self,
            n_tokens: int = 20,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
    ) -> None:
        self.n_tokens = n_tokens
        if initialize_from_vocab:
            init_prompt_value = self.wte.weight[
                                :n_tokens].clone().detach()  # self.transformer.wte.weight[:n_tokens].clone().detach()
        else:
            init_prompt_value = torch.FloatTensor(2, 10).uniform_(
                -random_range, random_range
            )
        self.soft_prompt = nn.Embedding(n_tokens, self.config.n_embd)
        # Initialize weight
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids) -> torch.Tensor:
        inputs_embeds = self.wte(input_ids)  # self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]

        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(self.device)

        # if labels is not None:
        #     labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        out_dict = super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            # labels=labels,
            # use_cache=use_cache,
            # return_dict=return_dict,
        )
        # out_dict['new_att_mask'] = attention_mask

        return out_dict, attention_mask


class GPTIDPTMixin:
    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: str,
            weights_path: str = None,
            n_tokens_0: int = None,
            n_tokens_IDPT: int = None,
            initialize_from_vocab: bool = True,
            random_range: float = 0.5,
            vocab_size: int = None,
            **kwargs,
    ):

        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # resize model, originally to include a padding token
        # if vocab_size is not None:
        #     model.resize_token_embeddings(vocab_size)

        # Make sure to freeze the transformer model
        for param in model.parameters():
            param.requires_grad = False

        model.define_IDPT_seq(pretrained_model_name_or_path,
                              n_tokens_0,
                              n_tokens_IDPT,
                              initialize_from_vocab,
                              vocab_size,
                              random_range
                              )

        if weights_path is not None:
            # model.set_soft_prompt_embeds(soft_prompt_path)
            model.load_unfrozen_parts('model_weights', weights_path)

        '''
        elif n_tokens is not None:
            print("Initializing soft prompt...")
            model.initialize_soft_prompt(
                n_tokens=n_tokens,
                initialize_from_vocab=initialize_from_vocab,
                random_range=random_range,
            )
        '''
        return model

    def define_IDPT_seq(self, model_name, n_tokens_0, n_tokens_IDPT, initialize_from_vocab, vocab_size, random_range):
        self.model_prompt = GPT2PromptTuningModel.from_pretrained(
            model_name,
            n_tokens=n_tokens_0,
            initialize_from_vocab=initialize_from_vocab,
            random_range=random_range,
            vocab_size=vocab_size
        )

        self.n_tokens_IDPT = n_tokens_IDPT
        self.model_dim = self.model_prompt.get_input_embeddings().weight.size(1)
        self.self_att = nn.MultiheadAttention(self.model_dim, num_heads=2, dropout=0.1)

    def save_unfrozen_parts(self, path: str, filename: str = "unfrozen_model.model"):
        torch.save({
            'soft_prompt': self.model_prompt.soft_prompt.state_dict(),
            'self_att': self.self_att.state_dict()
        }, os.path.join(path, filename))

    def load_unfrozen_parts(self, path: str, filename: str = "unfrozen_model.model"):
        checkpoint = torch.load(os.path.join(path, filename))

        self.model_prompt.soft_prompt.load_state_dict(checkpoint['soft_prompt'])
        self.self_att.load_state_dict(checkpoint['self_att'])

    def _cat_IDPT_to_input(self, prompt_IDPT, input_ids) -> torch.Tensor:
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([prompt_IDPT, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels_IDPT(self, labels, ignore_index=-100) -> torch.Tensor:
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)
        n_batches = labels.shape[0]

        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens_IDPT), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask_IDPT(self, attention_mask):

        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens_IDPT), 1).to(self.device), attention_mask],
            dim=1,
        )

    '''
    def save_soft_prompt(self, path: str, filename: str = "soft_prompt.model"):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.soft_prompt, os.path.join(path, filename))
        # print(f"Saved soft prompt: {os.path.join(path, filename)}")
        return
    '''
    # this is a search for the first zero in the attention mask (return length if no zero is found)
    # it is necessary when we run a batch and look for last n_tokens_IDPT of the model_prompt run
    def binsearch1(self, t, left, right):
        if left == right:
            return left
        mid = (left + right) // 2
        if t[mid].item() == 1:
            return self.binsearch1(t, mid + 1, right)
        else:
            return self.binsearch1(t, left, mid)

    def cut_model_out_to_prompt(self, hidden_states, attention_mask):
        # get last tokens of hidden states
        tokens_for_IDPT = torch.zeros([attention_mask.shape[0], self.n_tokens_IDPT, self.model_dim]).to(self.device)
        for i in range(attention_mask.shape[0]):
            loc_0 = self.binsearch1(attention_mask[i], 0, attention_mask.shape[1])  # first zero location
            if loc_0 > self.n_tokens_IDPT:
                tokens_for_IDPT[i] = hidden_states[i, (loc_0 - self.n_tokens_IDPT):loc_0, :]
            else:
                raise Exception("Output of model_prompt is too short! Not implemented.")
                # possible to pad with bos/eos token. requires building an embedding for bos/eos
                # extra_bos_pad = self.n_tokens_IDPT - loc_0
                # tokens_for_IDPT[i] = torch.cat([torch.full([extra_bos_pad], eos_id), hidden_states[i][:loc_0]])
        return tokens_for_IDPT

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):

        out_model_prompt, net_attention_mask = self.model_prompt.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )
        out_model_prompt_hidden_states = out_model_prompt["last_hidden_state"]

        prompt_idpt_att_out, attn_output_weights = self.self_att(out_model_prompt_hidden_states,
                                                                 out_model_prompt_hidden_states,
                                                                 out_model_prompt_hidden_states)
        prompt_idpt = self.cut_model_out_to_prompt(prompt_idpt_att_out, net_attention_mask)

        #prompt_idpt = self.cut_model_out_to_prompt(out_model_prompt_hidden_states, net_attention_mask)
        #prompt_idpt_att_out, attn_output_weights = self.self_att(prompt_idpt, prompt_idpt, prompt_idpt)

        inputs_embeds = self._cat_IDPT_to_input(prompt_idpt, input_ids).to(self.device)

        if labels is not None:
            labels = self._extend_labels_IDPT(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask_IDPT(attention_mask).to(self.device)

        # Drop most of the args for now
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            # use_cache=use_cache,
            return_dict=return_dict,
        )


class GPT2PromptTuningLM(GPTPromptTuningMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


# a tiny modification of GPT2PromptTuningLM: using GPT2Model; forward method modified
# we also use a slightly modified Mixin module: GPTPromptTuningModelMixin instead of GPTPromptTuningMixin
class GPT2PromptTuningModel(GPTPromptTuningModelMixin, GPT2Model):
    def __init__(self, config):
        super().__init__(config)

# module for Input-Dependent Prompt Tuning
class GPT2IDPTLM(GPTIDPTMixin, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class GPTNeoPromptTuningLM(GPTPromptTuningMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)
