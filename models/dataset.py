from torch.utils.data import Dataset
import torch
import json
import random
from datasets import load_dataset
import pandas as pd


class CustomDataset(Dataset):
    """
    Creating a custom dataset for reading a dataset, before
    loading it into the dataloader to pass it to the
    neural network for finetuning/inference on the model
    """

    def __init__(self, ds_name, tokenizer, num_examples, split_type, max_tok_len=512, max_char_len=1500):
        """
        Initializes a Dataset class

        Args:
            ds.name (string): Input HuggingFace dataset name
            tokenizer (transformers.tokenizer): Transformers tokenizer
            num_examples (int): number of examples to sample out of the dataset
            split_type (string): 'train'/'test'/'validation' split of the dataset
            max_tok_len (int): maximum length of tokenized data.
            max_tok_len (int): maximum length of characters in example data.
                               Longer examples are rejected. Done before tokenization.
        """
        self.tokenizer = tokenizer
        self.ds_name = ds_name
        self.num_examples = num_examples
        self.split_type = split_type
        self.max_tok_len = max_tok_len
        self.max_char_len = max_char_len

        self.df = self.get_dataset(self.ds_name)

    # In some cases the split names are inconsistent. This deals with that. d - datasetdict, split - split name
    def check_split_name(self, d):
        if self.split_type == "validation":
            if self.split_type in d:
                return self.split_type
            else:
                return 'val'
        return self.split_type

    """
        For a 'name' of HuggingFace dataset among a subset list (samsum, wiki_bio, wiki_qa, ...), 
        this codes and returns the data in terms of 'sources, targets'.
        Since some datasets have several input/output, one should be careful in using this
        to compare to general benchmarks without comparing the protocols first.
    """

    def get_dataset(self, name):
        dataset = load_dataset(name)

        self.split_type = self.check_split_name(dataset)  # check if the split name is consistent w/ dataset definition
        dataset = dataset[self.split_type]  # select the split - train, test, validation

        sources = None  # initialize the list of source text to None, allow exception later
        if name == 'samsum':
            source_feature = 'dialogue'
            target_feature = 'summary'
            sources = dataset[source_feature]
            targets = dataset[target_feature]

            sources = ['Dialogue:\n' + s for s in sources]
            targets = ['\nSummary: ' + t for t in targets]

            full_text = [s + t for s, t in zip(sources, targets)]

        if name == 'wiki_bio':
            sources = [json.dumps(dataset[i]['input_text']) for i in range(len(dataset))]
            targets = dataset['target_text']

            sources = ['Source:\n' + s for s in sources]
            targets = ['\nTarget Wiki: ' + t for t in targets]

            full_text = [s + t for s, t in zip(sources, targets)]

        if name == 'wiki_qa':
            sources = ['Title: ' + dataset[i]['document_title'] + ', Question: ' + dataset[i]['question'] for i in
                       range(len(dataset))]
            targets = dataset['answer']

            sources = ['' + s + '?' if s[-1] != '?' else '' + s for s in sources]  # add question mark on questions w/o
            targets = ['\nAnswer: ' + t for t in targets]

            full_text = [s + t for s, t in zip(sources, targets)]

        if name == 'rotten_tomatoes':
            sources = [dataset[i]['text'] for i in range(len(dataset))]
            targets = ['Positive' if dataset[i]['label'] == 1 else 'Negative' for i in range(len(dataset))]

            sources = ['Review text: ' + s for s in sources]  # add question mark on questions w/o
            targets = ['\nSentiment: ' + t for t in targets]

            full_text = [s + t for s, t in zip(sources, targets)]

        if name == 'quartz':
            sources = [dataset[i]['question'] + ('...' if '.' not in dataset[i]['question'][-2:] else '') +' Options are A: ' + dataset[i]['choices']['text'][0] + ' or B: ' + dataset[i]['choices']['text'][1] for i in range(len(dataset))]
            targets = [dataset[i]['answerKey'] for i in range(len(dataset))]

            sources = ['Complete: ' + s for s in sources]  # add question mark on questions w/o
            targets = ['\nAnswer: ' + t for t in targets]

            full_text = [s + t for s, t in zip(sources, targets)]

        if sources is None:
            raise Exception('Dataset not implemented.')

        df = pd.DataFrame({'input': sources,
                           'output': targets,
                           'full_text': full_text})

        # select in dataframe only entries where length of column 'full_text' is smaller than self.max_char_len
        df = df[df['full_text'].apply(lambda x: len(x) < self.max_char_len)]

        # sample from dataframe num_examples times
        if self.num_examples > len(df):
            print('Warning: number of examples is greater than the number of examples in the dataset. '
                  'Reducing number of examples to the number of examples in the dataset.')
            self.num_examples = len(df)
        df = df.sample(n=self.num_examples).reset_index()

        return df

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.df)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        full_text = self.df['full_text'][index]
        source_text = self.df['input'][index]
        target_text = self.df['output'][index]

        tokenized = self.tokenizer(
            full_text,
            max_length=self.max_tok_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source = self.tokenizer(
            source_text,
            max_length=self.max_tok_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            max_length=self.max_tok_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()

        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        ''' Because the model is autoregressive, we return:
            "input_ids"      - the full text of an example
            "attention_mask" - the attention mask of the full text
            "source_ids"     - the source ids of the data 
            "source_mask"    - the attention mask of source data 
            "target_ids"     - the target ids of the data 
            "target_mask"    - the attention mask of target data
        '''
        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }
