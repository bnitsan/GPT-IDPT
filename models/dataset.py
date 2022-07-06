from torch.utils.data import Dataset
import torch
import json
import random
from datasets import load_dataset
random.seed(0)


class CustomDataset(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model
    """
    
    def __init__(self, ds_name, tokenizer, num_examples, split_type, max_len=512):
        """
        Initializes a Dataset class

        Args:
            ds.name (string): Input HuggingFace dataset name
            tokenizer (transformers.tokenizer): Transformers tokenizer
            num_examples (int): number of examples to sample out of the dataset
            split_type (string): 'train'/'test'/'validation' split of the dataset
            max_len (int): maximum length of tokenized data
        """
        self.tokenizer = tokenizer
        self.ds_name = ds_name
        self.num_examples = num_examples
        self.split_type = split_type
        self.max_len = max_len

        self.full_texts, self.source_texts, self.target_texts = self.get_dataset(self.ds_name)

        random_selection = random.sample(range(len(self.source_texts)), self.num_examples)
        self.source_texts = [self.source_texts[i] for i in random_selection]
        self.target_texts = [self.target_texts[i] for i in random_selection]
        self.full_texts = [self.full_texts[i] for i in random_selection]

    """
        For a 'name' of HuggingFace dataset among a subset list (samsum, wiki_bio, wiki_qa, ...), 
        this codes and returns the data in terms of 'sources, targets'.
        Since some datasets have several input/output, one should be careful in using this
        to compare to general benchmarks without comparing the protocols first.
    """
    def get_dataset(self, name):
        dataset = load_dataset(name)
        dataset = dataset[self.split_type]
        if name == 'samsum':
            source_feature = 'dialogue'
            target_feature = 'summary'
            sources = dataset[source_feature]
            targets = dataset[target_feature]

            sources = ['Dialogue:\n'+ s for s in sources]
            targets = ['\nSummary: '+t for t in targets]

            full_text = [s+t for s,t in zip(sources,targets)]
            return full_text, sources, targets

        if name == 'wiki_bio':
            sources = [json.dumps(dataset[i]['input_text']) for i in range(len(dataset))]
            targets = dataset['target_text']
            
            sources = ['Source:\n'+ s for s in sources]
            targets = ['\nTarget Wiki: '+t for t in targets]
            
            full_text = [s+t for s,t in zip(sources,targets)]
            return full_text, sources, targets

        if name == 'wiki_qa':
            sources = ['Title: ' + dataset[i]['document_title'] + ', Question: ' + dataset[i]['question'] for i in range(len(dataset))]
            targets = dataset['answer']

            sources = [''+ s for s in sources]
            targets = ['\nAnswer: '+t for t in targets]
            
            full_text = [s+t for s,t in zip(sources,targets)]
            return full_text, sources, targets

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.target_texts)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        full_text = self.full_texts[index]
        source_text = self.source_texts[index]
        target_text = self.target_texts[index]

        tokenized = self.tokenizer( 
            full_text, 
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source = self.tokenizer( 
            source_text, 
            max_length=self.max_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer(
            target_text,
            max_length=self.max_len,
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

        ''' Because the model is autoregressive, we return:
            "input_ids"      - the full text of an example
            "attention_mask" - the attention mask of the full text
            "source_ids"     - the source ids of the data 
            "source_mask"    - the attention mask of source data 
            "target_ids"     - the target ids of the data 
        '''
        return {
            "input_ids": input_ids.to(dtype=torch.long),
            "attention_mask": attention_mask.to(dtype=torch.long),
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }
