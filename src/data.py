import re
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

# fetches jefferson text from dataset
def get_jefferson_text():
    dataset = load_dataset('khaihernlow/us-state-of-the-union-addresses-1790-2019', split='train')
    docs = []
    # filter explicitly for thomas jefferson
    for row in dataset:
        if row.get('President') == 'Thomas Jefferson':
            text_val = row.get('Text')
            docs.append(text_val)
    return " ".join(docs)

# clean and tokenize input text
def preprocess(text: str) -> list:
    # lowercase text
    text = text.lower()
    # normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # remove punctuation 
    text = re.sub(r'[^\w\s\']', '', text)
    tokens = text.split()
    return tokens

def build_vocab(tokens):
    vocab = sorted(list(set(tokens)))
    # create mappings for vocabulary
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

class TrigramDataset(Dataset):
    def __init__(self, tokens, word2idx):
        # generate trigram pairs and targets
        self.X = []
        self.y = []
        for i in range(len(tokens) - 2):
            w1 = word2idx[tokens[i]]
            w2 = word2idx[tokens[i+1]]
            w3 = word2idx[tokens[i+2]]
            self.X.append((w1, w2))
            self.y.append(w3)
            
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])
