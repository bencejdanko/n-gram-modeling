import re
from datasets import load_dataset
import torch
from torch.utils.data import Dataset

def get_jefferson_text():
    dataset = load_dataset('khaihernlow/us-state-of-the-union-addresses-1790-2019', split='train')
    docs = []
    for row in dataset:
        vals = [str(v) for v in row.values() if v]
        if any("Jefferson" in v for v in vals):
            if 'text' in row:
                docs.append(row['text'])
            elif 'speech' in row:
                docs.append(row['speech'])
            elif 'content' in row:
                docs.append(row['content'])
            else:
                docs.append(str(row))
    return " ".join(docs)

def preprocess(text: str) -> list:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\']', '', text)
    tokens = text.split()
    return tokens

def build_vocab(tokens):
    vocab = sorted(list(set(tokens)))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}
    return vocab, word2idx, idx2word

class TrigramDataset(Dataset):
    def __init__(self, tokens, word2idx):
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
