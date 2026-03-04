import torch
import torch.nn as nn
from collections import defaultdict
import math

class CountTrigramModel:
    def __init__(self, vocab_size, add_k=1.0):
        self.vocab_size = vocab_size
        self.add_k = add_k
        self.bigram_counts = defaultdict(int)
        self.trigram_counts = defaultdict(int)
        
    def train(self, tokens_idx):
        for i in range(len(tokens_idx) - 2):
            w1, w2, w3 = tokens_idx[i], tokens_idx[i+1], tokens_idx[i+2]
            self.bigram_counts[(w1, w2)] += 1
            self.trigram_counts[(w1, w2, w3)] += 1
            
    def get_prob(self, w1, w2, w3):
        # P(w3 | w1, w2) with Add-k smoothing
        count_tri = self.trigram_counts.get((w1, w2, w3), 0)
        count_bi = self.bigram_counts.get((w1, w2), 0)
        return (count_tri + self.add_k) / (count_bi + self.add_k * self.vocab_size)
    
    def get_log_prob(self, w1, w2, w3):
        return math.log(self.get_prob(w1, w2, w3))

class NeuralTrigramModel(nn.Module):
    def __init__(self, vocab_size, embed_size=64, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(embed_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x):
        # x is (batch, 2)
        embeds = self.embedding(x) # (batch, 2, embed_size)
        embeds = embeds.view(x.size(0), -1) # (batch, 2 * embed_size)
        out = self.fc1(embeds)
        out = self.relu(out)
        logits = self.fc2(out) # (batch, vocab_size)
        return logits
