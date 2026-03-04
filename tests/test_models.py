import sys
import os
import torch
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import CountTrigramModel, NeuralTrigramModel
from eval import calculate_perplexity_count, calculate_perplexity_neural

def test_count_model():
    model = CountTrigramModel(vocab_size=3, add_k=1.0)
    # vocab: 0, 1, 2
    # sequence: 0, 1, 2, 0, 1, 2
    tokens = [0, 1, 2, 0, 1, 2]
    model.train(tokens)
    
    # trigrams: (0, 1, 2) x 2, (1, 2, 0) x 1, (2, 0, 1) x 1
    # bigrams: (0, 1) x 2, (1, 2) x 2, (2, 0) x 1
    
    # prob of 2 | 0, 1 -> count(0, 1, 2)=2 + k=1 / count(0, 1)=2 + k*v=3 = 3/5 = 0.6
    p = model.get_prob(0, 1, 2)
    assert math.isclose(p, 0.6)

def test_neural_model():
    model = NeuralTrigramModel(vocab_size=10, embed_size=8, hidden_size=16)
    x = torch.tensor([[1, 2], [3, 4]])
    out = model(x)
    assert out.shape == (2, 10)
