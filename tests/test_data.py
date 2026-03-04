import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import preprocess, build_vocab

def test_preprocess():
    text = "Hello, world! This is Thomas Jefferson's test.   Multiple    spaces."
    tokens = preprocess(text)
    expected = ['hello', 'world', 'this', 'is', 'thomas', "jefferson's", 'test', 'multiple', 'spaces']
    assert tokens == expected

def test_build_vocab():
    tokens = ['hello', 'world', 'hello']
    vocab, word2idx, idx2word = build_vocab(tokens)
    assert vocab == ['hello', 'world']
    assert word2idx['hello'] == 0
    assert word2idx['world'] == 1
    assert idx2word[0] == 'hello'
    assert idx2word[1] == 'world'
