import torch
import torch.nn.functional as F
import numpy as np

def _predict_next(model, w1, w2, word2idx):
    if hasattr(model, 'get_prob'):
        # Count-based model
        probs = []
        vocab_size = model.vocab_size
        for i in range(vocab_size):
            p = model.get_prob(w1, w2, i)
            probs.append(p)
        probs = torch.tensor(probs)
        return probs / torch.sum(probs)
    else:
        # Neural model
        x = torch.tensor([[w1, w2]])
        out = model(x)
        probs = F.softmax(out, dim=-1)[0]
        return probs

def generate_greedy(model, w1, w2, word2idx, idx2word, num_words=80):
    words = [idx2word[w1], idx2word[w2]]
    curr_w1, curr_w2 = w1, w2
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad() if hasattr(model, 'eval') else torch.enable_grad():
        for _ in range(num_words):
            probs = _predict_next(model, curr_w1, curr_w2, word2idx)
            next_w = torch.argmax(probs).item()
            words.append(idx2word[next_w])
            curr_w1, curr_w2 = curr_w2, next_w
    return " ".join(words)

def generate_beam_search(model, w1, w2, word2idx, idx2word, num_words=80, beam_width=3):
    curr_w1, curr_w2 = w1, w2
    beams = [(0.0, [curr_w1, curr_w2])]
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad() if hasattr(model, 'eval') else torch.enable_grad():
        for _ in range(num_words):
            new_beams = []
            for score, seq in beams:
                probs = _predict_next(model, seq[-2], seq[-1], word2idx)
                log_probs = torch.log(probs + 1e-10)
                top_log_probs, top_indices = torch.topk(log_probs, beam_width)
                
                for lp, idx in zip(top_log_probs, top_indices):
                    new_score = score + lp.item()
                    new_seq = seq + [idx.item()]
                    new_beams.append((new_score, new_seq))
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

    best_seq = beams[0][1]
    return " ".join([idx2word[idx] for idx in best_seq])

def generate_top_k(model, w1, w2, word2idx, idx2word, num_words=80, k=5):
    words = [idx2word[w1], idx2word[w2]]
    curr_w1, curr_w2 = w1, w2
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad() if hasattr(model, 'eval') else torch.enable_grad():
        for _ in range(num_words):
            probs = _predict_next(model, curr_w1, curr_w2, word2idx)
            top_probs, top_indices = torch.topk(probs, k)
            top_probs = top_probs / torch.sum(top_probs)
            sampled_idx = torch.multinomial(top_probs, 1).item()
            next_w = top_indices[sampled_idx].item()
            words.append(idx2word[next_w])
            curr_w1, curr_w2 = curr_w2, next_w
    return " ".join(words)

def generate_nucleus(model, w1, w2, word2idx, idx2word, num_words=80, p=0.9):
    words = [idx2word[w1], idx2word[w2]]
    curr_w1, curr_w2 = w1, w2
    if hasattr(model, 'eval'):
        model.eval()
    with torch.no_grad() if hasattr(model, 'eval') else torch.enable_grad():
        for _ in range(num_words):
            probs = _predict_next(model, curr_w1, curr_w2, word2idx)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0
            probs = probs / torch.sum(probs)
            
            next_w = torch.multinomial(probs, 1).item()
            words.append(idx2word[next_w])
            curr_w1, curr_w2 = curr_w2, next_w
    return " ".join(words)
