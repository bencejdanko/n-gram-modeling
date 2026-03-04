import torch
import math
import torch.nn as nn

def calculate_perplexity_count(model, test_tokens_idx):
    log_prob_sum = 0
    N = len(test_tokens_idx) - 2
    if N <= 0:
        return float('inf')
        
    for i in range(N):
        w1, w2, w3 = test_tokens_idx[i], test_tokens_idx[i+1], test_tokens_idx[i+2]
        prob = model.get_prob(w1, w2, w3)
        if prob == 0:
            return float('inf')
        log_prob_sum += math.log(prob)
    
    avg_log_prob = log_prob_sum / N
    perplexity = math.exp(-avg_log_prob)
    return perplexity

def calculate_perplexity_neural(model, test_loader, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0
    total_items = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item()
            total_items += X_batch.size(0)
            
    if total_items == 0:
        return float('inf')
    avg_loss = total_loss / total_items
    perplexity = math.exp(avg_loss)
    return perplexity
