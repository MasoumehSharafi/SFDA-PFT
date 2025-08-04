# --- src/utils.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct / len(labels)
