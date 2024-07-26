import torch
import torch.nn as nn
from torch.nn import functional as F


dropout = 0.2

class Head(nn.Module):
    def __init__(self, head_size):
        pass


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        pass


class FeedForward(nn.Module):
    """Feedforward network"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        ])

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        pass

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

