import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 256

dropout = 0.2
n_embd = 384
n_head = 6
n_layer = 6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 56


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B,T,C = x.shape
        k = self.key(x) # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        # k.T --> (B, hs, T)
        # q @ k.T --> (B, T, T)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei)

        v = self.value(x) # --> (B, T, hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) --> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size=head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size*num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([ h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out

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

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads=n_head, head_size=n_embd//n_head)
        self.ffwd = FeedForward(n_embd=n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.t_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.p_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential([
            Block(n_embd, n_head=n_head) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lln = nn.Linear(n_embd*n_head, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.t_embedding_table(idx) # (B, T, C)
        pos_emb = self.p_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lln(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1)
            idx = torch.concat((idx, idx_next), dim=1)
        
        return idx