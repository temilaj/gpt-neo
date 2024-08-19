import torch.nn as nn

from layers.feed_forward import FeedFoward
from layers.multi_head_attention import MultiHeadAttention

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        # self attenion head: Communication
        self.sa = MultiHeadAttention(n_head, head_size)
        # Feedfoward layer: computation
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # residual connection
        # add feed foward after self attention before calculating logits
        # So after self attention (communication), the tokens can "think" on that data individually on a per-token level
        x = x + self.ffwd(self.ln2(x))  # residual connection
        return x