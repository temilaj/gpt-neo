import torch.nn as nn

from config import get_config

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        config = get_config()
        dropout = config['dropout']
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # projection back into the redidual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)